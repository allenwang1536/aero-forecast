import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Type, Union


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size):
        super(ConvLSTMCell, self).__init__()

        # Initialize the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # Convolutional layers for the input, forget, output, and cell state
        self.conv = nn.Conv2d(
            in_channels=input_size + hidden_size,
            out_channels=4 * hidden_size,
            kernel_size=kernel_size,
            padding=self.padding,
        )

    # Define the forward pass
    def forward(self, input, prev_state):
        h_prev, c_prev = prev_state

        # Ensure both input and previous states are on the same device
        device = input.device
        h_prev = h_prev.to(device)
        c_prev = c_prev.to(device)

        # Concatenate along the channel axis
        combined = torch.cat((input, h_prev), dim=1)
        combined_conv = self.conv(combined)

        # Split the combined convolutional tensor into separate input, forget, output, and cell state convolutions
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_size, dim=1)

        # Apply the activation functions
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Update the cell state and hidden state
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_size=input_size if i == 0 else hidden_size,
                hidden_size=hidden_size,
                kernel_size=kernel_size
            )
            for i in range(num_layers)
        ])
        # Add a fully connected layer or output layer if needed
        self.fc = nn.Linear(1920, 12 * 4 * 4)

    def forward(self, input, prev_state=None):
        # Retrieve the device of the input tensor
        device = input.device

        batch_size, seq_len, channels, height, width = input.size()
        if prev_state is None:
            # Initialize the previous states on the same device as the input
            prev_state = [
                (torch.zeros(batch_size, self.hidden_size, height, width, device=device),
                 torch.zeros(batch_size, self.hidden_size, height, width, device=device))
                for _ in range(self.num_layers)
            ]
        else:
            # Ensure all previous states are on the input device
            prev_state = [
                (h_prev.to(device), c_prev.to(device))
                for h_prev, c_prev in prev_state
            ]

        next_state = prev_state
        for t in range(seq_len):
            x_t = input[:, t]  # [batch_size, channels, height, width]
            for i, cell in enumerate(self.cells):
                h_prev, c_prev = next_state[i]
                h_next, c_next = cell(x_t, (h_prev, c_prev))
                next_state[i] = (h_next, c_next)
                x_t = h_next  # Pass to the next layer

        # Use the final state of the last layer as output
        final_output = next_state[-1][0]  # Final hidden state
        out = self.fc(final_output.view(batch_size, -1))
        out = out.view(batch_size, 1, channels, height, width)
        return out, next_state

class TemporalBlock2D(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock2D, self).__init__()

        # Convolutional layer with weight normalization
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.relu = nn.ReLU()

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Convolutional layer with weight normalization
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))

        # Sequential network
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout,
                                 self.conv2, self.relu, self.dropout)

        # Downsample layer
        self.downsample = nn.Conv2d(
            n_inputs, n_outputs, kernel_size=1, stride=stride) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize weights with a normal distribution since we are using weight normalization
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # Pass the input through the network
        out = self.net(x)
        # Add the input to the output (residual connection)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet2D(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet2D, self).__init__()
        layers = []
        num_levels = num_channels

        
        # Create a series of TemporalBlock2D layers
        for i in range(num_levels):
            # Calculate the dilation size
            dilation_size = 2 ** i
            # Create a TemporalBlock2D layer
            in_channels = num_inputs if i == 0 else num_channels
            out_channels = num_channels
            # Append the TemporalBlock2D layer to the list of layers
            layers += [TemporalBlock2D(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=(kernel_size // 2) * dilation_size, dropout=dropout)]

        # Create a sequential network
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        num_samples, seq_length, num_features, lat_size, lon_size = x.size()
        x = x.permute(1, 0, 2, 3, 4)

        convoluted_grids = []
        for grid in x:
            convoluted_grid = self.network(grid)
            # Flatten only the feature dimension, not the spatial dimensions
            convoluted_grids.append(convoluted_grid.view(convoluted_grid.size(0), -1, lat_size, lon_size))

        # Stack along the sequence dimension and then swap dimensions to [batch_size, sequence, features, lat, lon]
        output = torch.stack(convoluted_grids, dim=0).permute(1, 0, 2, 3, 4)  # [seq_length, batch_size, channels, lat, lon] -> [batch_size, seq_length, channels, lat, lon]
        return output

class WeatherForecasterCNNLSTM(nn.Module):
    def __init__(self, grid_features, hidden_size, num_layers, output_size, kernel_size=3, dropout=0.2):
        super(WeatherForecasterCNNLSTM, self).__init__()
        self.grid_features = grid_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.kernel_size = kernel_size

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(grid_features, hidden_size, kernel_size, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(hidden_size * 4 * 4, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
    
        # Assume x has dimensions [batch_size, sequence_length, channels, height, width]
        batch_size, seq_length, grid_data_length, grids_x, grids_y = x.size()

        # Separate all of the grids in the sequence
        convoluted_grids = []
        for grid in x:
            # Pass through CNN layers
            convoluted_grid = self.cnn(grid)
            convoluted_grids.append(convoluted_grid.flatten())

        output_sequence, (final_hidden_state, final_cell_state) = self.lstm(torch.reshape(torch.stack(convoluted_grids),  (batch_size, seq_length, -1)))
        
        lstm_out = output_sequence[:, -1, :]
        out = self.fc(lstm_out)
        out = torch.reshape(out, (batch_size, 1, grid_data_length, grids_x, grids_y))

        return out


class WeatherForecasterCNNTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3, dropout=0.2):
        super(WeatherForecasterCNNTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.kernel_size = kernel_size

        # Adjusted the input channel size to 12 if needed
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size, hidden_size, kernel_size, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_size, hidden_size, kernel_size, padding='same'),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer layer
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size * 4 * 4, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=num_layers)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size*4*4, output_size)

    def forward(self, x):
        batch_size, seq_length, grid_data_length, grids_x, grids_y = x.size()

        # Separate all of the grids in the sequence
        convoluted_grids = []
        for grid in x:
            # Pass through CNN layers
            convoluted_grid = self.cnn(grid)
            convoluted_grids.append(convoluted_grid.flatten())

        CNN_out = torch.reshape(torch.stack(convoluted_grids), (batch_size, seq_length, -1)).permute(1,0,2)
        transformer_out = self.transformer(CNN_out)
        transformer_out = transformer_out[-1]
        
        out = self.fc(transformer_out)
        out = torch.reshape(out, (batch_size, 1, grid_data_length, grids_x, grids_y))

        return out
