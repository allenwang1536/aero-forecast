import torch
import os
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import WeatherForecasterCNNLSTM, WeatherForecasterCNNTransformer, TemporalConvNet2D, ConvLSTM
from typing import Optional
from tqdm import tqdm
from visualize import visualize_results

def main():
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    input_size = 1
    hidden_size_cnnlstm = 5
    hidden_size_cnntransformer = 5
    hidden_size_tcn = 5
    hidden_size_clstm = 5
    num_layers = 2
    output_size = 12 * 4 * 4
    kernel_size = 3
    dropout = 0.5
    learning_rate = 0.0005
    num_epochs = 1
    num_features = 12


    # Make each of the 4 models
    CNNLSTM = WeatherForecasterCNNLSTM(num_features, hidden_size_cnnlstm, num_layers, output_size, kernel_size, dropout).to(device)
    CNNTransformer = WeatherForecasterCNNTransformer(num_features, hidden_size_cnntransformer, num_layers, output_size, kernel_size, dropout).to(device)
    TCN = TemporalConvNet2D(12, num_features, kernel_size, dropout).to(device)
    CLSTM = ConvLSTM(num_features, 120, kernel_size, num_layers).to(device)

    # Dataset
    train_dataset = torch.load('./data/train_dataset_norm_simple.pth')
    test_dataset = torch.load('./data/test_dataset_norm_simple.pth')

    # Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Loss and optimizers for each model
    criterion = nn.MSELoss()

    # Lists to store training and test losses
    train_losses = []
    test_losses = [0, 0, 0, 0]
    model_names = ['CNNLSTM', 'CNNTransformer', 'TCN', 'CLSTM']

    ## For each model: Instantiate, define optimizer, train, test loss

    models = {
        'CNNLSTM': {
            'model_path': 'cnn-lstm-model.pth',
            'model': CNNLSTM,
            'optimizer': Adam(CNNLSTM.parameters(), lr=learning_rate),
            'index': 0
        },
        'CNNTransformer': {
            'model_path': 'cnn-transformer-model.pth',
            'model': CNNTransformer,
            'optimizer': Adam(CNNTransformer.parameters(), lr=learning_rate),
            'index': 1
        },
        'CLSTM': {
            'model_path': 'clstm-model.pth',
            'model': CLSTM,
            'optimizer': Adam(CLSTM.parameters(), lr=learning_rate),
            'index': 3
        }
    }

    for model_name, model_info in models.items():
      model_path = model_info['model_path']
      model = model_info['model']
      model_optimizer = model_info['optimizer']
      index = model_info['index']

      if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
      else:
        train_losses.append(train(model, train_data_loader, model_optimizer, criterion, device, num_epochs))
        torch.save(model.state_dict(), model_path)
      test_losses[index] = test(model, test_data_loader, criterion, device)


    # CNNLSTM = WeatherForecasterCNNLSTM(num_features, hidden_size_cnnlstm, num_layers, output_size, kernel_size, dropout).to(device)
    # cnnlstm_optimizer = Adam(CNNLSTM.parameters(), lr=learning_rate)
    # train_losses.append(train(CNNLSTM, train_data_loader, cnnlstm_optimizer, criterion, device, num_epochs))
    # test_losses[0] = test(CNNLSTM, test_data_loader, criterion, device)

    # CNNTransformer = WeatherForecasterCNNTransformer(num_features, hidden_size_cnntransformer, num_layers, output_size, kernel_size, dropout).to(device)
    # cnntransformer_optimizer = Adam(CNNTransformer.parameters(), lr=learning_rate)
    # train_losses.append(train(CNNTransformer, train_data_loader, cnntransformer_optimizer, criterion, device, num_epochs))
    # # test_losses[1] = test(CNNTransformer, test_data_loader, criterion, device)

    # TCN = TemporalConvNet2D(12, num_features, kernel_size, dropout).to(device)
    # tcn_optimizer = Adam(TCN.parameters(), lr=learning_rate)
    # train_losses.append(train(TCN, train_data_loader, tcn_optimizer, criterion, device, num_epochs))
    # test_losses[2] = test(TCN, test_data_loader, criterion, device)

    # CLSTM = ConvLSTM(num_features, 120, kernel_size, num_layers).to(device)
    # clstm_optimizer = Adam(CLSTM.parameters(), lr=learning_rate)
    # train_losses.append(train(CLSTM, train_data_loader, clstm_optimizer, criterion, device, num_epochs))
    # test_losses[3] = test(CLSTM, test_data_loader, criterion, device)

    # Visualize the results
    visualize_results(train_losses, test_losses, model_names, num_epochs)


def train(model, dataloader, optimizer, criterion, device, num_epochs):
    model.train()
    losses = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            if isinstance(model, ConvLSTM):
                outputs, _ = model(inputs)  # ConvLSTM returns a tuple
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # progress bar to see the loss as the model trains
            progress_bar.set_postfix({'loss': f'{running_loss / len(dataloader):.4f}'})
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")
        losses.append(running_loss/len(dataloader))
    return losses

def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_correct, total_samples = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if isinstance(model, ConvLSTM):
                outputs, _ = model(inputs)  # ConvLSTM returns a tuple
            else:
                outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            num_correct += accuracy_within_tolerance(outputs, labels)
            total_samples += len(inputs)
    accuracy = (num_correct / total_samples).item()

    print('Test Accuracy: ', accuracy)
    return running_loss / len(dataloader)

def predict_sequence(num_steps: int, model, data_index):
    # Assume data has dimensions [batch_size, sequence_length, channels, height, width]
    original_dataset = torch.load('./data/original_dataset_norm_simple.pth')
    if data_index < 120 or data_index >= len(original_dataset) - 120:
        return None
    # The idea here is to predict for one input, put the prediction into the data as if it was the next data point, then predict again for num_steps
    produced_outputs = []
    corresponding_labels = []
    for step in range(num_steps):
        index = data_index + step
        default_intervals = [-120, -56, -28, -12, -8, -4, -3, -2, -1, 0, 4]
        for i in default_intervals:
            i += index
        model.eval()
        with torch.no_grad():
            output = model(original_dataset[default_intervals])
            produced_outputs.append(output)
            corresponding_labels.append(original_dataset[index])
            original_dataset[index] = output
    return produced_outputs, corresponding_labels

def sequence_prediction_accuracy(num_steps: int, model):
    # Assume data has dimensions [batch_size, sequence_length, channels, height, width]
    original_dataset = torch.load('./data/original_dataset_norm_simple.pth')
    model.eval()
    accuracies = []
    for index in range(120, len(original_dataset) - 120):
        produced_outputs, corresponding_labels = predict_sequence(num_steps, model, index)
        if produced_outputs is not None:
            # Calculate accuracy
            accuracy = 0
            accuracies.append(accuracy)
    return sum(accuracies) / len(accuracies)   


def accuracy_within_tolerance(predictions, labels, tolerance=0.5, required_fraction=0.8):
    batch_size, _, num_features, height, width = predictions.shape
    
    differences = torch.abs(predictions - labels)
    within_tolerance = differences < tolerance
    
    num_elements = num_features * height * width
    fractions_within_tolerance = torch.sum(within_tolerance, axis=(1, 2, 3, 4)) / num_elements
    
    count = torch.sum(fractions_within_tolerance >= required_fraction)
    
    return count

if __name__ == "__main__":
    main()