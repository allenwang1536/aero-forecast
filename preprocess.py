import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr

selection = {
    'variables': [
        'geopotential',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'vertical_velocity',
        '10m_wind_speed',
        'total_precipitation_6hr',
        'total_cloud_cover',
        '2m_temperature',
        'specific_humidity',
        'surface_pressure',
        'toa_incident_solar_radiation',
        'total_column_water_vapour'
    ],
    "levels": [500],
    "time_slice": slice('2016-01-01', '2017-01-01'),
    "lat_slice": slice(30,50),
    "long_slice": slice(70,90),
}


def preprocess_data(split_percentage=0.8, batch_size=20, use_level=False,  window_size=10):
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
    data = xr.open_zarr(obs_path)

    print("preprocessing")
    print("dataset shape: ", data.sizes)

    data = data[selection['variables']].sel(level=selection['levels'], time=selection['time_slice'],
                                            latitude=selection['lat_slice'], longitude=selection['long_slice'])

    time_size = data.sizes['time']
    level_size = data.sizes['level']
    lon_size = data.sizes['longitude']
    lat_size = data.sizes['latitude']
    
    feature_size = len(selection['variables'])

    
    dataset_shape = (time_size, level_size, lon_size, lat_size) if use_level else (time_size, lon_size, lat_size)
    time_arrays = []
    for i, var_name in enumerate(data.data_vars):
        var_data = data[var_name].values

        num_dims = len(var_data.shape)
        time_array = np.empty(dataset_shape, dtype=var_data.dtype)
        has_level_size = num_dims == 4
        if use_level:
            if has_level_size:
                time_array[:] = var_data
            else:
                time_array[:] = np.expand_dims(var_data, axis=1)
        else:
            if has_level_size:
                time_array[:] = var_data[:, 0, :, :]
            else:
                time_array[:] = var_data

        
        time_arrays.append(time_array)

    dataset = np.stack(time_arrays, axis=-1)
    print("processed dataset shape:", dataset.shape)

    #processing to time series
    default_intervals = [-120, -56, -28, -12, -8, -4, -3, -2, -1, 0, 4]
    #todo: handle longer window size

    inputs = []
    labels = []
    for i in range(len(dataset)):
        sequence = []
        for interval in default_intervals:
            index = i+interval
            if 0 <= index < len(dataset):
                sequence.append(dataset[index])
        
        if len(sequence) == len(default_intervals):
            inputs.append(sequence[0:len(sequence)-1])
            labels.append([sequence[-1]])
    inputs = np.stack(inputs, axis=0)
    labels = np.stack(labels, axis=0)


    num_samples = len(inputs)
    
    if use_level:
        inputs = np.transpose(inputs, (0, 1, 5, 4, 3, 2))
        labels = np.transpose(labels, (0, 1, 5, 4, 3, 2))


    else:
        inputs = np.transpose(inputs, (0, 1, 4, 3, 2))
        labels = np.transpose(labels, (0, 1, 4, 3, 2))

    flattened_inputs = inputs.reshape(-1, inputs.shape[2])
    flattened_labels = labels.reshape(-1, labels.shape[2])

    inputs_mean = np.mean(flattened_inputs, axis=0).reshape(1,1,12,1,1)
    labels_mean = np.mean(flattened_labels, axis=0).reshape(1,1,12,1,1)
    inputs_std = np.std(flattened_inputs, axis=0).reshape(1,1,12,1,1)
    labels_std = np.std(flattened_labels, axis=0).reshape(1,1,12,1,1)

    normalized_inputs = (inputs - inputs_mean) / inputs_std 
    normalized_labels = (labels - labels_mean) / labels_std

    inputs = normalized_inputs.reshape(inputs.shape)
    labels = normalized_labels.reshape(labels.shape)

    #keep a copy of original data before splitting and shuffling for sequence prediction testing
    original_inputs = torch.tensor(inputs.copy())
    original_labels = torch.tensor(labels.copy())

    #shuffle
    inputs = inputs[torch.randperm(len(inputs), axis=0)]
    labels = inputs[torch.randperm(len(labels), axis=0)]

    #split into training and testing
    training_size = math.floor(split_percentage*num_samples)
    X_train = torch.tensor(inputs[0:training_size], dtype=torch.float32)
    X_test = torch.tensor(inputs[training_size::], dtype=torch.float32)
    Y_train = torch.tensor(labels[0:training_size], dtype=torch.float32)
    Y_test = torch.tensor(labels[training_size::], dtype=torch.float32)

    # X: (num_samples, sequence_len, features, lat, lon)
    # Y: (num_samples, 1, features, lat, lon)
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    original_dataset = TensorDataset(original_inputs, original_labels)


    torch.save(train_dataset, './data/train_dataset_norm_simple.pth')
    torch.save(test_dataset, './data/test_dataset_norm_simple.pth')
    torch.save(original_dataset, './data/original_dataset_norm_simple.pth')

    return train_dataset, test_dataset

    # np.save('../data/test_data_array.npy', dataset)


preprocess_data(0.8)