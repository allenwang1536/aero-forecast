# ENSO-Watch: Deep Learning for Weather Forecasting

## Introduction

This project explores the application of deep learning techniques for weather forecasting, leveraging the power of neural networks to capture the intricate patterns and relationships within atmospheric data. Inspired by the challenges faced by traditional Numerical Weather Prediction (NWP) models, which often require intensive computational resources and can be prone to inaccuracies due to simplifications and assumptions, we propose a deep learning approach that aims to address these issues by training networks to learn robust representations and make accurate predictions.

## Data Preprocessing

The project utilizes the WeatherBench2 dataset, which is built upon the ECMWF Reanalysis v5 (ERA5) dataset. Specifically, we use the equiangular conservative data containing 64 x 32 grids and weather data in 6-hour intervals ranging from 1959 to 2022. This dataset provides the necessary weather information at a relatively low resolution, allowing for efficient preprocessing.

To prepare the data for our models, we perform the following preprocessing steps:

1. **Feature Selection**: We filter the original dataset to include 12 relevant weather features, such as geopotential, wind components, temperature, and precipitation, as well as latitude and longitude coordinates corresponding to the East Coast region.

2. **Standardization**: We standardize the dimensions of each feature to ensure consistency across the dataset.

3. **Temporal Sequence Formation**: To capture the spatiotemporal nature of the weather data, we concatenate each timestamp with a sequence of previous timestamps, forming a time series input. The labels are then taken as the weather data one day (or 4 time steps) ahead.

4. **Normalization**: To avoid issues like vanishing gradients, we normalize the data based on the mean and standard deviation for each feature.

By performing these preprocessing steps, we transform the raw weather data into a format suitable for training our deep learning models.

## Model Architectures

We explore and implement several deep learning model architectures to tackle the weather forecasting task:

1. **CNN-LSTM**: This architecture combines convolutional neural networks (CNN) for spatial pattern recognition and long short-term memory (LSTM) networks for temporal modeling. The CNN component captures local spatial patterns, while the LSTM component models the temporal dependencies across the sequence of time steps.

2. **CNN-Transformer**: Similar to the CNN-LSTM model, the CNN-Transformer uses CNN layers to encode spatial information. However, instead of LSTMs, it employs a Transformer encoder to process the flattened CNN output, leveraging self-attention mechanisms to capture long-range temporal dependencies.

3. **Temporal Convolutional Network (TCN)**: The TCN architecture consists of stacked TemporalBlock2D layers, each applying dilated 2D convolutions along the spatial dimensions. The dilations increase exponentially with depth, enabling the TCN to capture patterns at various temporal resolutions simultaneously.

4. **S4 Model**: We attempt to implement the state-of-the-art S4 model, which is specifically designed for handling long sequences. The S4 architecture incorporates intricate components like the S4Layer and S4Block, employing structured state-space transformations and non-circular convolutions with Cauchy kernels along the time dimension.

While the CNN-LSTM, CNN-Transformer, and TCN models show promising results, fully implementing the complex S4 architecture proves challenging due to its intricate nature.

## Results and Challenges

Our models demonstrate good prediction accuracy, with the CNN-LSTM and CNN-Transformer architectures performing particularly well. However, we acknowledge that there is room for improvement through further fine-tuning and optimization.

One of the biggest challenges we faced was the implementation of the S4 Model due to its complexity. Understanding the theoretical foundations and implementation details of the S4 Model's intricate components, such as the S4Layer and S4Block, proved to be a significant hurdle.

Additionally, we encountered challenges in dealing with the complexity of the shapes during training and preprocessing our multidimensional, spatiotemporal weather data. Ensuring that our preprocessing correctly handled these shapes and that our models were compatible with the resulting input and output shapes required careful management and trial-and-error.

## Future Work

Looking ahead, we aim to continue improving our models and exploring advanced techniques to enhance their performance further. Some potential areas for future work include:

- Allocating more time and resources to thoroughly understand and implement the S4 Model.
- Conducting extensive hyperparameter tuning and experimenting with different model configurations to optimize prediction accuracy.
- Exploring advanced techniques such as transfer learning, ensemble methods, and attention mechanisms.
- Investigating more sophisticated data preprocessing techniques, such as incorporating additional meteorological features or applying advanced feature engineering methods.
- Leveraging the knowledge gained from this project to tackle other complex, multidimensional data problems in various domains.

## Conclusion

 By exploring and implementing various model architectures, we gained valuable insights into the strengths and limitations of different approaches, as well as the importance of understanding the theoretical foundations and implementation details of complex architectures like the S4 Model. While we faced challenges in fully implementing the S4 Model and dealing with the complexities of spatiotemporal data, this project served as a learning experience that highlighted the value of effective data preprocessing, handling complex data shapes, and thoroughly understanding the underlying mathematics and algorithms of modern deep learning architectures.
