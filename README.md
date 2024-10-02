# Spatial Wind Speed Forecasting Using Machine Learning - Graph Convolutional Networks

## Overview
This project uses machine learning techniques to forecast spatial wind speed in South Africa. The aim is to use historical wind speed data to create predictive models to predict future wind speed patterns.

## Related Work
The project builds upon the following research:

- **Graph Neural Networks in literature**: 
  - [NeuralLAM](https://arxiv.org/abs/2309.17370) , [Keisler](https://arxiv.org/abs/2202.07575) , [GraphCast](https://arxiv.org/abs/2212.12794)

## Data Description
- **Source**: ERA5 wind speed data for South Africa (2018-2022).
- **Format**: Data is stored as a .nc file and opened using xarray.
- **Shape**: The dataset has the following shape: `[time_steps, latitude, longitude, wind_speed_values]`.

## Data Processing
1. **Windowing**: The data is windowed based on a specified `window_size` and `step_size` to create features, forcings, and targets for the machine learning models.
   
2. **Feature Extraction**:
   - **Features**: Previous `n` steps based on the `window_size`.
   - **Forcings**: Current hour of the day and month of the year at the time of prediction.
   - **Target**: The `m` number of forecasting steps based on the defined `steps` variable.

## Model Description - Graph Convolutional Network Model
- **Encoding**: The encoder component of the GraphCast architecture maps local regions of the input into nodes of the multi-mesh graph representation (GraphCast). 
- **Processing**: The processor component updates each multi-mesh node using learned message-passing.
- **Decoding**: The decoder component maps the processed multi-mesh features back onto the grid representation.
- **Tools**: This paper shows that encoding/decoding can be done using an MLP to concatenate multiple states.

## Implementations to be tested
- **Land mass forcings**: Add a layer that identifies the land mass of South Africa
- **Training for multiple steps**: In the training, increase the number of steps that the model is training.

## Issues
- **Computer crashing on PyTorch implementation**: Simulation computer with NVIDIA GeForce GPU crashing (temperature @ 73 \degrees C)
