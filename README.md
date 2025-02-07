# Guided Hyperparameter Search for Custom Deep Learning Models and Training Loops

a tensorflow-based pipeline for tuning deep learning model hyperparameters with a simple guided hyperparameter search strategy.

## Features
- **Custom Model (`SuperbModel`)**: A basic multi layer perceptron with batch normalization, dropout, and adaptive layer configuration.
- **Guided Hyperparameter Search**: Hyperparameter search algorithm based on iteratively perturbing the best set of hyperparameters. Tracks training progress with validation accuracy monitoring.

## File Overview
- **`model.py`**: Defines `SuperbModel` and `SuperbLayer` with customizable hyperparameters. Can be modified to implement any deep neural network structure.
- **`train_utils.py`**: Contains utility functions for batch sampling, status display, and training logic. The logic can be modified to implement any custom training loop.
- **`hyperparameter_search.py`**: Implements the guided hyperparameter search algorithm.
- **`train.py`**: Loads the MNIST dataset by default and executes hyperparameter search.

## **Usage**  
1. **Install dependencies**:  
   pip install tensorflow numpy

2. **Run hyperparameter search**:
   python train.py

3. Modify train.py, model.py, train_utils.py to implement custom architectures and use different datasets.

