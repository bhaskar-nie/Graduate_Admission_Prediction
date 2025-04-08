# Graduate Admission Prediction

This project implements a neural network model to predict the chances of admission to graduate programs based on various academic parameters from an Indian perspective.

## Dataset Overview

The dataset contains 400 records with several parameters considered important during the application process for Masters Programs:

1. GRE Scores (out of 340)
2. TOEFL Scores (out of 120)
3. University Rating (out of 5)
4. Statement of Purpose (SOP) Strength (out of 5)
5. Letter of Recommendation (LOR) Strength (out of 5)
6. Undergraduate CGPA (out of 10)
7. Research Experience (binary: 0 or 1)
8. Chance of Admit (target variable, ranging from 0 to 1)

## Requirements

- Python
- NumPy
- Pandas
- TensorFlow/Keras
- Scikit-learn
- Matplotlib

## Data Preparation

The dataset is processed through the following steps:

1. Data loading and exploration
2. Feature selection (using the first 7 columns as features)
3. Splitting data into training (80%) and testing (20%) sets
4. Feature scaling using Min-Max normalization

## Model Architecture

A simple neural network with the following structure:
- Input layer: 7 neurons (one for each feature)
- First hidden layer: 7 neurons with ReLU activation
- Second hidden layer: 7 neurons with ReLU activation
- Output layer: 1 neuron with linear activation (for regression)

Total parameters: 120

## Training Configuration

- Loss function: Mean Squared Error
- Optimizer: Adam
- Epochs: 100
- Validation split: 20% of training data

## Performance

The model achieves an R² score of approximately 0.787 on the test set, indicating a good fit for predicting graduate admission chances.

## How to Run the Code

1. Open the notebook in Google Colab or a Jupyter environment
2. Run the import statements to load all required libraries
3. Load the "Admission_Predict.csv" dataset using pandas
4. Perform exploratory data analysis to understand the dataset structure
5. Separate features (X) and target variable (y)
6. Split the data into training and testing sets
7. Apply Min-Max scaling to normalize the features
8. Create the neural network model with the specified architecture
9. Compile the model with appropriate loss function and optimizer
10. Train the model with the specified number of epochs
11. Evaluate the model using R² score
12. Visualize the training and validation loss curves

## Visualization

The notebook includes code to visualize:
- Training and validation loss over epochs
- Dataset exploration results

## Potential Improvements

- Feature engineering to create more informative variables
- Hyperparameter tuning for better performance
- Trying different neural network architectures
- Implementing cross-validation for more robust evaluation
- Exploring other regression algorithms for comparison
