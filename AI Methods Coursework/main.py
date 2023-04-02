import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import random

# Read in the prepped data from the training excel file
dataset = pd.read_excel('TRAINING.xlsx', usecols=[0,1,2,3,4,5,6])
# Read in the prepped data from the testing excel file
testing = pd.read_excel('TESTING.xlsx', usecols=[0,1,2,3,4,5,6])
#DEFINING FORMULAS FOR ANN
#Sigmoid activation function 
def sigmoid(x):
     return 1 / (1 + np.exp(-x))


#Derivative of sigmoid function
def sigmoidDerivative(x):
    return x * (1 - x)

# Mean Squared Error function for ANN
def meanSquaredError(y_true, y_pred):
    return ((y_pred-y_true)**2).sum() / (2* y_pred.size)

# INITIALISATION
# Set the number of epochs
epochs = 10000

# Set the learning rate
learning_rate = 0.01

# Set the number of neurons in the hidden layer
hidden_neurons = 4

# Set the number of neurons in the output layer
output_neurons = 1

# Set the number of neurons in the input layer
input_neurons = 5

# Set the bias for the hidden layer
bias_hidden = np.random.uniform(-input_neurons, input_neurons, size=hidden_neurons)

# Set the bias for the output layer
bias_output = np.random.uniform(-input_neurons, input_neurons)

#ADDING WEIGHTS
# Set the weights for the hidden layer
weights_hidden = np.random.uniform(-input_neurons, input_neurons, size=(input_neurons, hidden_neurons))

# Set the weights for the output layer
weights_output = np.random.uniform(-hidden_neurons, output_neurons, size=(hidden_neurons, output_neurons))

def convert_date(date):
    date_str = str(date)
    if len(date_str) == 5:
        month = date_str[0]
        day = date_str[1:3]
        year = '19' + date_str[3:]
    else:
        month = date_str[:2]
        day = date_str[2:4]
        year = '19' + date_str[4:]
    return f'{year}-{month}-{day}'

# Apply the conversion function to the Date column
dataset['Date'] = dataset['Date'].apply(convert_date)
testing['Date'] = testing['Date'].apply(convert_date)
testing['Date'] = pd.to_datetime(testing['Date'])
dataset['Date'] = pd.to_datetime(dataset['Date'])

#TRAINING
# Set the training data
# select all rows and all columns except the first and last one (PanE as we are predicting this)
X = dataset.iloc[:, 1:-1].values

# Select only the last column (PanE)
Y = dataset.iloc[:, -1].values

testingValues = testing.iloc[:, 1:-1].values
testingValuesY = testing.iloc[:, -1].values

#Store any errors found in the output layer


#BACKPROPAGATION
errors = []
def backpropagation(weights_hidden, weights_output, bias_hidden, bias_output, errors):

    # Main loop for epochs
    for i in range(epochs):
        # Set error to 0
        error = 0
        weights_hidden_update = np.zeros_like(weights_hidden)
        weights_output_update = np.zeros_like(weights_output)
        bias_hidden_update = np.zeros_like(bias_hidden)
        bias_output_update = np.zeros_like(bias_output)
    
        # Loop through each row in the training data
        for rows in range(len(X)):
            # Forward pass
            hidden_layer_input = np.dot(X[rows], weights_hidden) + bias_hidden
            hidden_layer_output = sigmoid(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
            output_layer_output = sigmoid(output_layer_input)

            # Track errors
            error += meanSquaredError(Y[rows], output_layer_output)

            # Backward pass
            output_error = (Y[rows] - output_layer_output)
            hidden_error = np.dot(output_error, weights_output.T) * sigmoidDerivative(hidden_layer_output)

            # Accumulate the weight updates for the output layer
            weights_output_update += np.outer(hidden_layer_output, output_error)

            # Accumulate the weight updates for the hidden layer
            weights_hidden_update += np.outer(X[rows], hidden_error)

            # Accumulate bias updates
            bias_output_update += np.sum(output_error)
            bias_hidden_update += np.sum(hidden_error)

        # divide weight and bias updates by the number of training examples
        weights_output_update /= len(X)
        weights_hidden_update /= len(X)
        bias_output_update /= len(X)
        bias_hidden_update /= len(X)

        # Update weights for output and hidden layers
        weights_output += learning_rate * weights_output_update
        weights_hidden += learning_rate * weights_hidden_update
        

        # Update biases for output and hidden layers
        bias_output += learning_rate * bias_output_update
        bias_hidden += learning_rate * bias_hidden_update 

        # Calculate the mean squared error
        error /= len(X)

        # Store the errors
        errors.append(error)

        # Print the error every 100 epochs
        if i % 100 == 0:
            print("Error: ", error)
        
    return weights_hidden, weights_output, bias_hidden, bias_output, errors

# PREDICTION
def predict(weights_hidden, weights_output, bias_hidden, bias_output,X):
    predictions = []

    # Loop through each row in the testing data
    for rows in range(len(testingValues)):
        
        # Forward pass
        
        hidden_layer_input = np.dot(testingValues[rows], weights_hidden) + bias_hidden
        hidden_layer_output = sigmoid(hidden_layer_input)
        # print("HIDDEN LAYER OUTPUT", hidden_layer_output)

        output_layer_input = np.dot(hidden_layer_output, weights_output) + bias_output
        output_layer_output = sigmoid(output_layer_input)
        # print("OUTPUT LAYER OUTPUT", output_layer_output)

        # Store the predictions
        predictions.append(output_layer_output.reshape(-1))
        # print("PREDICTIONS", output_layer_output.reshape(-1))
  
    return np.array(predictions)

# Run the backpropagation function
weights_hidden, weights_output, bias_hidden, bias_output, errors = backpropagation(weights_hidden, weights_output, bias_hidden, bias_output, errors)
# Print the updated weights_hidden
predictions = predict(weights_hidden, weights_output, bias_hidden, bias_output, X)
            
# write the predictions to a csv file
df = pd.DataFrame()
df['Trained Data'] = predictions.ravel()
df['Untrained Data'] = testing['PANE2']
df['Date'] = testing['Date']

# convert date column to string format
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

#Convert the date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# write to excel
df.to_excel('PREDICTIONS2.xlsx', index=False)


# Plot the errors
plt.plot(errors)
plt.title("Errors with Training data")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()
