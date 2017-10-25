# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the Keras libraries and packages
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras import backend

# Importing the training set
script_dir = os.path.dirname(__file__)
train_set_path = os.path.join(script_dir, '../dataset/Google_Stock_Price_Train.csv')
# Importing the training set
training_set = pd.read_csv(train_set_path)
training_set = training_set.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
X_train = training_set[0:1257]
y_train = training_set[1:1258]

# Reshaping
X_train = np.reshape(X_train, (1257, 1, 1))

# Part 2 - Building the RNN

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size=32, epochs=200)

# Part 3 - Making the predictions and visualising the results

script_dir = os.path.dirname(__file__)
test_set_path = os.path.join(script_dir, '../dataset/Google_Stock_Price_Test.csv')

# Getting the real stock price of 2017
test_set = pd.read_csv(test_set_path)
real_stock_price = test_set.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (20, 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Align real values with their predictions
predicted_stock_price = predicted_stock_price[:-1]
real_stock_price = real_stock_price[1:]

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

# Homework

# Getting the real stock price of 2012 - 2016
script_dir = os.path.dirname(__file__)
real_set_path = os.path.join(script_dir, '../dataset/Google_Stock_Price_Train.csv')
real_stock_price_train = pd.read_csv(real_set_path)
real_stock_price_train = real_stock_price_train.iloc[:, 1:2].values

# Getting the predicted stock price of 2012 - 2016
predicted_stock_price_train = regressor.predict(X_train)
predicted_stock_price_train = sc.inverse_transform(predicted_stock_price_train)

# Visualising the results
plt.plot(real_stock_price_train, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price_train, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
backend.clear_session()

# Part 4 - Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error

rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
print("RMSE =", rmse)
