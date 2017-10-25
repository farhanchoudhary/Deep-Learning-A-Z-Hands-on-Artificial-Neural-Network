# Recurrent Neural Network

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime

# Importing the training set
training_set = pdr.get_data_yahoo(symbols='GOOGL', start=datetime(2012, 1, 1), end=datetime(2016, 1, 1))
training_set = training_set[['Open']].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
X_train = training_set[0:-1]
y_train = training_set[1:]

# Reshaping
X_train = np.reshape(X_train, (len(X_train), 1, 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
# input_shape=(None, 1) = input_shape=(timestep_count, features_count)
regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))

# Adding the output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size=32, epochs=200)


# Use regressor.summary() to get a summary of the NN


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
test_set = pdr.get_data_yahoo(symbols='GOOGL', start=datetime(2012, 1, 1), end=datetime(2016, 1, 1))
real_stock_price = test_set[['Open']].values

inputs = real_stock_price
inputs = sc.transform(inputs)
inputs = np.reshape(inputs, (len(inputs), 1, 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color='red', label='Real Google Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()