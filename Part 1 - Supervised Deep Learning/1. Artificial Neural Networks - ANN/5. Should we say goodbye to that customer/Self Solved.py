import pandas as pd
import os

# I execute the code remotely, this piece of code helps to find the right path for the files
script_dir = os.path.dirname(__file__)
train_path = os.path.join(script_dir, 'Churn_Modelling_train.csv')
test_path = os.path.join(script_dir, 'Churn_Modelling_test.csv')

# Importing the dataset
train_df_X = pd.read_csv(train_path)
test_df_X = pd.read_csv(test_path)
test_df_len = len(test_df_X)

y_train = train_df_X['Exited']
# Merge test and train to do the operations on both dataset and split them after
X = train_df_X.append(test_df_X)
# Keep only useful columns
X.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1, inplace=True)

# Encoding categorical data
X = pd.get_dummies(X, columns=['Geography', 'Gender'], drop_first=True)

# Split back train/test dataset
X_test = X[-test_df_len:]
X_train = X[:-test_df_len]

# Check that the size of the original dataset has been kept
assert(len(train_df_X) == len(X_train))

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras import backend

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=20, validation_split=0.1)

# Part 4 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

print("Should we say goodbye to that customer ?", *y_pred)

# Always close the Keras session to free up used resources
backend.clear_session()

