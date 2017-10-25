# This model gives an accuracy of about 0.83875, this is far from the optimal accuracy we can get
from tensorflow.contrib.keras.api.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout
from tensorflow.contrib.keras import backend
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

# I execute the code remotely, this piece of code helps to find the right path for the files
script_dir = os.path.dirname(__file__)
abs_path = os.path.join(script_dir, 'Churn_Modelling.csv')

# Importing the dataset
df = pd.read_csv(abs_path)

y = df['Exited']
# Keep only useful columns
df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1, inplace=True)

# Encoding categorical data
X = pd.get_dummies(df, columns=['Geography', 'Gender'], drop_first=True)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [128, 256, 512],
              'epochs': [300, 500],
              'optimizer': ['adam']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
print("Best parameters:", best_parameters)
print("Best accuracy:", best_accuracy)

# Always close the Keras session to free up used resources
backend.clear_session()

