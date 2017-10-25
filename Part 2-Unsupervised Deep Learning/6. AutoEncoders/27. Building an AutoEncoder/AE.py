# AutoEncoders

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1')

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t')
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


# Array with users in lines and movies in columns
# [user_1, user_2, ..., user_943]
# [[movie_1_rating] [movie_2_rating] ... [movie_1682_rating]]
training_set = convert(training_set)
test_set = convert(test_set)

# Converting the data into Torch tensors
training_set = torch.FloatTensor(training_set)#.cuda()
test_set = torch.FloatTensor(test_set)#.cuda()


# Creating the architecture of the Neural Network
class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=0.01, weight_decay=0.5)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.  # Number of users who rated at least 1 movie
    for id_user in range(nb_users):
        # Creates a 2D array instead of 1D, Pytorch only accept batch of data
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()  # Outputs should be the same as inputs
        if torch.sum(target.data) > 0:  # User rated at least 1 movie
            output = sae.forward(input)
            target.require_grad = False  # For code optimization
            output[target == 0] = 0
            loss = criterion(output, target)
            loss.backward()  # Perform backpropagation
            optimizer.step()  # Define the intensity of backward pass

            # 1e-10 so that we never divide by 0
            # mean_corrector corresponds to the average of the error of the rated movies only
            # it's not used in the backprop calculation, just for metrics
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            train_loss += np.sqrt(loss.data[0] * mean_corrector)
            s += 1.  # Increment number of users who rated at least 1 movie
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    # The training set contains movies that the user has not yet watched
    input = Variable(training_set[id_user]).unsqueeze(0)
    # The test set contains the movies that the user watched
    target = Variable(test_set[id_user])
    if torch.sum(target.data) > 0:
        output = sae.forward(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)

        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data[0] * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss / s))
