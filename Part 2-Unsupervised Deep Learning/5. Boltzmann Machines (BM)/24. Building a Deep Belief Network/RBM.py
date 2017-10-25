# Boltzmann Machines

# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-1m/movies.dat', sep='::', header=None, engine='python', encoding='latin-1',
                     names=['movie_id', 'title', 'category'])
users = pd.read_csv('ml-1m/users.dat', sep='::', header=None, engine='python', encoding='latin-1',
                    names=['user_id', 'gender', 'age', 'user_job_id', 'zip_code'])
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', header=None, engine='python', encoding='latin-1',
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])

# Preparing the training set and the test set
training_set = pd.read_csv('ml-100k/u1.base', delimiter='\t',
                           names=['user_id', 'movie_id', 'rating', 'timestamp'])
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter='\t',
                       names=['user_id', 'movie_id', 'rating', 'timestamp'])
test_set = np.array(test_set, dtype='int')

# Getting the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# Converting the data into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for user_id in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == user_id]
        id_ratings = data[:, 2][data[:, 0] == user_id]
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
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1


# Creating the architecture of the Neural Network
class RBM:
    def __init__(self, n_visible_nodes, n_hidden_nodes):
        self.W = torch.randn(n_hidden_nodes, n_visible_nodes)
        self.a = torch.randn(1, n_hidden_nodes)
        self.b = torch.randn(1, n_visible_nodes)

    def sample_h(self, x):
        """
        Sample the probabilities of the hidden nodes given
        the visible nodes
        :param x: inputs
        :return:
        """
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)  # Weights + bias
        p_h_given_v = torch.sigmoid(activation)  # output
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        """
        Sample the probabilities of the visibles nodes given the
        hidden nodes
        :param y:
        :return:
        """
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)  # Weights + bias
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)


nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

# Training the RBM
nb_epoch = 10
for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user + batch_size]
        v0 = training_set[id_user:id_user + batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1.
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user + 1]
    vt = test_set[id_user:id_user + 1]
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1.
print('test loss: ' + str(test_loss / s))
