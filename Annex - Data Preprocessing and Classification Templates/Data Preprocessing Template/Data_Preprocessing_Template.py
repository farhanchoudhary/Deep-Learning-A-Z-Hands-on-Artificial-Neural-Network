# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

script_dir = os.path.dirname(__file__)
abs_file_path = os.path.join(script_dir, 'Data.csv')

# Importing the dataset
dataset = pd.read_csv(abs_file_path)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values