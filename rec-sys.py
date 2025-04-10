import numpy as np
import pandas as pd
from sklearn import model_selection, metrics, preprocessing
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# -- Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read Dataset
df = pd.read_csv("ml-latest-small/ratings.csv")
df.info()