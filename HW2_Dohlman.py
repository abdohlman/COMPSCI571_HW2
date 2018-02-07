import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve
from sklearn.datasets import make_classification
	
warnings.simplefilter("ignore")

## Problem 1: Classifiers for Basketball Courts

#				X1		X2		Y
Xy = np.matrix([[0.75, 0.10, -1.0],
				[0.85, 0.80, -1.0],
				[0.85, 0.95, 1.0],
				[0.15, 0.10, -1.0],
				[0.05, 0.25, 1.0],
				[0.05, 0.50, 1.0],
				[0.85, 0.25, -1.0]])

X, y = X