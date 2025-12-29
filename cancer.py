import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()
print(data)

df= pd.read_csv("C:/Users/mm/Downloads/Compressed/Data/Breast_Cancer (1).csv")
data=pd.DataFrame(df)
print(data)
wf=[]