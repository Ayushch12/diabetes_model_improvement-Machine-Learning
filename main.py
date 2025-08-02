from src.load_data import load_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model
from utils.plot_confusion_matrix import plot_conf_matrix

from sklearn.model_selection import train_test_split
import joblib
import os

# Step 1: Load Data
df = load_data("data/diabetes.csv")
print(df.columns)