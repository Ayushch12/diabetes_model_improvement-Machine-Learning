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

# Step 2: Preprocess
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train
model = train_model(X_train, y_train)

# Step 4: Evaluate
cm = evaluate_model(model, X_test, y_test)
plot_conf_matrix(cm)