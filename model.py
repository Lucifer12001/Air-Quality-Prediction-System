import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load datasets
sensor_data = pd.read_csv("sensor_data.csv")
quality_data = pd.read_csv("quality_control_data.csv")

# Merge datasets
rawdataset = sensor_data.merge(quality_data, on="prod_id")
dataset = rawdataset.drop(columns='prod_id')

# Split dataset into features (X) and target variable (Y)
X = dataset.iloc[:, :-1]  # All columns except the last one
Y = dataset.iloc[:, -1]   # Last column (Target variable)

# Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=8)

# Train the Decision Tree Classifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Save the trained model as a .pkl file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model has been trained and saved as 'model/model.pkl'")