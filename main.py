import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data.csv")

# Features and target
X = df[['study_hours', 'attendance']]
y = df['marks']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n--- Student Performance Predictor ---")

study_hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))

new_data = pd.DataFrame([[study_hours, attendance]], 
                        columns=['study_hours', 'attendance'])

prediction = model.predict(new_data)

print("Predicted Marks:", prediction)
print("\nThank you for using the predictor!")