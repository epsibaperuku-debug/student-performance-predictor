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

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Predict new value
new_data = pd.DataFrame([[6, 85]], columns=['study_hours', 'attendance'])
prediction = model.predict(new_data)
print("Predicted Marks:", prediction)

# Plot graph
plt.scatter(df['study_hours'], df['marks'])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()