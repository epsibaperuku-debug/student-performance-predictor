import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample dataset
data = {
    "study_hours": [1, 2, 3, 4, 5],
    "attendance": [60, 65, 70, 75, 80],
    "marks": [50, 55, 65, 70, 80]
}

df = pd.DataFrame(data)

# Features and target
X = df[['study_hours', 'attendance']]
y = df['marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
prediction = model.predict([[6, 85]])

print("Predicted Marks:", prediction)
