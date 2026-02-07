import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Load data
data = pd.read_csv("../Data/crowd_data.csv")

# Convert time index to numbers
X = np.arange(len(data)).reshape(-1, 1)
y = data["people_count"].values

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict next 5 time steps
future_steps = np.arange(len(data), len(data) + 5).reshape(-1, 1)
predictions = model.predict(future_steps)

print("Future crowd predictions:")
for i, p in enumerate(predictions, 1):
    print(f"Step {i}: {int(p)} people")
