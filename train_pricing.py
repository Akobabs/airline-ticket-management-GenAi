import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load data
data = pd.read_csv("processed_prices.csv")
X = data[["scaled_day", "scaled_demand", "scaled_distance"]]
y = data["price"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "pricing_model.pkl")
print("Saved pricing model")