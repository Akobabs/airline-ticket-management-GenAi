import pandas as pd
from sklearn.metrics import mean_absolute_error
import joblib

# Load preprocessed data
data = pd.read_csv("processed_prices.csv")
X = data[["scaled_day", "scaled_demand", "scaled_distance"]]
y = data["price"]

# Load model
model = joblib.load("pricing_model.pkl")

# Make predictions
predictions = model.predict(X)

# Calculate error
mae = mean_absolute_error(y, predictions)
avg_price = y.mean()
mae_percentage = (mae / avg_price) * 100

# Print results
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Average Price: ${avg_price:.2f}")
print(f"MAE as Percentage of Average Price: {mae_percentage:.2f}%")

# Save results
with open("pricing_results.txt", "w") as f:
    f.write(f"Mean Absolute Error: ${mae:.2f}\n")
    f.write(f"Average Price: ${avg_price:.2f}\n")
    f.write(f"MAE as Percentage of Average Price: {mae_percentage:.2f}%")
print("Saved results to 'pricing_results.txt'")