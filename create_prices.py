import pandas as pd
import random

# Mock pricing data
routes = ["Lagos-Abuja", "Lagos-Accra", "Abuja-London"]
data = []
for _ in range(1000):
    route = random.choice(routes)
    day = random.randint(0, 6)  # 0=Mon, 6=Sun
    demand = random.choice(["low", "high"])
    distance = {"Lagos-Abuja": 500, "Lagos-Accra": 1000, "Abuja-London": 5000}[route]
    base_price = 100 + distance * 0.1
    price = base_price * (1.2 if demand == "high" else 0.8)
    data.append({"date": "2025-06-01", "route": route, "day_of_week": day, "demand": demand, "distance": distance, "price": round(price, 2)})

# Save to CSV
df = pd.DataFrame(data)
df.to_csv("prices.csv", index=False)
print("Created prices.csv with 1000 samples")