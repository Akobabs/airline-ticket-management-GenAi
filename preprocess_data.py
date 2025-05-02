import pandas as pd
from sklearn.preprocessing import StandardScaler
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize

# Preprocess chatbot data
queries_df = pd.read_csv("queries.csv")
queries_df["query"] = queries_df["query"].str.lower().str.replace("[^a-z0-9 ]", "", regex=True)
queries_df["tokens"] = queries_df["query"].apply(word_tokenize)
queries_df["category_code"] = queries_df["category"].astype("category").cat.codes
# Save category mapping
category_map = dict(enumerate(queries_df["category"].astype("category").cat.categories))
pd.DataFrame.from_dict(category_map, orient="index", columns=["category"]).to_csv("category_map.csv", index_label="code")
queries_df.to_csv("processed_queries.csv", index=False)
print("Saved processed_queries.csv and category_map.csv")

# Preprocess pricing data
prices_df = pd.read_csv("prices.csv")
prices_df["demand_encoded"] = prices_df["demand"].map({"low": 0, "high": 1})
features = prices_df[["day_of_week", "demand_encoded", "distance"]]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
prices_df[["scaled_day", "scaled_demand", "scaled_distance"]] = scaled_features
prices_df.to_csv("processed_prices.csv", index=False)
import joblib
joblib.dump(scaler, "scaler.pkl")
print("Saved processed_prices.csv and scaler.pkl")