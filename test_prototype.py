import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import mean_absolute_error
import joblib

# Load category mapping for chatbot
category_map = pd.read_csv("category_map.csv", index_col="code")["category"].to_dict()

# Load chatbot model
tokenizer = DistilBertTokenizer.from_pretrained("chatbot_model")
model = DistilBertForSequenceClassification.from_pretrained("chatbot_model")

# Define test queries covering all 10 categories
test_queries = [
    {"query": "What’s the cheapest flight to Lagos?", "expected": "price_inquiry"},
    {"query": "Can I cancel my booking to Abuja?", "expected": "cancellation"},
    {"query": "Check status of my flight to London", "expected": "flight_status"},
    {"query": "Baggage rules for Accra flight", "expected": "baggage"},
    {"query": "Use my miles for a ticket to Paris", "expected": "loyalty"},
    {"query": "Refund for my New York flight", "expected": "refunds"},
    {"query": "Check-in for Lagos flight", "expected": "check_in"},
    {"query": "Vegetarian meal for my flight", "expected": "special_requests"},
    {"query": "What’s the weather in Accra?", "expected": "other"},
    {"query": "Change my flight to London", "expected": "booking_change"},
    {"query": "How much is a flght to Abuja?", "expected": "price_inquiry"},  # Variation with typo
    {"query": "Please cancel my Paris booking", "expected": "cancellation"},  # Variation with polite phrasing
]

# Test chatbot
model.eval()
correct = 0
chatbot_results = []
for test in test_queries:
    inputs = tokenizer(test["query"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    label = torch.argmax(outputs.logits, dim=1).item()
    predicted = category_map[label]
    is_correct = predicted == test["expected"]
    if is_correct:
        correct += 1
    chatbot_results.append({
        "query": test["query"],
        "predicted": predicted,
        "expected": test["expected"],
        "correct": is_correct
    })
chatbot_accuracy = (correct / len(test_queries)) * 100
print(f"Chatbot Accuracy: {chatbot_accuracy:.2f}% ({correct}/{len(test_queries)} correct)")

# Load pricing data and model
data = pd.read_csv("processed_prices.csv")
X = data[["scaled_day", "scaled_demand", "scaled_distance"]]
y = data["price"]
pricing_model = joblib.load("pricing_model.pkl")

# Test pricing model
predictions = pricing_model.predict(X)
mae = mean_absolute_error(y, predictions)
avg_price = y.mean()
mae_percentage = (mae / avg_price) * 100
print(f"Pricing Mean Absolute Error: ${mae:.2f}")
print(f"Average Price: ${avg_price:.2f}")
print(f"MAE as Percentage of Average Price: {mae_percentage:.2f}%")

# Save results to file
with open("prototype_results.txt", "w") as f:
    f.write(f"Chatbot Accuracy: {chatbot_accuracy:.2f}% ({correct}/{len(test_queries)} correct)\n")
    f.write("\nChatbot Test Cases:\n")
    for result in chatbot_results:
        f.write(f"Query: {result['query']}\n")
        f.write(f"Predicted: {result['predicted']}\n")
        f.write(f"Expected: {result['expected']}\n")
        f.write(f"Correct: {result['correct']}\n\n")
    f.write(f"Pricing Mean Absolute Error: ${mae:.2f}\n")
    f.write(f"Average Price: ${avg_price:.2f}\n")
    f.write(f"MAE as Percentage of Average Price: {mae_percentage:.2f}%\n")
print("Saved results to 'prototype_results.txt'")