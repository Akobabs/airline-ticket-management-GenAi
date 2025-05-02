import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load category mapping
category_map = pd.read_csv("category_map.csv", index_col="code")["category"].to_dict()

# Load model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("chatbot_model")
model = DistilBertForSequenceClassification.from_pretrained("chatbot_model")

# Test queries
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
    {"query": "Change my flight to London", "expected": "booking_change"}
]

# Test predictions
model.eval()
correct = 0
for test in test_queries:
    inputs = tokenizer(test["query"], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    label = torch.argmax(outputs.logits, dim=1).item()
    predicted = category_map[label]
    is_correct = predicted == test["expected"]
    if is_correct:
        correct += 1
    print(f"Query: {test['query']}\nPredicted: {predicted}\nExpected: {test['expected']}\nCorrect: {is_correct}\n")
accuracy = correct / len(test_queries) * 100
print(f"Chatbot Accuracy: {accuracy}%")

# Save results
with open("chatbot_results.txt", "w") as f:
    f.write(f"Chatbot Accuracy: {accuracy}%\n")
    for test in test_queries:
        inputs = tokenizer(test["query"], return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        label = torch.argmax(outputs.logits, dim=1).item()
        predicted = category_map[label]
        f.write(f"Query: {test['query']}, Predicted: {predicted}, Expected: {test['expected']}\n")
print("Saved results to 'chatbot_results.txt'")