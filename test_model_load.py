from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
tokenizer = DistilBertTokenizer.from_pretrained("chatbot_model")
model = DistilBertForSequenceClassification.from_pretrained("chatbot_model")
print("Model loaded successfully")