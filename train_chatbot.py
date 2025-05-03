import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Load processed data
data = pd.read_csv("processed_queries.csv")
queries, labels = data["query"], data["category_code"]

# Tokenize queries
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
encodings = tokenizer(list(queries), truncation=True, padding=True)

# Create custom dataset
class QueryDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

dataset = QueryDataset(encodings, labels)

# Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(set(labels)))

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="no",
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train model
trainer.train()

# Save model and tokenizer (force PyTorch format)
model.save_pretrained("chatbot_model", safe_serialization=False)
tokenizer.save_pretrained("chatbot_model")
print("Saved chatbot model to 'chatbot_model/'")