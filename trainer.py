from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import pandas as pd

# data_open_path = "/content/drive/My Drive/training_data_new.csv"
# model_save_path = "/content/drive/My Drive/query_classifier"
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))

# Load dataset
df = pd.read_csv("training_data_new.csv")

# Convert labels to numerical format
label_map = {"Direct Embedding Search": 0, "Keyword Extraction Needed": 1}
df["label"] = df["label"].map(label_map)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
def preprocess(example):
    return tokenizer(example["query"], truncation=True, padding="max_length")

dataset = dataset.map(preprocess, batched=True)

# Load model and move to GPU
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
model.to(device)  # Move model to GPU

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,  # Add if evaluation dataset exists
    num_train_epochs=3,
    fp16=True,  # Mixed precision training for speedup on GPU
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("query_classifier")
tokenizer.save_pretrained("query_classifier")
