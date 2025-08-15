# -*- coding: utf-8 -*-
"""Reward_Model_Training.ipynb

This notebook performs the fine-tuning process for the BERT Reward Model and
evaluates its performance on a dedicated test set.
"""

# @title 1. Install Libraries
!pip install torch transformers scikit-learn numpy pandas tqdm accelerate datasets evaluate -q

# @title 2. Import Libraries and Set Up Environment
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import evaluate
from datasets import Dataset as HFDataset
from tqdm.auto import tqdm

# --- IMPORTANT CONFIGURATION ---
# Path to your custom dataset. Make sure to upload this file to Colab.
csv_filepath = '/content/en-disfluent-sentences-labelled.csv.csv'

# Set your model save path
REWARD_MODEL_OUTPUT_DIR = './reward_model_fluency_classifier_custom_data'
# -------------------------------

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# @title 3. Load and Prepare Your Dataset
# This step loads the data and splits it for training, validation, and testing.
try:
    df = pd.read_csv(csv_filepath)
    disfluent_sentences = df['Disfluent Sentence'].tolist()
    fluent_sentences = df['Fluent Sentence'].tolist()
    print(f"Dataset loaded successfully with {len(disfluent_sentences)} samples.")
except FileNotFoundError:
    print(f"Error: '{csv_filepath}' not found. Please upload it to Colab.")
    raise

# Create a combined dataset with labels for classification
reward_data_list = []
for sentence in fluent_sentences:
    reward_data_list.append({"text": str(sentence), "labels": 0}) # Label 0 for FLUENT
for sentence in disfluent_sentences:
    reward_data_list.append({"text": str(sentence), "labels": 1}) # Label 1 for DISFLUENT

# Split the data into training, validation, and test sets (80/10/10)
train_data, temp_data = train_test_split(reward_data_list, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

reward_train_dataset = HFDataset.from_list(train_data)
reward_val_dataset = HFDataset.from_list(val_data)
reward_test_dataset = HFDataset.from_list(test_data)

print(f"\nTrain samples: {len(reward_train_dataset)}")
print(f"Validation samples: {len(reward_val_dataset)}")
print(f"Test samples: {len(reward_test_dataset)}")


# @title 4. Define Tokenization and Model for BERT
model_name = "bert-base-uncased"
reward_tokenizer = AutoTokenizer.from_pretrained(model_name)
reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label={0: "FLUENT", 1: "DISFLUENT"},
    label2id={"FLUENT": 0, "DISFLUENT": 1}
).to(device)

def tokenize_data_reward(examples):
    return reward_tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

tokenized_reward_train_dataset = reward_train_dataset.map(tokenize_data_reward, batched=True)
tokenized_reward_val_dataset = reward_val_dataset.map(tokenize_data_reward, batched=True)
tokenized_reward_test_dataset = reward_test_dataset.map(tokenize_data_reward, batched=True)

tokenized_reward_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_reward_val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_reward_test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


# @title 5. Fine-tune the BERT Model
def compute_metrics_reward(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(
    output_dir=REWARD_MODEL_OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=reward_model,
    args=training_args,
    train_dataset=tokenized_reward_train_dataset,
    eval_dataset=tokenized_reward_val_dataset,
    tokenizer=reward_tokenizer,
    compute_metrics=compute_metrics_reward
)
print("\nStarting BERT model fine-tuning for Reward Model...")
trainer.train()

# Save the fine-tuned BERT model locally
reward_model.save_pretrained(REWARD_MODEL_OUTPUT_DIR)
reward_tokenizer.save_pretrained(REWARD_MODEL_OUTPUT_DIR)
print("\nReward model fine-tuning complete and saved locally.")


# @title 6. Evaluate the Fine-tuned Model on the Test Set
print("\n--- Evaluating Reward Model on Test Set ---")
predictions_reward = []
references_reward = []

with torch.no_grad():
    test_loader = DataLoader(tokenized_reward_test_dataset, batch_size=16)
    for batch in tqdm(test_loader, desc="Evaluating Reward Model"):
        outputs = reward_model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        predictions_reward.extend(predictions)
        references_reward.extend(batch['labels'].numpy())

accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

accuracy = accuracy_metric.compute(predictions=predictions_reward, references=references_reward)
f1 = f1_metric.compute(predictions=predictions_reward, references=references_reward, average='binary')
precision = precision_metric.compute(predictions=predictions_reward, references=references_reward, average='binary')
recall = recall_metric.compute(predictions=predictions_reward, references=references_reward, average='binary')

print(f"Accuracy: {accuracy['accuracy']:.2f}")
print(f"F1-Score: {f1['f1']:.2f}")
print(f"Precision: {precision['precision']:.2f}")
print(f"Recall: {recall['recall']:.2f}")
