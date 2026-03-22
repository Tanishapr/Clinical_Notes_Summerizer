import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from torch.optim import AdamW
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import ClinicalBERTModel

# Configuration
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
BATCH_SIZE = 8  # Reduce if CPU memory is low
EPOCHS = 2
LR = 2e-5
NUM_LABELS = 2  # Change based on task (binary classification here)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class ClinicalNotesDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = str(self.data.iloc[index]["cleaned_note"])
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        # Dummy labels (replace with actual if available)
        label = torch.tensor(0)  
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label,
        }

def train_model():
    # Load data
    df = pd.read_csv("data/processed/clinical_notes_cleaned.csv")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = ClinicalNotesDataset(df, tokenizer, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ClinicalBERTModel(model_name=MODEL_NAME, num_labels=NUM_LABELS)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, leave=True)
        for batch in loop:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            
            # Make sure logits and labels match batch_size
            labels = labels.view(-1)  # Flatten to (batch_size,)
            if logits.size(0) != labels.size(0):
                logits = logits[: labels.size(0), :]

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

    # Save the trained model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/clinical_bert.pth")
    print("Training completed and model saved to models/clinical_bert.pth")

if __name__ == "__main__":
    train_model()