import torch
from transformers import AutoTokenizer
from model import ClinicalBERTModel

MODEL_PATH = "models/clinical_bert.pth"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def load_model(device=None):
    """
    Loads the trained ClinicalBERT model with weights.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ClinicalBERTModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_summary(text, model, tokenizer, device, max_len=128):
    """
    Generates embeddings or logits for a given input clinical note.
    (Right now, this is a classifier head output; for summarization,
    you will later plug in a seq2seq model like BART/T5.)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    ).to(device)

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.softmax(logits, dim=-1)

    return probs.cpu().numpy()


def get_embedding(text, model, tokenizer, device, max_len=128):
    """
    Extracts the last hidden state (embedding) from the ClinicalBERT model.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    ).to(device)

    with torch.no_grad():
        outputs = model.bert(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        # CLS token embedding
        embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.cpu().numpy()