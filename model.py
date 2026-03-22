# src/model.py
# This file is optional now; T5-small is loaded directly in app.py

from transformers import AutoModelForSeq2SeqLM

def load_t5_model(model_name="t5-small"):
    """
    Load the T5 model for summarization.
    """
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return model