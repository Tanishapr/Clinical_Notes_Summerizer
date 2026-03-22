import pandas as pd
import evaluate
from utils import load_model

DATA_PATH = "data/processed/clinical_notes_cleaned.csv"

def evaluate_model():
    model, tokenizer, device = load_model()
    
    rouge = evaluate.load("rouge")

    df = pd.read_csv(DATA_PATH)

    # Using 'note' as reference and 'cleaned_note' as prediction
    references = df["note"].astype(str).tolist()
    predictions = df["cleaned_note"].astype(str).tolist()

    results = rouge.compute(predictions=predictions, references=references)

    print("\n📊 ROUGE Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    evaluate_model()