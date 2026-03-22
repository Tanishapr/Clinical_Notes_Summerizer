import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

def generate_embeddings(input_file: str, output_file: str):
    # Load BioBERT
    tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    # Load preprocessed notes
    df = pd.read_csv(input_file)

    embeddings = []

    for note in df["cleaned_note"].tolist()[:50]:  # limit to 50 for quick test
        inputs = tokenizer(
            note,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Take mean of last hidden states
        vector = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(vector)

    # Save embeddings as CSV
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.to_csv(output_file, index=False)
    print(f"Embeddings saved to {output_file}")


if __name__ == "__main__":
    input_path = "data/processed/clinical_notes_cleaned.csv"
    output_path = "data/processed/embeddings.csv"
    generate_embeddings(input_path, output_path)