import os
import pandas as pd
import spacy

# Load SciSpacy model
nlp = spacy.load("en_core_web_sm")

def split_sentences(text):
    """Split text into sentences using SciSpacy."""
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def preprocess_note(note):
    """Basic preprocessing: clean text, split into sentences."""
    note = str(note).strip()
    note = note.replace("\n", " ").replace("\r", " ")
    sentences = split_sentences(note)
    return " ".join(sentences)

def preprocess_dataset(input_file, output_file):
    df = pd.read_csv(input_file)

    # Handle column naming
    if 'note' in df.columns:
        pass  # already correct
    elif 'text' in df.columns:
        df = df.rename(columns={'text': 'note'})
    elif 'TEXT' in df.columns:
        df = df.rename(columns={'TEXT': 'note'})
    else:
        raise ValueError("CSV must have a column named 'note', 'text', or 'TEXT'")

    # Apply preprocessing
    df["cleaned_note"] = df["note"].apply(preprocess_note)

    df.to_csv(output_file, index=False)
    print(f"Preprocessed data saved to {output_file}")

# Example usage:
if __name__ == "__main__":
    input_path = "data/raw/medical_data.csv"
    output_path = "data/processed/clinical_notes_cleaned.csv"
    preprocess_dataset(input_path, output_path)