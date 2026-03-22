import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Clinical Note Summarization",
    layout="wide"
)

# -----------------------------
# Page Title
# -----------------------------
st.title("🩺 BioBERT Clinical Notes Summarizer")
st.write("This app summarizes clinical notes using transformer models.")

# -----------------------------
# Load summarization model
# -----------------------------
@st.cache_resource
def load_summarizer():
    summarizer = pipeline("summarization", model="t5-small")
    return summarizer

summarizer = load_summarizer()

# -----------------------------
# Input
# -----------------------------
clinical_note = st.text_area("Enter Clinical Note")

if st.button("Generate Summary"):
    result = summarizer(clinical_note, max_length=60, min_length=20)
    st.write(result[0]["summary_text"])
