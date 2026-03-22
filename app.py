# import streamlit as st
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # -----------------------------
# # Load Model and Tokenizer
# # -----------------------------
# @st.cache_resource
# def load_summarizer():
#     tokenizer = AutoTokenizer.from_pretrained("t5-small")
#     model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
#     model.eval()
#     return tokenizer, model

# tokenizer, model = load_summarizer()

# # -----------------------------
# # Streamlit UI
# # -----------------------------
# st.set_page_config(page_title="Clinical Note Summarization", layout="wide")
# st.title("🩺 Clinical Note Summarization")
# st.write("Enter a clinical note below and get a summarized output using T5-small model.")

# user_input = st.text_area("Clinical Note", height=200)

# if st.button("Summarize"):
#     if user_input.strip() == "":
#         st.warning("⚠️ Please enter some text.")
#     else:
#         input_text = "summarize: " + user_input
#         inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

#         with torch.no_grad():
#             summary_ids = model.generate(
#                 **inputs,
#                 max_length=150,
#                 min_length=40,
#                 length_penalty=2.0,
#                 num_beams=4,
#                 early_stopping=True
#             )

#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         st.subheader("📝 Summary")
#         st.success(summary)


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