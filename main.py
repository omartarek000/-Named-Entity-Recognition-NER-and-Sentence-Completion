import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM
from transformers import pipeline
import torch

@st.cache_resource
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
    return pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

@st.cache_resource
def load_mlm_model():
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
    return tokenizer, model

def perform_ner(text, ner_pipeline):
    entities = ner_pipeline(text)
    return entities

def perform_mlm(text, tokenizer, model):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    masked_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    
    with torch.no_grad():
        output = model(input_ids)
    
    masked_token_logits = output.logits[0, masked_index, :]
    top_k = 5
    top_k_tokens = torch.topk(masked_token_logits, top_k, dim=-1).indices[0].tolist()
    predicted_words = [tokenizer.decode([token_id]) for token_id in top_k_tokens]
    return predicted_words

st.title("NER and Masked Language Model Prediction")

# Load models
ner_pipeline = load_ner_model()
mlm_tokenizer, mlm_model = load_mlm_model()

# Create tabs
ner_tab, mlm_tab = st.tabs(["Named Entity Recognition", "Masked Language Model"])

with ner_tab:
    st.header("Named Entity Recognition")
    ner_input = st.text_area("Enter text for NER:", "John Doe works at OpenAI and lives in San Francisco.")
    if st.button("Perform NER"):
        entities = perform_ner(ner_input, ner_pipeline)
        for entity in entities:
            st.write(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}")

with mlm_tab:
    st.header("Masked Language Model Prediction")
    mlm_input = st.text_input("Enter text with [MASK]:", "The quick brown [MASK] jumps over the lazy dog.")
    if st.button("Predict Masked Word"):
        if "[MASK]" in mlm_input:
            predicted_words = perform_mlm(mlm_input, mlm_tokenizer, mlm_model)
            for i, word in enumerate(predicted_words):
                st.write(f"Top {i+1} predicted word: {word}")
        else:
            st.warning("Please include [MASK] in your input text.")

st.sidebar.info("This app demonstrates Named Entity Recognition and Masked Language Model prediction using Hugging Face Transformers.")