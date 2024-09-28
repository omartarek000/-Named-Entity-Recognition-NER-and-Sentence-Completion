import streamlit as st
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForMaskedLM
from transformers import pipeline
import torch

@st.cache_resource
def load_ner_model():
    tokenizer = AutoTokenizer.from_pretrained("eventdata-utd/conflibert-named-entity-recognition")
    model = AutoModelForTokenClassification.from_pretrained("eventdata-utd/conflibert-named-entity-recognition")
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
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]
    
    results = []
    for idx in mask_token_index:
        with torch.no_grad():
            output = model(input_ids)
        
        mask_token_logits = output.logits[0, idx, :]
        top_k = 5
        top_k_tokens = torch.topk(mask_token_logits, top_k, dim=-1).indices.tolist()
        predicted_words = [tokenizer.decode([token_id]) for token_id in top_k_tokens]
        results.append(predicted_words)
    
    return results

st.title("NER and Masked Language Model Prediction")

# Load models
ner_pipeline = load_ner_model()
mlm_tokenizer, mlm_model = load_mlm_model()

# Create tabs
ner_tab, mlm_tab = st.tabs(["Named Entity Recognition", "Masked Language Model"])

with ner_tab:
    st.header("Named Entity Recognition")
    ner_input = st.text_area("Enter text for NER:", 
                             "John Doe works at OpenAI and lives in San Francisco.", 
                             height=150)
    if st.button("Perform NER"):
        entities = perform_ner(ner_input, ner_pipeline)
        for entity in entities:
            st.write(f"Entity: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']:.4f}")

with mlm_tab:
    st.header("Masked Language Model Prediction")
    mlm_input = st.text_area("Enter text with [MASK] (you can use multiple masks):", 
                             "The [MASK] brown [MASK] jumps over the lazy dog.", 
                             height=150)
    if st.button("Predict Masked Words"):
        if "[MASK]" in mlm_input:
            predicted_words_list = perform_mlm(mlm_input, mlm_tokenizer, mlm_model)
            for i, predicted_words in enumerate(predicted_words_list):
                st.write(f"Predictions for MASK {i+1}:")
                for j, word in enumerate(predicted_words):
                    st.write(f"  Top {j+1} predicted word: {word}")
        else:
            st.warning("Please include at least one [MASK] in your input text.")

st.sidebar.info("This app demonstrates Named Entity Recognition and Masked Language Model prediction using Hugging Face Transformers.")





st.markdown("---")
st.markdown(" this app was made by Ayman Saber && Omar Tarek ")