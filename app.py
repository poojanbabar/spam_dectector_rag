import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

st.title("Spam Detector (RAG + AI)")

# Data
texts = [
    "Win a free iPhone now",
    "Call me later",
    "Limited offer just for you",
    "Meeting at 5 PM",
    "Congratulations you won lottery"
]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings
embeddings = model.encode(texts)

# FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Retrieve
def retrieve(query):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k=2)
    return [texts[i] for i in I[0]]

# Classify
def classify(query):
    if any(word in query.lower() for word in ["free", "offer", "win", "money"]):
        return "spam"
    return "not spam"

# UI
user_input = st.text_input("Enter message")

if user_input:
    result = classify(user_input)
    
    if result == "spam":
        st.error("Spam 🚫")
    else:
        st.success("Not Spam ✅")
