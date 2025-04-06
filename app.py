import streamlit as st
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Load CSV file once (assumes it's in the same directory as this script)
@st.cache_data
def load_data():
    return pd.read_csv("shl_assessments_sample.csv")

# Load the model once
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load everything
df = load_data()
model = load_model()

# Generate embeddings if not already present
@st.cache_data
def generate_embeddings(df):
    df['embedding'] = df['Description'].apply(lambda x: model.encode(x, convert_to_tensor=True))
    return df

df = generate_embeddings(df)

def recommend_assessments(query, top_k=5):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = [torch.nn.functional.cosine_similarity(query_embedding, emb, dim=0).item() for emb in df['embedding']]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    return df.iloc[top_indices][['Assessment Name', 'URL']]

# Streamlit interface
st.title("🔍 SHL Assessment Recommender")

user_query = st.text_input("Enter job description or requirement:")

if user_query:
    recommendations = recommend_assessments(user_query)
    st.subheader("Top Recommended Assessments:")
    for idx, row in recommendations.iterrows():
        st.markdown(f"**{row['Assessment Name']}**  \n[Visit Assessment]({row['URL']})")
