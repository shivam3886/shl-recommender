import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model with caching
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Load SHL dataset
@st.cache_data
def load_data():
    df = pd.read_csv("shl_assessments.csv")
    return df

df = load_data()

# Streamlit UI
st.title("🔍 SHL Assessment Recommender")
st.markdown("Enter your interests or career goals, and we'll recommend the best SHL assessments for you.")

# User input
user_input = st.text_area("What are your career interests?", height=150)

if st.button("Get Recommendations"):
    if not user_input.strip():
        st.warning("Please enter your interests before clicking the button.")
    else:
        # Encode user input and dataset
        user_embedding = model.encode([user_input])
        assessment_embeddings = model.encode(df['assessment'], show_progress_bar=True)

        # Calculate similarity
        similarity_scores = cosine_similarity(user_embedding, assessment_embeddings)[0]

        # Add similarity to dataframe and sort
        df['score'] = similarity_scores
        top_matches = df.sort_values(by='score', ascending=False).head(5)

        # Show results
        st.subheader("Top SHL Assessment Recommendations:")
        for idx, row in top_matches.iterrows():
            st.markdown(f"**{row['assessment']}**")
            st.markdown(f"{row['description']}")
            st.markdown("---")
