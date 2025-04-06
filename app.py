import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Page title
st.title("SHL Assessment Recommender")

# Load the sentence transformer model without cache
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
        return model
    except Exception as e:
        st.error("Error loading the model. Please check your internet connection or model name.")
        st.stop()

model = load_model()

# Sample SHL test descriptions
shl_tests = {
    "Deductive Reasoning": "Test your ability to apply general rules to specific problems.",
    "Inductive Reasoning": "Measure how well you can identify patterns and logical rules.",
    "Numerical Reasoning": "Assess your ability to interpret, analyze and draw conclusions from numerical data.",
    "Verbal Reasoning": "Evaluate how well you understand and reason using concepts framed in words.",
    "Situational Judgment": "Understand how you might behave in work-related situations.",
    "Personality Questionnaire": "Explore your personality traits and behavioral preferences.",
    "General Ability": "A mix of numerical, verbal, and logical reasoning skills.",
    "Mechanical Comprehension": "Test your understanding of mechanical and physical concepts."
}

# Convert test descriptions to a DataFrame
test_df = pd.DataFrame(list(shl_tests.items()), columns=["Test Name", "Description"])

# Input: User's interest/skill
user_input = st.text_area("Describe your interests or strengths:", "")

if st.button("Recommend SHL Tests"):
    if user_input.strip() == "":
        st.warning("Please enter some text describing your interests or strengths.")
    else:
        # Embed input and test descriptions
        with st.spinner("Finding the best match..."):
            user_embedding = model.encode([user_input])
            test_embeddings = model.encode(test_df["Description"].tolist())

            # Calculate cosine similarity
            similarities = cosine_similarity(user_embedding, test_embeddings)[0]
            test_df["Similarity"] = similarities

            # Sort and display top 3 recommendations
            top_matches_
