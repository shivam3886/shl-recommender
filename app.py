import streamlit as st
from sentence_transformers import SentenceTransformer, util

# SHL Tests and Descriptions
shl_tests = {
    "Cognitive Ability Test": "Measures problem-solving, logical reasoning, and analytical thinking.",
    "Personality Questionnaire": "Evaluates traits like extraversion, conscientiousness, and emotional stability.",
    "Situational Judgment Test": "Assesses judgment in work-related scenarios.",
    "Numerical Reasoning Test": "Tests the ability to interpret and analyze numerical data.",
    "Verbal Reasoning Test": "Assesses understanding and interpretation of written information.",
    "Inductive Reasoning Test": "Measures ability to identify patterns and logical rules.",
    "Deductive Reasoning Test": "Tests logical thinking using provided information.",
    "Technical Test": "Evaluates specific technical skills required for a job.",
    "Language Proficiency Test": "Measures reading, writing, listening, and speaking skills in a specific language.",
    "Motivation Questionnaire": "Identifies what drives and motivates a candidate."
}

# Load model with error handling
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        st.stop()

model = load_model()

# Streamlit App
st.title("SHL Assessment Recommender")

user_input = st.text_area("Describe your interests or strengths:")

if st.button("Recommend SHL Tests"):
    if not user_input.strip():
        st.warning("Please enter a description of your interests or strengths.")
    else:
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        results = []

        for test, description in shl_tests.items():
            test_embedding = model.encode(description, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(user_embedding, test_embedding).item()
            results.append((test, description, similarity))

        results.sort(key=lambda x: x[2], reverse=True)

        st.subheader("Top SHL Tests Recommended:")
        for test, description, score in results[:3]:
            st.markdown(f"### {test}")
            st.write(description)
            st.caption(f"Similarity Score: {score:.2f}")
