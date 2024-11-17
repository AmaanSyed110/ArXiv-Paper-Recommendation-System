import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load Dataset
@st.cache_data
def load_data():
    data = pd.read_csv("arxiv_data.csv")  # Ensure your dataset file name matches
    return data

# Compute Embeddings for Papers
@st.cache_resource
def compute_embeddings(texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

# Recommend Papers
def recommend_papers(user_input, embeddings, data, top_n=5):
    user_embedding = model.encode([user_input])
    similarity_scores = cosine_similarity(user_embedding, embeddings)
    top_indices = similarity_scores[0].argsort()[-top_n:][::-1]
    return data.iloc[top_indices]

# Streamlit App UI
st.title("ArXiv Paper Recommendation System")
st.markdown("""
    Welcome to the ArXiv Paper Recommendation System! Enter a research interest or paste an abstract 
    to get relevant paper recommendations from ArXiv.
""")

# Load Data
data = load_data()
titles = data["titles"].tolist()
summaries = data["summaries"].tolist()
corpus = [f"{title}. {summary}" for title, summary in zip(titles, summaries)]

# Compute Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = compute_embeddings(corpus)

# Input Box
user_input = st.text_area("Enter your research interest or a paper abstract:", height=200)

# Generate Recommendations
if st.button("Recommend"):
    if user_input.strip():
        recommendations = recommend_papers(user_input, embeddings, data)
        st.markdown("### Recommended Papers:")
        for _, row in recommendations.iterrows():
            st.write(f"**Title**: {row['titles']}")
            st.write(f"**Abstract**: {row['summaries']}")
            st.write(f"**Categories**: {row['terms']}")
            st.write("---")
    else:
        st.warning("Please enter some text to get recommendations!")
