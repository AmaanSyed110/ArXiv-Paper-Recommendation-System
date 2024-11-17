# ArXiv-Paper-Recommendation-System

## Overview
The **ArXiv Paper Recommendation System** is a web application designed to help researchers discover relevant papers from the ArXiv repository based on their research interests or paper abstracts. By using sentence embeddings and cosine similarity, the system matches user input with the most relevant papers in the dataset, ensuring accurate recommendations. Built with Streamlit for a simple user interface, this system leverages advanced NLP techniques via the SentenceTransformers library to compute high-quality embeddings of research paper abstracts. The app is ideal for researchers looking to explore papers that align with their interests, making it easy to stay up-to-date with the latest research.

## Features
- **Content-Based Filtering**: Recommends papers using textual analysis of abstracts.
- **User Profiling**: Captures user preferences for personalized recommendations.
- **Natural Language Processing**: Extracts and processes key features from paper abstracts.
- **Interactive Interface**: Provides an easy-to-use system for researchers.
- **Top-N Recommendations**: Get up to 5 highly relevant research papers.

## Tech Stack
- **Python**: The programming language used for the development of the entire project.
- **Streamlit**: An open-source Python library for building interactive web applications.
- **SentenceTransformers**: A Python library that provides pre-trained models for transforming sentences into dense vector representations (embeddings).
- **scikit-learn**: A popular machine learning library in Python, used here for calculating cosine similarity between user input and paper embeddings to determine the most relevant papers.
