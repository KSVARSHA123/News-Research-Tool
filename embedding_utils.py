import os
from sentence_transformers import SentenceTransformer

def embed_texts(texts):
    # Load the Hugging Face API key from environment variables
    hf_api_key = os.getenv('HF_API_KEY')

    # Define the SentenceTransformer model
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model = SentenceTransformer(embedding_model_name, use_auth_token=hf_api_key)

    # Encode the texts to obtain embeddings
    embeddings = embedding_model.encode(texts)
    
    return embeddings
