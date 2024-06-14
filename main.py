import os
import streamlit as st
import pickle
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from sentence_transformers import SentenceTransformer
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.schema import Document

from dotenv import load_dotenv
from embedding_utils import embed_texts

load_dotenv()  # Load environment variables from .env file

# Load the Hugging Face API key
hf_api_key = os.getenv('HF_API_KEY')

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

main_placeholder = st.empty()

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    main_placeholder.text("Data Loading...Completed! ✅✅✅")

    # Split data into documents
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    main_placeholder.text("Text Splitting...Completed! ✅✅✅")

    # Extract text content from documents
    texts = [doc.page_content for doc in docs]

    # Embedding the texts
    embeddings = embed_texts(texts)

    # Initialize FAISS vector store
    index = "IVF256,Flat"
    docstore = {idx: doc for idx, doc in enumerate(texts)}
    index_to_docstore_id = {idx: idx for idx in range(len(texts))}

    # Initialize FAISS vector store with chosen components
    vectorstore_hf = FAISS(embedding_function=embed_texts, index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
    main_placeholder.text("Embedding Vector Building...Completed! ✅✅✅")

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hf, f)
    main_placeholder.text(f"FAISS Index Saved to {file_path}")

# Query input
query = st.text_input("Enter your question:")

# Process query if submitted
if st.button("Ask"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

            # Initialize QA pipeline
            qa_pipeline = pipeline(
                "question-answering", 
                model=AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad"), 
                tokenizer=AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
            )

            # Use the vector store retriever for the RetrievalQAWithSourcesChain
            chain = RetrievalQAWithSourcesChain.from_llm(llm=qa_pipeline, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            # Display answer
            st.subheader("Answer:")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", [])
            if sources:
                st.subheader("Sources:")
                for source in sources:
                    st.write(source)