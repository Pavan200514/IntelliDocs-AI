##embedding + ChromaDB
import streamlit as st

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


@st.cache_resource
def create_vectorstore(all_chunks):

    embeddings = OllamaEmbeddings(
        model="qwen3-embedding:0.6b"
    )

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name="intellidocs_app"
    )

    return vectorstore