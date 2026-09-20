##retrieval + prompt + Gemma
from langchain_ollama import OllamaLLM

import streamlit as st

from langchain_ollama import OllamaLLM


@st.cache_resource
def create_llm():

    llm = OllamaLLM(
        model="gemma:2b"
    )

    return llm


def ask_question(vectorstore, llm, query):

    # Retrieve relevant chunks
    results = vectorstore.similarity_search(
        query,
        k=3
    )

    # Combine retrieved chunks
    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    # RAG prompt
    prompt = f"""
You are IntelliDocs AI, a document question-answering assistant.

Answer the question using ONLY the context provided below.

Do not use outside knowledge.

If the answer is not present in the context, say:

"I couldn't find the answer in the uploaded documents."

Do not guess or make up information.

Context:
{context}

Question:
{query}

Answer:
"""

    # Generate answer
    response = llm.invoke(prompt)

    return response, results