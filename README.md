# IntelliDocs AI

IntelliDocs AI is a document-based AI assistant that lets users upload PDF files and ask questions about their content.

The project uses Retrieval-Augmented Generation (RAG) to retrieve relevant parts of the uploaded documents and provide answers using a local LLM.

## Features

- Upload multiple PDF files
- Ask questions about the uploaded documents
- Get answers with PDF name and page number
- Generate MCQ quizzes from the uploaded content
- Generate summaries of uploaded documents
- Local embeddings using Ollama
- ChromaDB for vector storage
- Streamlit interface

## How It Works

```text
PDF Files
   ↓
Extract Text
   ↓
Split into Chunks
   ↓
Create Embeddings
   ↓
Store in ChromaDB
   ↓
User Question
   ↓
Similarity Search
   ↓
Relevant Chunks
   ↓
Local LLM
   ↓
Answer + Sources

RAG Pipeline
The application follows these steps:

1.Upload one or more PDF files.
2.Extract text from the PDFs using PyPDF.
3.Split the extracted text into smaller chunks.
4.Create embeddings for the chunks.
5.Store the embeddings in ChromaDB.
6.When a question is asked, perform similarity search to find relevant chunks.
7.Send the retrieved content along with the question to the local LLM.
8.Generate an answer based on the retrieved content.
9.Display the answer along with the document and page information.


Project Structure

IntelliDocs-AI/
│
├── data/
├── notebooks/
│   └── 01_test_ollama.ipynb
├── src/
│   ├── pdf_processor.py
│   ├── vectorstore.py
│   └── rag.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md


Technologies Used
->Python
->LangChain
->Ollama
->ChromaDB
->Streamlit
->PyPDF
->Local Embeddings

Setup
1. Clone the repository
git clone https://github.com/Pavan200514/IntelliDocs-AI.git
cd IntelliDocs-AI
2. Create the environment
conda create -n genai python=3.11
conda activate genai
3. Install dependencies
pip install -r requirements.txt

4. Install and run Ollama
Install Ollama and make sure it is running.
Then download the models:

ollama pull llama3.2:3b
ollama pull qwen3-embedding:0.6b

5. Run the application
streamlit run app.py

The application will open in the browser.

Usage
1.Upload one or more PDF files.
2.Wait for the files to be processed.
3.Ask questions about the documents.
4.Check the sources shown with the answer.
5.Use the quiz option to generate MCQs.
6.Use the summary option to get a summary of the documents.

Privacy
The project uses Ollama for local LLM inference and local embeddings, so the application can be run locally without requiring a cloud LLM API.

Future Improvements
->Conversation memory
->Better document management
->Persistent vector database
->Authentication
->Document-specific chat
->Better source highlighting
->Deployment


Author
Pavan Mangali
Electronics & Communication Engineering student interested in Software Development and Generative AI.