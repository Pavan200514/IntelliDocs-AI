##reading + chunking code

import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def process_pdfs(file_data):

    all_docs = []

    for file_name, file_bytes in file_data:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(file_bytes)
            pdf_path = temp_file.name

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        # Keep original PDF filename
        for doc in docs:
            doc.metadata["source"] = file_name

        all_docs.extend(docs)


    # Create chunks

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    all_chunks = text_splitter.split_documents(all_docs)

    return all_docs, all_chunks