import streamlit as st

from src.pdf_processor import process_pdfs
from src.vectorstore import create_vectorstore
from src.rag import create_llm, ask_question


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="IntelliDocs AI",
    page_icon="📚"
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("📚 IntelliDocs AI")

    st.write(
        "AI-powered document assistant "
        "that answers questions from your PDFs."
    )

    st.divider()

    st.subheader("🛠️ Tech Stack")

    st.write("• Python")
    st.write("• LangChain")
    st.write("• Ollama")
    st.write("• ChromaDB")
    st.write("• Streamlit")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

st.title("📚 IntelliDocs AI")

st.write(
    "Upload your PDF documents and ask questions about them."
)


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)


# --------------------------------------------------
# PROCESS PDFs
# --------------------------------------------------

if uploaded_files:

    file_data = [
        (file.name, file.getvalue())
        for file in uploaded_files
    ]

    with st.spinner("📚 Processing your PDFs..."):

        all_docs, all_chunks = process_pdfs(file_data)

        vectorstore = create_vectorstore(
            all_chunks
        )

    llm = create_llm()

    st.success(
        f"✅ Ready! {len(uploaded_files)} PDF(s) loaded — "
        f"{len(all_docs)} pages and "
        f"{len(all_chunks)} chunks created."
    )

    st.info(
        "💬 Your documents are ready. Ask a question below."
    )


    # --------------------------------------------------
    # DOCUMENT LIST IN SIDEBAR
    # --------------------------------------------------

    with st.sidebar:

        st.subheader("📄 Documents")

        for file in uploaded_files:

            st.write(
                f"• {file.name}"
            )


    # --------------------------------------------------
    # DISPLAY CHAT HISTORY
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.write(message["content"])


    # --------------------------------------------------
    # QUESTION
    # --------------------------------------------------

    query = st.chat_input(
        "Ask a question about your PDFs..."
    )


    if query:

        # Save user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )


        # Display user message
        with st.chat_message("user"):

            st.write(query)


        # --------------------------------------------------
        # RAG
        # --------------------------------------------------

        response, results = ask_question(
            vectorstore,
            llm,
            query
        )


        # --------------------------------------------------
        # DISPLAY ANSWER + SOURCES
        # --------------------------------------------------

        with st.chat_message("assistant"):

            st.write(response)

            st.markdown("### 📚 Sources")

            sources = set()

            for doc in results:

                source = doc.metadata.get("source")
                page = doc.metadata.get("page")

                sources.add(
                    (source, page)
                )

            for source, page in sources:

                st.write(
                    f"📄 **{source}** — Page {page + 1}"
                )


        # Save assistant response
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )