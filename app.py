import streamlit as st

from src.pdf_processor import process_pdfs
from src.vectorstore import create_vectorstore
from src.rag import (
    create_llm,
    ask_question,
    generate_quiz,
    summarize_documents
)


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="IntelliDocs AI",
    page_icon="📚"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = None

if "summary" not in st.session_state:
    st.session_state.summary = None


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
        st.session_state.quiz = None
        st.session_state.quiz_score = None
        st.session_state.summary = None

        st.rerun()


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

st.markdown(
    """
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <h1>📚 IntelliDocs AI</h1>
        <p style="font-size:18px;">
            Your AI-powered study assistant for PDF documents
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.info(
    "📄 Upload your study PDFs below, then ask questions, "
    "generate quizzes, or create a summary."
)


# --------------------------------------------------
# PDF UPLOAD
# --------------------------------------------------

st.markdown("### 📄 Upload Your Documents")

uploaded_files = st.file_uploader(
    "Choose one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple study documents at once."
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

        try:
            all_docs, all_chunks = process_pdfs(
                file_data
            )

            if not all_chunks:
                st.error("❌ No readable text was found in the uploaded PDFs.")
                st.stop()

            vectorstore = create_vectorstore(
                all_chunks
            )

        except Exception as e:
            st.error(
                "❌ Something went wrong while processing the PDF. "
                "Please check the file and try again."
            )

            st.exception(e)
            st.stop()
    llm = create_llm()


    # --------------------------------------------------
    # READY MESSAGE
    # --------------------------------------------------

    st.success(
        f"✅ {len(uploaded_files)} PDF(s) loaded successfully"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "📄 Documents",
            len(uploaded_files)
        )

    with col2:
        st.metric(
            "📑 Pages",
            len(all_docs)
        )

    with col3:
        st.metric(
            "🧩 Chunks",
            len(all_chunks)
        )

    st.info(
        "💬 Your documents are ready. Ask a question below."
    )


    # --------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------

    with st.sidebar:

        st.divider()

        st.subheader("📄 Uploaded Documents")

        for file in uploaded_files:

            st.success(
                f"📄 {file.name}"
            )


    # --------------------------------------------------
    # QUIZ + SUMMARY BUTTONS
    # --------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🧠 Generate Quiz",
            use_container_width=True
        ):

            with st.spinner(
                "🧠 Generating quiz..."
            ):

                try:
                    st.session_state.quiz = generate_quiz(
                        vectorstore,
                        llm
                    )

                    st.session_state.quiz_score = None

                except Exception as e:
                    st.error(
                        "❌ Quiz generation failed. Please try again."
                    )

                    st.exception(e)
                    st.session_state.quiz = None

    with col2:

        if st.button(
            "📝 Summarize Documents",
            use_container_width=True
        ):

            with st.spinner(
                "📝 Creating summary..."
            ):

                try:
                    st.session_state.summary = summarize_documents(
                        vectorstore,
                         llm
                    )

                except Exception as e:
                    st.error(
                    "❌ Summary generation failed. Please try again."
                    )

                    st.exception(e)
                    st.session_state.summary = None


    # --------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------

    if st.session_state.summary:

        st.divider()

        st.markdown("### 📝 Document Summary")

        st.write(
            st.session_state.summary
        )


    # --------------------------------------------------
    # DISPLAY QUIZ
    # --------------------------------------------------

    if st.session_state.quiz:

        quiz = st.session_state.quiz

        st.divider()

        st.markdown("### 🧠 Your Quiz")


        # Display questions
        for i, item in enumerate(quiz):

            with st.expander(
                f"Question {i + 1}: {item['question']}",
                expanded=True
            ):

                options = item["options"]

                st.radio(
                    "Choose your answer:",
                    [
                        f"A) {options[0]}",
                        f"B) {options[1]}",
                        f"C) {options[2]}",
                        f"D) {options[3]}"
                    ],
                    key=f"quiz_question_{i}"
                )


        # --------------------------------------------------
        # SUBMIT QUIZ
        # --------------------------------------------------

        if st.button(
            "✅ Submit Quiz",
            type="primary",
            use_container_width=True
        ):

            score = 0

            for i, item in enumerate(quiz):

                user_answer = st.session_state[
                    f"quiz_question_{i}"
                ]

                selected_letter = (
                    user_answer
                    .split(")", 1)[0]
                    .strip()
                    .upper()
                )

                correct_answer = (
                    item["answer"].upper()
                )

                if selected_letter == correct_answer:
                    score += 1

            st.session_state.quiz_score = score


        # --------------------------------------------------
        # SHOW SCORE
        # --------------------------------------------------

        if st.session_state.quiz_score is not None:

            score = st.session_state.quiz_score
            total = len(quiz)

            st.divider()

            st.subheader("🎯 Quiz Result")

            st.metric(
                "Your Score",
                f"{score} / {total}"
            )

            if score == total:

                st.success(
                    "🏆 Perfect score! Excellent work!"
                )

            elif score >= total * 0.6:

                st.success(
                    "🎉 Good job! Keep practicing."
                )

            else:

                st.info(
                    "📚 Keep practicing and try the quiz again."
                )


    # --------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


    # --------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------

    st.divider()

    st.subheader("💬 Ask Your Documents")

    st.caption(
        "Ask anything about the information inside your uploaded PDFs."
    )

    query = st.chat_input(
        "Ask a question about your PDFs..."
    )


    # --------------------------------------------------
    # PROCESS QUESTION
    # --------------------------------------------------

    if query:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": query
            }
        )


        with st.chat_message("user"):

            st.write(query)


        # --------------------------------------------------
        # RAG
        # --------------------------------------------------

        try:
            response, results = ask_question(
                vectorstore,
                llm,
                query
            )

        except Exception as e:
            st.error(
               "❌ I couldn't generate an answer right now. "
               "Please try again."
            )

            st.exception(e)
            response = None
            results = []


        # --------------------------------------------------
        # DISPLAY ANSWER
        # --------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(
                "### 🤖 IntelliDocs AI"
            )
            if response:
                st.write(response)

                st.divider()

                st.markdown(
                    "#### 📚 Sources"
                )

            sources = set()

            for doc in results:

                source = doc.metadata.get(
                    "source"
                )

                page = doc.metadata.get(
                    "page"
                )

                sources.add(
                    (source, page)
                )

            for source, page in sources:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"📄 **{source}**"
                    )

                    st.caption(
                        f"Page {page + 1}"
                    )


        # --------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # --------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )