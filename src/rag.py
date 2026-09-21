import re
import streamlit as st

from langchain_ollama import OllamaLLM


# --------------------------------------------------
# MAIN LLM
# --------------------------------------------------

@st.cache_resource
def create_llm():

    return OllamaLLM(
        model="llama3.2:3b",
        num_predict=300
    )


# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

def ask_question(vectorstore, llm, query):

    results = vectorstore.similarity_search(
        query,
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    prompt = f"""
You are IntelliDocs AI.

Answer the question using ONLY the context below.

If the answer is not present in the context, say:

"I couldn't find the answer in the uploaded documents."

Do not guess.

Context:
{context}

Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    return response, results


# --------------------------------------------------
# GENERATE QUIZ
# --------------------------------------------------

def generate_quiz(vectorstore, llm):

    results = vectorstore.similarity_search(
        "HTML CSS forms selectors webpage structure",
        k=3
    )

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    prompt = f"""
Create exactly 5 multiple choice questions using ONLY the study notes below.

Rules:

- Create exactly 5 questions.
- Each question must have exactly 4 options.
- Options must be A, B, C and D.
- Only one answer is correct.
- Do not explain the answers.
- Do not use markdown.

Use this exact format:

Question 1: What is HTML?
A) Programming language
B) Markup language
C) Database
D) Operating system
Answer: B

Question 2: Example question?
A) Option
B) Option
C) Option
D) Option
Answer: A

Question 3: Example question?
A) Option
B) Option
C) Option
D) Option
Answer: C

Question 4: Example question?
A) Option
B) Option
C) Option
D) Option
Answer: D

Question 5: Example question?
A) Option
B) Option
C) Option
D) Option
Answer: A

STUDY NOTES:

{context}
"""

    response = llm.invoke(prompt).strip()

    # Show the raw response in terminal for debugging
    print("\n===== RAW QUIZ RESPONSE =====")
    print(response)
    print("===== END QUIZ RESPONSE =====\n")

    # Remove thinking sections if the model produces them
    response = re.sub(
        r"<think>.*?</think>",
        "",
        response,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # --------------------------------------------------
    # PARSE QUESTIONS
    # --------------------------------------------------

    question_blocks = re.split(
        r"(?=Question\s*\d+\s*[:.)-])",
        response,
        flags=re.IGNORECASE
    )

    questions = []

    for block in question_blocks:

        block = block.strip()

        if not block:
            continue

        # Question text
        question_match = re.search(
            r"Question\s*\d+\s*[:.)-]\s*(.*?)(?=\n\s*A\s*[).:-])",
            block,
            flags=re.IGNORECASE | re.DOTALL
        )

        # Options
        option_a = re.search(
            r"\n\s*A\s*[).:-]\s*(.*?)(?=\n\s*B\s*[).:-])",
            block,
            flags=re.IGNORECASE | re.DOTALL
        )

        option_b = re.search(
            r"\n\s*B\s*[).:-]\s*(.*?)(?=\n\s*C\s*[).:-])",
            block,
            flags=re.IGNORECASE | re.DOTALL
        )

        option_c = re.search(
            r"\n\s*C\s*[).:-]\s*(.*?)(?=\n\s*D\s*[).:-])",
            block,
            flags=re.IGNORECASE | re.DOTALL
        )

        option_d = re.search(
            r"\n\s*D\s*[).:-]\s*(.*?)(?=\n\s*Answer\s*[:.)-])",
            block,
            flags=re.IGNORECASE | re.DOTALL
        )

        answer_match = re.search(
            r"Answer\s*[:.)-]\s*([A-D])",
            block,
            flags=re.IGNORECASE
        )

        # Skip invalid question blocks
        if not all([
            question_match,
            option_a,
            option_b,
            option_c,
            option_d,
            answer_match
        ]):
            continue

        question_text = question_match.group(1).strip()

        options = [
            option_a.group(1).strip(),
            option_b.group(1).strip(),
            option_c.group(1).strip(),
            option_d.group(1).strip()
        ]

        answer = answer_match.group(1).upper()

        # Clean whitespace
        question_text = re.sub(
            r"\s+",
            " ",
            question_text
        )

        options = [
            re.sub(r"\s+", " ", option)
            for option in options
        ]

        # Make sure all four options are different
        if len(set(
            option.lower()
            for option in options
        )) != 4:
            continue

        questions.append({
            "question": question_text,
            "options": options,
            "answer": answer
        })

    print("QUIZ GENERATED:", len(questions), "questions")

    return questions


# --------------------------------------------------
# SUMMARIZE DOCUMENT
# --------------------------------------------------

def summarize_documents(vectorstore, llm):

    results = vectorstore.similarity_search(
        "main topics important concepts key points summary",
        k=5
    )

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    prompt = f"""
Create a clear and concise summary of the study material below.

Rules:

1. Use ONLY the provided study material.
2. Cover the main topics and important concepts.
3. Use simple language.
4. Use headings and bullet points.
5. Do not invent information.

STUDY MATERIAL:

{context}

SUMMARY:
"""

    response = llm.invoke(prompt)

    return response