
import os

import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from evaluation import evaluate_retrieval
from rag import (
    create_vector_store,
    index_pdf,
    retrieve,
    generate_answer
)


# =====================================================
# CONFIGURATION
# =====================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

st.set_page_config(
    page_title="CORPUS | Document Intelligence Engine",
    page_icon="C",
    layout="wide"
)


# =====================================================
# CUSTOM UI
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b1020;
    color: #e5e7eb;
}

[data-testid="stSidebar"] {
    background: #11182a;
    border-right: 1px solid #26334c;
}

[data-testid="stMetric"] {
    background: #151f34;
    padding: 18px;
    border: 1px solid #293650;
    border-radius: 14px;
}

.stButton > button {
    background: linear-gradient(90deg, #635bff, #3984ff);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 600;
    min-height: 42px;
}

.stButton > button:hover {
    border: 1px solid #8b83ff;
    color: white;
}

.hero {
    background: linear-gradient(120deg, #192746, #27204b);
    border: 1px solid #3c4770;
    padding: 36px;
    border-radius: 18px;
    margin-bottom: 25px;
}

.hero-label {
    color: #a5a0ff;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    margin-bottom: 16px;
}

.hero h1 {
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 10px;
    color: white;
}

.hero p {
    color: #c2c9df;
    font-size: 15px;
    max-width: 650px;
}

.brand-name {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: 3px;
    color: white;
}

.brand-subtitle {
    font-size: 10px;
    letter-spacing: 2px;
    color: #9ca9c7;
    margin-top: 4px;
}

.section-label {
    color: #9ca9c7;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}

div[data-testid="stChatMessage"] {
    background: #141e32;
    border: 1px solid #273550;
    border-radius: 14px;
    padding: 12px;
}

.stTextArea textarea {
    background: #11182a;
    color: white;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD RESOURCES
# =====================================================

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    collection = create_vector_store()

    return embedder, collection


with st.spinner("Initializing Corpus intelligence engine..."):
    embedder, collection = load_resources()


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 10px 0 20px 0;">
        <div class="brand-name">CORPUS</div>
        <div class="brand-subtitle">
            DOCUMENT INTELLIGENCE ENGINE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload your PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button(
        "Process Documents",
        use_container_width=True
    ):

        if not uploaded_files:
            st.warning("Upload a PDF document first.")

        else:
            for file in uploaded_files:

                try:
                    with st.spinner(f"Indexing {file.name}..."):

                        stats = index_pdf(
                            file.getvalue(),
                            file.name,
                            embedder,
                            collection
                        )

                    if file.name not in st.session_state.indexed_files:
                        st.session_state.indexed_files.append(file.name)

                    st.success(
                        f"{file.name}: "
                        f"{stats['chunks']} chunks indexed"
                    )

                except Exception as error:
                    st.error(f"Failed to process {file.name}: {error}")

    st.divider()

    st.markdown("### Retrieval Settings")

    top_k = st.slider(
        "Number of retrieved chunks",
        min_value=1,
        max_value=6,
        value=3
    )

    st.caption(
        "Top-K determines how many relevant document "
        "chunks are retrieved for answer generation."
    )

    st.divider()

    st.markdown("### Indexed Documents")

    if st.session_state.indexed_files:

        for filename in dict.fromkeys(
            st.session_state.indexed_files
        ):
            st.write(f"📄 {filename}")

    else:
        st.caption("Your knowledge base is currently empty.")

    st.caption(
        f"Total indexed chunks: {collection.count()}"
    )

    if st.button("Clear Chat", use_container_width=True):

        st.session_state.messages = []
        st.session_state.last_metrics = None

        st.rerun()


# =====================================================
# MAIN DASHBOARD
# =====================================================

st.markdown("""
<div class="hero">
    <div class="hero-label">
        DOCUMENT INTELLIGENCE ENGINE
    </div>

    <h1>Knowledge, extracted.</h1>

    <p>
        A RAG-powered platform to read, retrieve and reason
        over your documents.
    </p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# METRICS
# =====================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Indexed Chunks",
    collection.count()
)

col2.metric(
    "Documents Uploaded",
    len(set(st.session_state.indexed_files))
)

if st.session_state.last_metrics:

    col3.metric(
        "Last Retrieval",
        f"{st.session_state.last_metrics['retrieval']} ms"
    )

else:
    col3.metric("Last Retrieval", "—")


st.divider()


# =====================================================
# WELCOME SCREEN
# =====================================================

if not st.session_state.messages:

    st.markdown("### Welcome to Corpus")

    st.write(
        "Your documents contain knowledge. "
        "Ask questions to retrieve it."
    )

    suggestions = [
        "Summarize the main concepts in this document.",
        "Explain the most important topics.",
        "What are the key takeaways?",
        "Find a specific concept and explain it."
    ]

    cols = st.columns(2)

    for index, suggestion in enumerate(suggestions):

        with cols[index % 2]:
            st.info(suggestion)


# =====================================================
# CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("View retrieved sources"):

                for source in message["sources"]:

                    metadata = source["metadata"]

                    st.markdown(
                        f"**{metadata['source']}** "
                        f"— Page {metadata['page']}"
                    )

                    st.caption(
                        f"Similarity: {source['similarity']:.3f}"
                    )

                    st.write(source["text"])
                    st.divider()

        if message.get("metrics"):

            st.caption(
                f"Retrieval: {message['metrics']['retrieval']} ms"
                f" | Generation: {message['metrics']['generation']} ms"
            )


# =====================================================
# CHAT INPUT
# =====================================================

question = st.chat_input(
    "Ask anything about your documents..."
)

if question:

    if collection.count() == 0:

        st.warning(
            "Upload and process a PDF before asking questions."
        )

    elif not GROQ_API_KEY:

        st.error(
            "Groq API key missing. "
            "Configure GROQ_API_KEY in your .env file."
        )

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        try:

            with st.spinner("Retrieving knowledge..."):

                # Step 1: Retrieve relevant document chunks

                sources, retrieval_time = retrieve(
                    question,
                    embedder,
                    collection,
                    top_k
                )

                # Step 2: Generate an answer using Groq

                answer, generation_time = generate_answer(
                    question,
                    sources,
                    GROQ_API_KEY,
                    GROQ_MODEL
                )

            metrics = {
                "retrieval": retrieval_time,
                "generation": generation_time
            }

            st.session_state.last_metrics = metrics

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "metrics": metrics
            })

        except Exception as error:

            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "Something went wrong while processing "
                    "your question. Please try again."
                )
            })

            st.error(f"Error: {error}")

        st.rerun()


# =====================================================
# RETRIEVAL EVALUATION
# =====================================================

with st.expander("Evaluate RAG Retrieval"):

    st.write(
        "Measure how effectively Corpus retrieves "
        "the expected information from your documents."
    )

    st.caption(
        "Enter one test case per line using: "
        "question | expected phrase"
    )

    st.code(
        "What is dependency injection? | dependency injection"
    )

    raw_cases = st.text_area(
        "Evaluation test cases",
        placeholder=(
            "What is FastAPI? | FastAPI\n"
            "Explain dependency injection | dependency injection"
        )
    )

    if st.button("Run Evaluation"):

        cases = []

        for line in raw_cases.splitlines():

            if "|" in line:

                question_text, expected = line.split("|", 1)

                if question_text.strip() and expected.strip():

                    cases.append({
                        "question": question_text.strip(),
                        "expected_phrase": expected.strip()
                    })

        if not cases:

            st.warning("Enter at least one valid test case.")

        elif collection.count() == 0:

            st.warning(
                "Index at least one document before running evaluation."
            )

        else:

            with st.spinner("Evaluating retrieval performance..."):

                result = evaluate_retrieval(
                    cases,
                    embedder,
                    collection,
                    top_k=top_k
                )

            st.divider()

            st.markdown("### Evaluation Results")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Retrieval Hit Rate",
                f"{result['hit_rate']}%"
            )

            col2.metric(
                "Successful Retrievals",
                result["hits"]
            )

            col3.metric(
                "Total Test Cases",
                result["total"]
            )

            st.divider()

            for item in result["results"]:

                if item["hit"]:
                    st.success(
                        f"PASS: {item['question']}"
                    )

                else:
                    st.error(
                        f"MISS: {item['question']}"
                    )
