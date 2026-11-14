
# DocuMind AI

A document-based Retrieval-Augmented Generation
application built using Python and Streamlit.

## Features

- PDF document ingestion
- Text extraction and chunking
- HuggingFace embeddings
- ChromaDB vector storage
- Semantic similarity search
- Context-grounded answer generation
- Source and page references
- Retrieval inspection
- Retrieval hit-rate evaluation

## Tech Stack

Python, Streamlit, PyPDF, Sentence Transformers,
ChromaDB, Groq and Pytest.

## RAG Pipeline

PDF -> Text Extraction -> Chunking -> Embeddings
-> ChromaDB -> Semantic Retrieval -> Groq LLM
-> Grounded Answer

## Run Locally

1. Clone the repository.
2. Create a virtual environment.
3. Install dependencies:

   pip install -r requirements.txt

4. Configure GROQ_API_KEY in .env.
5. Start the application:

   streamlit run app.py

## Evaluation

The application includes a manually labelled
retrieval evaluation using expected phrases.

It measures retrieval hit rate on a custom test set.

## Future Improvements

- Hybrid search
- Reranking
- Multi-document filtering
- Better RAG evaluation
