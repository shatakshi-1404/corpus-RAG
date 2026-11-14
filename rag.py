
import hashlib
import time
from io import BytesIO

import chromadb
from groq import Groq
from pypdf import PdfReader


CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def create_vector_store():
    client = chromadb.PersistentClient(
        path="data/chroma_db"
    )

    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def extract_text(pdf_bytes):
    reader = PdfReader(BytesIO(pdf_bytes))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text
            })

    return pages


def chunk_text(text):
    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def index_pdf(pdf_bytes, filename, embedder, collection):
    pages = extract_text(pdf_bytes)

    documents = []
    metadatas = []
    ids = []

    file_hash = hashlib.md5(pdf_bytes).hexdigest()

    for page in pages:
        chunks = chunk_text(page["text"])

        for index, chunk in enumerate(chunks):
            documents.append(chunk)

            metadatas.append({
                "source": filename,
                "page": page["page"],
                "chunk": index
            })

            ids.append(
                f"{file_hash}_{page['page']}_{index}"
            )

    if not documents:
        raise ValueError("No extractable text found in PDF.")

    embeddings = embedder.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return {
        "pages": len(pages),
        "chunks": len(documents)
    }


def retrieve(question, embedder, collection, top_k=3):
    start_time = time.perf_counter()

    query_embedding = embedder.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    count = collection.count()

    if count == 0:
        return [], 0

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    sources = []

    for doc, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        sources.append({
            "text": doc,
            "metadata": metadata,
            "distance": distance,
            "similarity": 1 - distance
        })

    retrieval_time = (
        time.perf_counter() - start_time
    ) * 1000

    return sources, round(retrieval_time, 2)


def generate_answer(question, sources, api_key, model):
    if not sources:
        return "No relevant information was found.", 0

    # Reject weak retrieval results.
    if sources[0]["distance"] > 0.65:
        return (
            "I couldn't find sufficient relevant "
            "information in your documents."
        ), 0

    context = ""

    for source in sources:
        metadata = source["metadata"]

        context += (
            f"\nSource: {metadata['source']}, "
            f"Page: {metadata['page']}\n"
            f"{source['text']}\n"
        )

    prompt = f"""
You are DocuMind, a document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:
1. Do not invent facts.
2. If the answer is unavailable, say so.
3. Keep the answer clear and concise.
4. Cite sources using [filename, Page X].
5. Distinguish facts from any inference.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    client = Groq(api_key=api_key)

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    generation_time = (
        time.perf_counter() - start_time
    ) * 1000

    answer = response.choices[0].message.content

    return answer, round(generation_time, 2)
