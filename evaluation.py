
def evaluate_retrieval(
    test_cases,
    embedder,
    collection,
    top_k=3
):
    hits = 0
    results = []

    for case in test_cases:
        question = case["question"]
        expected = case["expected_phrase"].lower()

        query_embedding = embedder.encode(
            question,
            normalize_embeddings=True
        ).tolist()

        retrieved = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents"]
        )

        documents = retrieved["documents"][0]

        found = any(
            expected in document.lower()
            for document in documents
        )

        if found:
            hits += 1

        results.append({
            "question": question,
            "hit": found
        })

    hit_rate = (
        hits / len(test_cases) * 100
        if test_cases else 0
    )

    return {
        "hit_rate": round(hit_rate, 2),
        "hits": hits,
        "total": len(test_cases),
        "results": results
    }
