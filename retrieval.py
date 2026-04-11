from embeddings import embed_texts

def retrieve_context(query: str, collection, top_k=5):
    query_embedding = embed_texts([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    context_chunks = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = round(1 - dist, 3)
        if similarity >= 0.4:  # filtra contesto irrilevante
            context_chunks.append({
                "rank": i + 1,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page", "N/A"),
                "similarity": similarity
            })
    return context_chunks