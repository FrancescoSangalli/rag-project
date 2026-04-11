from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")

def embed_texts(texts: list) -> list:
    embeddings = model.encode(texts, show_progress_bar=False)
    return embeddings.tolist()