import chromadb
from embeddings import embed_texts
import uuid

def get_collection(persist_dir="./chroma_db", collection_name="documents"):
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

def index_chunks(chunks, collection):
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]  # ID univoco garantito
    embeddings = embed_texts(texts)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Indicizzati {len(chunks)} chunk")

def collection_is_empty(collection) -> bool:
    return collection.count() == 0

def get_indexed_files(collection) -> list:
    """Restituisce la lista dei file già indicizzati"""
    try:
        results = collection.get(include=["metadatas"])
        if not results['metadatas']:
            return []
        return list(set([m.get('source') for m in results['metadatas']]))
    except Exception as e:
        print(f"Errore nel leggere i file indicizzati: {e}")
        return []