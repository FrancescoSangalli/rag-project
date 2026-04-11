from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["source"] = chunk.metadata.get("source", "unknown")
    print(f"Generati {len(chunks)} chunk")
    return chunks