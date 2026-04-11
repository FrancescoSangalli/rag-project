from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader

def load_documents(path: str):
    if path.endswith(".pdf"):
        loader = PyPDFLoader(path)
    elif path.endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = loader.load()
    print(f"Caricati {len(docs)} documenti")
    return docs