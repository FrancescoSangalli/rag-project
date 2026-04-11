import streamlit as st
import shutil, os
import tempfile
from ingestion import load_documents
from chunking import chunk_documents
from vector_store import get_collection, index_chunks, collection_is_empty, get_indexed_files
from retrieval import retrieve_context
from generation import generate_answer

st.set_page_config(page_title="RAG Decision Support", page_icon="🧠", layout="wide")
st.title("🧠 AI Decision Support System")
st.caption("Carica documenti, poi interrogali in linguaggio naturale.")

# --- SIDEBAR: caricamento documenti ---
with st.sidebar:
    st.header("📂 Carica Documenti")
    uploaded_files = st.file_uploader(
        "Carica PDF o TXT",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("📥 Indicizza Documenti", disabled=not uploaded_files):
        with st.spinner("Indicizzazione in corso..."):
        
        # Svuota il database prima di reindicizzare
            # if os.path.exists("./chroma_db"):
            #     shutil.rmtree("./chroma_db")
            
            collection = get_collection()
            already_indexed = get_indexed_files(collection)
            all_chunks = []
            new_files_count = 0

            for uploaded_file in uploaded_files:
                if uploaded_file.name in already_indexed:
                    st.warning(f"⚠️ {uploaded_file.name} già indicizzato, salto...")
                    continue
                
                new_files_count += 1
                suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".txt"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                docs = load_documents(tmp_path)
                for doc in docs:
                    doc.metadata["source"] = uploaded_file.name
                chunks = chunk_documents(docs)
                all_chunks.extend(chunks)
                os.unlink(tmp_path)

            if all_chunks:
                index_chunks(all_chunks, collection)
                st.success(f"✅ {len(all_chunks)} nuovi chunk aggiunti da {new_files_count} file.")
            else:
                st.info("ℹ️ Nessun nuovo file da indicizzare.")

    st.divider()

    # Impostazioni
    st.header("⚙️ Impostazioni")
    top_k = st.slider("Chunk da recuperare", min_value=1, max_value=10, value=5)
    
    st.divider()
    
    st.header("📊 Info Database")
    if st.button("📋 Vedi file indicizzati"):
        collection = get_collection()
        if not collection_is_empty(collection):
            files = get_indexed_files(collection)
            st.info(f"File nel database: {len(files)}\n\n" + "\n".join(files))
        else:
            st.warning("Nessun file indicizzato")

# --- AREA PRINCIPALE: chat ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Mostra storico
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input domanda
if prompt := st.chat_input("Fai una domanda sui tuoi documenti..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Ricerca nel contesto..."):
            collection = get_collection()

            if collection_is_empty(collection):
                answer_text = "⚠️ Nessun documento indicizzato. Carica dei file dalla barra laterale."
                st.warning(answer_text)
            else:
                context = retrieve_context(prompt, collection, top_k=top_k)
                result = generate_answer(prompt, context)
                answer_text = result["answer"]

                st.markdown(answer_text)

                # Mostra fonti in expander
                if result["sources"]:
                    with st.expander(f"📄 Fonti utilizzate ({len(result['sources'])})"):
                        for s in result["sources"]:
                            st.markdown(
                                f"**[Fonte {s['rank']}]** `{s['source']}` "
                                f"| Pag. {s['page']} "
                                f"| Similarità: `{s['similarity']}`"
                            )
                            st.caption(s["text"][:300] + "...")
                            st.divider()

                st.caption(f"🔢 Token usati: {result['tokens_used']}")

    st.session_state["messages"].append({"role": "assistant", "content": answer_text})