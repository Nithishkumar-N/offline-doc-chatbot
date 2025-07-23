import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Offline Document Q&A Chatbot")
st.title("🧠 Offline Document Q&A Chatbot")

uploaded_file = st.file_uploader("Upload PDF or DOCX file", type=["pdf", "docx"])

query = st.text_input("Ask your question")

if uploaded_file and query:
    with st.spinner("Processing..."):
        # Save uploaded file to temp path
        if uploaded_file.name.endswith(".pdf"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            loader = PyPDFLoader(tmp_path)

        elif uploaded_file.name.endswith(".docx"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            loader = Docx2txtLoader(tmp_path)
        else:
            st.error("Unsupported file type")
            st.stop()

        # Load documents and create embeddings
        documents = loader.load()
        embeddings = OllamaEmbeddings(model="mistral")
        db = FAISS.from_documents(documents, embeddings)

        # Set up local LLM with Ollama
        retriever = db.as_retriever()
        llm = ChatOllama(model="mistral")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Ask and show result
        result = qa_chain.run(query)
        st.success(result)
