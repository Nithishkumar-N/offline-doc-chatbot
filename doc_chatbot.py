import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os

st.title("📄 Document Q&A Chatbot")

api_key = st.text_input("Enter your OpenAI API key", type="password")
os.environ["OPENAI_API_KEY"] = api_key

uploaded_file = st.file_uploader("Upload PDF or DOCX file", type=["pdf", "docx"])
query = st.text_input("Ask your question:")

if uploaded_file and query and api_key:
    with st.spinner("Finding answer..."):
        if uploaded_file.name.endswith(".pdf"):
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.read())
            loader = PyPDFLoader("temp.pdf")
        else:
            with open("temp.docx", "wb") as f:
                f.write(uploaded_file.read())
            loader = Docx2txtLoader("temp.docx")

        documents = loader.load()
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(chunks, embeddings)

        results = db.similarity_search(query, k=2)
        model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        chain = load_qa_chain(model, chain_type="stuff")
        response = chain.run(input_documents=results, question=query)

    st.success("Answer:")
    st.write(response)

