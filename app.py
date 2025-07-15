import streamlit as st
from src.rag import load_and_process_pdf, get_response

st.title("DocTalk")
uploaded_file = st.file_uploader("Upload a PDF or HTML file", type=["pdf", "html"])
if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    with open(f"temp.{file_extension}", "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"{file_extension.upper()} file uploaded successfully!")
    vectorstore = load_and_process_pdf(f"temp.{file_extension}")
    st.session_state.vectorstore = vectorstore

user_input = st.text_input("Ask a question about the document:")
if user_input and "vectorstore" in st.session_state:
    response = get_response(st.session_state.vectorstore, user_input)
    st.write("Response:", response)
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from bs4 import BeautifulSoup

def parse_html_to_paragraphs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return paragraphs

def load_and_process_pdf(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif file_path.endswith(".html"):
        documents = parse_html_to_paragraphs(file_path)
        documents = [{"page_content": p} for p in documents]  # Format for LangChain
    else:
        raise ValueError("Unsupported file type")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_response(vectorstore, query):
    # Placeholder for LLM integration (e.g., using LangChain's LLM)
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI  # Replace with your LLM setup
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    return qa_chain.run(query)
import streamlit as st
from src.rag import load_and_process_pdf, get_response

st.set_page_config(layout="wide")
st.title("DocTalk")
st.markdown("Chat with your PDFs or HTML files! Upload a file and ask questions.")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("", type=["pdf", "html"], key="file_uploader")
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        with open(f"temp.{file_extension}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"{file_extension.upper()} file uploaded successfully!")
        vectorstore = load_and_process_pdf(f"temp.{file_extension}")
        st.session_state.vectorstore = vectorstore

# Main chat area
st.header("Chat Interface")
user_input = st.text_input("Your Question:", key="user_input")
if user_input and "vectorstore" in st.session_state:
    response = get_response(st.session_state.vectorstore, user_input)
    st.write("**Response:**", response)
    st.session_state.chat_history = st.session_state.get("chat_history", []) + [(user_input, response)]

# Display chat history
if "chat_history" in st.session_state:
    st.subheader("Chat History")
    for question, answer in st.session_state.chat_history[-5:]:  # Show last 5 interactions
        st.write(f"**Q:** {question}")
        st.write(f"**A:** {answer}")
        st.write("---")
