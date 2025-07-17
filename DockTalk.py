import streamlit as st
import time
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import re

st.markdown(
    """
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .animated-logo {
        animation: fadeIn 2s;
        text-align: center;
        font-size: 50px;
        font-weight: bold;
        color: #4CAF50;
    }
    .subtext {
        text-align: center;
        font-size: 18px;
        color: #666;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        cursor: pointer;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="animated-logo">DocTalk</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Developed by Pavan Gunnam</div>', unsafe_allow_html=True)
time.sleep(1)

st.markdown("### Welcome to DocTalk!")
st.markdown("Upload a PDF document and ask questions about its content.")

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "nn_model" not in st.session_state:
    st.session_state.nn_model = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embedding_model = load_embedding_model()

def extract_and_chunk_pdf(file_path):
    reader = PdfReader(file_path)
    text_chunks = []
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        if text:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            for sentence in sentences:
                if sentence.strip():
                    text_chunks.append(sentence.strip())
    return text_chunks

def generate_embeddings(text_chunks, model):
    embeddings = model.encode(text_chunks, convert_to_tensor=False)
    return embeddings

st.markdown("### Upload Your PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")

if uploaded_file is not None and not st.session_state.pdf_processed:
    os.makedirs("data", exist_ok=True)
    pdf_path = os.path.join("data", "temp.pdf")

    with st.spinner("Processing PDF... This may take a moment."):
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.text_chunks = extract_and_chunk_pdf(pdf_path)
        
        if st.session_state.text_chunks:
            st.session_state.embeddings = generate_embeddings(st.session_state.text_chunks, embedding_model)
            st.session_state.nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
            st.session_state.nn_model.fit(st.session_state.embeddings)
            st.session_state.pdf_processed = True
            st.success("PDF uploaded and processed successfully! You can now ask questions.")
        else:
            st.error("Could not extract text from the PDF. Please try a different file.")
            st.session_state.pdf_processed = False
            uploaded_file = None

elif uploaded_file is None and st.session_state.pdf_processed:
    st.session_state.pdf_processed = False
    st.session_state.text_chunks = []
    st.session_state.embeddings = None
    st.session_state.nn_model = None
    st.session_state.chat_history = []
    st.rerun()

if st.session_state.pdf_processed:
    st.markdown("### Ask a Question")
    user_input = st.text_input("Your Question:", key="user_question_input")

    if user_input:
        if st.session_state.nn_model is None or st.session_state.embeddings is None:
            st.error("PDF data not fully loaded. Please re-upload the PDF.")
        else:
            with st.spinner("Finding the answer..."):
                question_embedding = embedding_model.encode([user_input], convert_to_tensor=False)
                distances, indices = st.session_state.nn_model.kneighbors(question_embedding)
                relevant_chunks = [st.session_state.text_chunks[i] for i in indices[0]]
                
                google_api_key = "YOUR_ACTUAL_GOOGLE_API_KEY_HERE"  # Replace with your Google API key

                if not google_api_key:
                    st.error("Please provide a valid Gemini API key in the code.")
                    st.stop()

                llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
                
                prompt = f"""
                You are a helpful assistant that answers questions based on the provided text.
                If the answer is not available in the provided text, state that you cannot find it.

                PDF Content:
                {''.join(relevant_chunks)}

                Question: {user_input}
                Answer:
                """
                
                try:
                    response = llm.invoke(prompt)
                    answer_text = response.content if hasattr(response, 'content') else str(response)
                    st.write("**Answer:**", answer_text)
                    st.session_state.chat_history.append({"question": user_input, "answer": answer_text})
                except Exception as e:
                    st.error(f"An error occurred while generating the answer: {e}")

if st.session_state.chat_history:
    st.markdown("### Chat History")
    for chat in st.session_state.chat_history[-5:]:
        st.write(f"**Q:** {chat['question']}")
        st.write(f"**A:** {chat['answer']}")
        st.write("---")
