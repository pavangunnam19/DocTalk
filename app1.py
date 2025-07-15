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
        with open(f"data/temp.{file_extension}", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"{file_extension.upper()} file uploaded successfully!")
        vectorstore = load_and_process_pdf(f"data/temp.{file_extension}")
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
