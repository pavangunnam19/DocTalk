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
    # Placeholder for LLM integration (replace with your LLM setup)
    from langchain.chains import RetrievalQA
    from langchain.llms import OpenAI  # Example; use your preferred LLM
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)
    return qa_chain.run(query)
