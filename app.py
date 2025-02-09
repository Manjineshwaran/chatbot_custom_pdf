import streamlit as st
import os
import shutil
import warnings
from typing import List
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')

def load_pdf(file_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return documents
    except FileNotFoundError:
        st.error(f"Error: File {file_path} not found.")
        return []
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 100) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document], embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        vector_store_path = "chroma_vector_store"  
        if os.path.exists(vector_store_path):
            shutil.rmtree(vector_store_path)  

        vector_store = Chroma.from_documents(documents, embeddings, persist_directory=vector_store_path)
        
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def create_qa_chain(vector_store):
    try:
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_retries=2
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}  
        )
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

class PDFProcessor:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self.processed_files = []
    
    def process_pdfs(self, pdf_files):
        new_files = [f for f in pdf_files if f.name not in self.processed_files]
        
        if not new_files:
            st.info("No new PDFs to process.")
            return "No new PDFs to process."
        
        all_documents = []
        for pdf_file in new_files:
            with open(pdf_file.name, "wb") as f:
                f.write(pdf_file.read())
            documents = load_pdf(pdf_file.name)
            if not documents:
                st.error(f"Failed to load PDF: {pdf_file.name}")
                return f"Failed to load PDF: {pdf_file.name}"
            all_documents.extend(documents)

        st.info(f"Splitting {len(all_documents)} documents...")
        split_docs = split_documents(all_documents)
        
        if self.vector_store is None:
            self.vector_store = create_vector_store(split_docs)
        else:
            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            self.vector_store.add_documents(split_docs)
        
        if not self.vector_store:
            st.error("Failed to create vector store")
            return "Failed to create vector store"
        
        st.info("Creating QA chain...")
        self.qa_chain = create_qa_chain(self.vector_store)
        if not self.qa_chain:
            st.error("Failed to create QA chain")
            return "Failed to create QA chain"
        
        self.processed_files.extend([f.name for f in new_files])
        
        st.success(f"Successfully processed {len(new_files)} PDF(s).")
        return f"Successfully processed {len(new_files)} PDF(s). Total processed files: {len(self.processed_files)}"
    
    def query_pdfs(self, query):
        if not self.qa_chain:
            return "Please upload and process PDFs first", []
        
        try:
            response = self.qa_chain.invoke({"query": query})
            if not response:
                return "No response from the QA chain.", []
            
            return response['result'], []
        
        except Exception as e:
            st.error(f"Error processing query: {e}")
            return f"Error processing query: {e}", []
def main():
    st.title("Eshwaran's AI Assistant")
    st.markdown("### Upload PDFs and ask questions about their content.")

    if "pdf_processor" not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    pdf_processor = st.session_state.pdf_processor
    pdf_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process PDFs"):
        if pdf_files:
            status = pdf_processor.process_pdfs(pdf_files)
            st.success(status)
        else:
            st.warning("Please upload PDF files first.")
    
    query = st.text_input("Ask a Question", "")
    if st.button("Submit Query"):
        if query.strip():
            answer, sources = pdf_processor.query_pdfs(query)
            if answer:
                st.text_area("Answer", value=answer, height=100, disabled=True)
            else:
                st.warning("No answer returned. Ensure PDFs are processed correctly.")
            
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
