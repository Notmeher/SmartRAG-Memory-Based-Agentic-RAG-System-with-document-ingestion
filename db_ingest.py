import os
import sys
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
import hashlib

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables.")
    sys.exit(1)

# Define paths
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")
DOCUMENT_HASHES_PATH = os.path.join(os.getcwd(), "processed_document_hashes.txt")

def get_document_hash(file_path):
    """Generate a hash for a document to avoid re-embedding."""
    with open(file_path, 'rb') as f:
        content = f.read()
        return hashlib.md5(content).hexdigest()

def load_processed_hashes():
    """Load the set of processed document hashes."""
    if not os.path.exists(DOCUMENT_HASHES_PATH):
        return set()
    
    with open(DOCUMENT_HASHES_PATH, 'r') as f:
        return {line.strip() for line in f.readlines()}

def save_processed_hash(document_hash):
    """Save a document hash to the processed hashes file."""
    with open(DOCUMENT_HASHES_PATH, 'a') as f:
        f.write(f"{document_hash}\n")

def ingest_document(file_path):
    """Ingest a document into the vector database."""
    print(f"Processing document: {file_path}")
    
    # Generate document hash
    document_hash = get_document_hash(file_path)
    
    # Check if document has already been processed
    processed_hashes = load_processed_hashes()
    if document_hash in processed_hashes:
        print(f"Document {file_path} has already been processed. Skipping.")
        return
    
    # Load document
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    else:
        print(f"Unsupported file format: {file_path}")
        return
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add source information to metadata
    for chunk in chunks:
        chunk.metadata['source'] = file_path
        chunk.metadata['document_hash'] = document_hash
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Check if Chroma DB already exists
    if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
        # Load existing DB and add new documents
        db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        db.add_documents(chunks)
    else:
        # Create a new DB
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR
        )
    
    # Persist the database to disk
    db.persist()
    
    # Save the document hash
    save_processed_hash(document_hash)
    
    print(f"Document {file_path} has been successfully ingested.")
    return len(chunks)

def ingest_from_directory(directory_path):
    """Ingest all supported documents from a directory."""
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return
    
    total_chunks = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.pdf'):
                file_path = os.path.join(root, file)
                chunks = ingest_document(file_path)
                if chunks:
                    total_chunks += chunks
    
    print(f"Ingestion complete. Total chunks added: {total_chunks}")

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            ingest_document(path)
        elif os.path.isdir(path):
            ingest_from_directory(path)
    else:
        # Default path from the requirements
        default_path = r"D:\agentic_rag\documents\2412.15605v2.pdf"
        if os.path.exists(default_path):
            ingest_document(default_path)
        else:
            print("Please provide a valid file or directory path.")
            print("Usage: python db_ingest.py [file_path or directory_path]")