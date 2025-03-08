import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import db_ingest

# Load environment variables
load_dotenv()

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    st.error("Error: OPENAI_API_KEY not found in environment variables.")
    st.stop()

# Define paths
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")

# RAG Agent prompt template
RAG_PROMPT = """
You are an intelligent Retrieval-Augmented Generation (RAG) agent designed to provide accurate and well-structured responses by retrieving relevant information from a knowledge base. Given a user query, carefully analyze its intent, search the vector database for the most relevant documents, and retrieve multiple sources to ensure diverse perspectives. Rank and filter results based on relevance and quality. Summarize key points from the retrieved documents while maintaining factual accuracy. If logical reasoning is required, apply step-by-step deduction. Ensure context consistency across multi-turn conversations. Clearly articulate the response in a structured, coherent, and reader-friendly manner, using bullet points, tables, or examples where useful.

Context information from the knowledge base:
{context}

User Query: {question}

Please provide a comprehensive response:
"""

def initialize_rag():
    """Initialize the RAG system with the vector database."""
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        st.warning("Vector database not found. Please ingest documents first.")
        return None
    
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Load the Chroma database
    db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # Create a retriever
    retriever = db.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance for diversity
        search_kwargs={"k": 5, "fetch_k": 10}  # Fetch 10 and return top 5 most diverse
    )
    
    # Initialize the LLM
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    
    # Create the prompt from the template
    prompt = PromptTemplate(
        template=RAG_PROMPT,
        input_variables=["context", "question"]
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    
    return qa_chain

def main():
    st.title("Agentic RAG System")
    
    # Add a sidebar for document ingestion
    with st.sidebar:
        st.header("Document Ingestion")
        
        # Option to ingest the default document
        if st.button("Ingest Default Document"):
            default_path = r"D:\agentic_rag\documents\2412.15605v2.pdf"
            with st.spinner("Ingesting default document..."):
                if os.path.exists(default_path):
                    db_ingest.ingest_document(default_path)
                    st.success(f"Document {default_path} ingested successfully!")
                else:
                    st.error(f"Default document path not found: {default_path}")
        
        # Option to upload and ingest a custom document
        uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
        if uploaded_file is not None:
            # Save the uploaded file temporarily
            temp_path = os.path.join(os.getcwd(), "temp_upload.pdf")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Ingest the document
            with st.spinner("Ingesting uploaded document..."):
                db_ingest.ingest_document(temp_path)
                st.success("Document ingested successfully!")
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Main chat interface
    st.header("Ask Questions")
    
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    user_query = st.chat_input("Ask a question about the documents...")
    
    if user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Initialize RAG system
        qa_chain = initialize_rag()
        
        if qa_chain:
            # Display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Get the response from the RAG chain
                    response = qa_chain({"query": user_query})
                    answer = response["result"]
                    source_docs = response.get("source_documents", [])
                    
                    # Display the answer
                    st.markdown(answer)
                    
                    # Display the sources if available
                    if source_docs:
                        with st.expander("Sources"):
                            for i, doc in enumerate(source_docs):
                                st.markdown(f"**Source {i+1}**: {doc.metadata.get('source', 'Unknown')}")
                                st.markdown(f"**Content**: {doc.page_content[:200]}...")
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # Display error message if RAG system is not initialized
            with st.chat_message("assistant"):
                st.markdown("I can't answer your question because no documents have been ingested yet. Please use the sidebar to ingest documents first.")
            
            # Add error message to chat history
            st.session_state.messages.append({"role": "assistant", "content": "I can't answer your question because no documents have been ingested yet. Please use the sidebar to ingest documents first."})

if __name__ == "__main__":
    main()