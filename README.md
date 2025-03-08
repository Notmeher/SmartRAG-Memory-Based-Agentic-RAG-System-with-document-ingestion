# SmartRAG

An intelligent Retrieval-Augmented Generation (RAG) system powered by LangChain, ChromaDB, and GPT-4.

## Overview

SmartRAG is an RAG system designed to provide accurate and well-structured responses by retrieving relevant information from a knowledge base. The system analyzes query intent, searches a vector database for the most relevant documents, and retrieves multiple sources to ensure diverse perspectives.

![screencapture-localhost-8501-2025-03-08-13_26_31](https://github.com/user-attachments/assets/b6eaa122-cdae-4161-8d3b-bbfb3c778a7f)


## Features

- **Intelligent Document Retrieval**: Uses Maximum Marginal Relevance to ensure diverse and relevant information retrieval
- **Automatic Document Ingestion**: Process PDF documents and store them in a vector database
- **Deduplication**: Prevents re-embedding already processed documents using hash tracking
- **User-Friendly Interface**: Streamlit-based chat interface for easy interaction
- **Source Transparency**: View the exact documents and passages that inform each response
- **Persistent Database**: ChromaDB maintains your document embeddings between sessions

## Tech Stack

- LangChain for orchestrating the components
- OpenAI's text-embedding-3-small for document embeddings
- GPT-4 for response generation
- ChromaDB as the vector database
- Streamlit for the web interface

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Notmeher/SmartRAG-Memory-Based-RAG-System-with-document-ingestion.git
   
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Document Ingestion

Use the `db_ingest.py` script to add documents to the vector database:

```bash
# Ingest a specific PDF file
python db_ingest.py path/to/your/document.pdf

# Ingest all PDFs in a directory
python db_ingest.py path/to/your/documents/
```

### Running the Application

Launch the Streamlit application:

```bash
streamlit run main.py
```

The application will start and be accessible at http://localhost:8501 by default.

### Using the Interface

1. **Ingest Documents**: Use the sidebar to ingest the default document or upload your own PDFs
2. **Ask Questions**: Type your query in the chat input field
3. **View Responses**: Get comprehensive answers with citations to the source documents
4. **Explore Sources**: Expand the "Sources" section to see which documents informed the response

## How It Works

1. Documents are loaded, split into chunks, and embedded using OpenAI's text-embedding-3-small
2. Embeddings are stored in a ChromaDB vector database
3. When a user asks a question, the system:
   - Converts the question to an embedding
   - Searches the vector database for similar document chunks
   - Retrieves the most relevant chunks while ensuring diversity
   - Sends these chunks along with the question to GPT-4
   - Returns a comprehensive answer based on the retrieved information

## Project Structure

```
smartrag/
├── db_ingest.py          # Document ingestion script
├── main.py               # Streamlit application
├── chroma_db/            # Vector database storage
├── processed_document_hashes.txt  # Record of processed documents
├── .env                  # Environment variables
└── README.md             # This file
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [ChromaDB](https://github.com/chroma-core/chroma) for the vector database
- [OpenAI](https://openai.com/) for the embedding and LLM models
- [Streamlit](https://streamlit.io/) for the web interface
