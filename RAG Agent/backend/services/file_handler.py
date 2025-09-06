import os
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from services.vectorstore import get_vectorstore, add_documents
from services.embeddings import get_embeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from fastapi import HTTPException

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "uploads"

def save_and_process_files(files):
    """Save multiple files, extract text, split, embed, and add to Chroma."""
    try:
        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Get vector database and embeddings model
        logger.info("Initializing vector database and embeddings model")
        db = get_vectorstore()
        embeddings = get_embeddings()
        
        # Create text splitter for chunking documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        processed_files = []

        for file in files:
            try:
                file_path = os.path.join(UPLOAD_DIR, file.filename)

                # Save file locally
                with open(file_path, "wb") as f:
                    f.write(file.file.read())

                # Load based on type
                if file.filename.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file.filename.endswith(".txt"):
                    loader = TextLoader(file_path)
                elif file.filename.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                else:
                    logger.warning(f"Unsupported file type: {file.filename}")
                    continue  # skip unsupported

                documents = loader.load()

                # Check if documents were extracted
                if not documents:
                    logger.warning(f"No content extracted from file: {file.filename}")
                    continue
                    
                logger.info(f"Extracted {len(documents)} documents from {file.filename}")
                    
                # Add metadata for traceability
                for doc in documents:
                    doc.metadata["source"] = file.filename

                # Split documents into chunks
                chunks = text_splitter.split_documents(documents)
                
                # Check if chunks were created
                if not chunks:
                    logger.warning(f"No chunks created from file: {file.filename}")
                    continue
                    
                logger.info(f"Created {len(chunks)} chunks from {file.filename}")
                
                # Add documents to vector database
                db.add_documents(chunks)
                logger.info(f"Successfully added {len(chunks)} chunks to vector database")
                
                processed_files.append(file.filename)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue

        if not processed_files:
            return {"status": "warning", "message": "No files were successfully processed", "files_processed": []}
            
        return {"status": "success", "files_processed": processed_files}
    
    except Exception as e:
        logger.error(f"Error in save_and_process_files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File processing failed: {str(e)}")
