import os
import logging
from langchain_chroma import Chroma
from services.embeddings import get_embeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "db", "chroma_db")

# Create database directory if it doesn't exist
os.makedirs(DB_DIR, exist_ok=True)
logger.info(f"Using vector database directory: {DB_DIR}")

def get_vectorstore():
    """Get a Chroma vectorstore instance with the specified embeddings."""
    try:
        embeddings = get_embeddings()
        logger.info("Initializing Chroma vector database")
        db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        return db
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        raise RuntimeError(f"Failed to initialize vector database: {str(e)}")

def add_documents(docs):
    """Add new documents into existing Chroma DB."""
    try:
        if not docs:
            logger.warning("No documents to add to vector database")
            return None
            
        logger.info(f"Adding {len(docs)} documents to Chroma DB")
        db = get_vectorstore()
        db.add_documents(docs)
        logger.info("Documents successfully added to vector database")
        return db
    except Exception as e:
        logger.error(f"Error adding documents to vector database: {str(e)}")
        raise RuntimeError(f"Failed to add documents to vector database: {str(e)}")
