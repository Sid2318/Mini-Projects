from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to cache the embeddings model
_embeddings_instance = None

def get_embeddings():
    """
    Get or create a HuggingFaceEmbeddings instance.
    Uses a singleton pattern to avoid reloading the model multiple times.
    """
    global _embeddings_instance
    
    # If we already have an instance, return it
    if _embeddings_instance is not None:
        return _embeddings_instance
    
    try:
        # Set environment variable to use CPU for HuggingFace models
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        # Create a directory to cache the model
        os.makedirs("./model_cache", exist_ok=True)
        
        logger.info("Loading HuggingFace embedding model...")
        _embeddings_instance = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./model_cache"  # Cache the model locally
        )
        
        logger.info("HuggingFace embedding model loaded successfully")
        return _embeddings_instance
    
    except Exception as e:
        logger.error(f"Error initializing embeddings model: {str(e)}")
        raise RuntimeError(f"Failed to initialize embeddings model: {str(e)}")
