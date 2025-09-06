from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.file_handler import save_and_process_files
from services.rag_pipeline import get_answer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to handle cross-origin requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods including OPTIONS
    allow_headers=["*"],
    expose_headers=["*"],
)

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    """
    Upload files for processing and adding to the RAG system.
    Supports PDF, TXT, and DOCX files.
    """
    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="No files provided"
            )
        
        # Log the files being uploaded
        logger.info(f"Received {len(files)} files for upload: {[f.filename for f in files]}")
        
        result = save_and_process_files(files)
        return result
    
    except HTTPException as e:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process files: {str(e)}"
        )

@app.get("/ask/")
async def ask_question(q: str):
    """
    Answer questions using the RAG system.
    This endpoint retrieves relevant context from the vector database
    and generates an answer using the LLM.
    """
    try:
        if not q:
            raise HTTPException(
                status_code=400,
                detail="No question provided"
            )
        
        logger.info(f"Received question: {q}")
        answer, context = get_answer(q)
        logger.info("Generated answer successfully")
        return {"answer": answer, "context": context}
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate answer: {str(e)}"
        )
