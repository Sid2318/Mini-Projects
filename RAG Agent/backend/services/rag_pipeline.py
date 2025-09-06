from services.vectorstore import get_vectorstore
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))  # Load Groq API key


def get_answer(query: str):
    """Retrieve context from Chroma and ask ChatGroq LLM like a teacher."""

    db = get_vectorstore()
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)

    # Build context with metadata (file info)
    context_parts = []
    for doc in relevant_docs:
        source = doc.metadata.get("source", "Unknown file")
        page = doc.metadata.get("page", "N/A")
        context_parts.append(f"[File: {source}, Page: {page}]\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a knowledgeable teacher. "
                "Answer the student's question clearly using ONLY the provided context. "
                "Also, explain from which file and section (if available) the information was taken."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{context}\n\nAnswer:",
        },
    ]

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
    )

    return response.choices[0].message.content, context
