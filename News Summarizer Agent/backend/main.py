from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import requests
import os
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")
if not API_KEY:
    raise ValueError("❌ NEWS_API_KEY not found in .env file")

# Initialize FastAPI app
app = FastAPI(title="News Summarizer Agent", version="1.0")

# Summarizer pipeline
print("⚡ Loading summarizer model...")
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

# Response schema
class NewsArticle(BaseModel):
    title: str
    summary: str
    url: str

@app.get("/news", response_model=List[NewsArticle])
def get_news(date: str = Query(..., description="Date in YYYY-MM-DD format")):
    """
    Fetch and summarize top news for a given date.
    Example: /news?date=2025-09-01
    """
    # Call NewsAPI with date
    url = f"https://newsapi.org/v2/everything?q=top&from={date}&to={date}&sortBy=popularity&apiKey={API_KEY}"
    response = requests.get(url)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch news")

    articles = response.json().get("articles", [])
    if not articles:
        return []

    digest = []
    for a in articles[:10]:  # summarize top 5
        content = a.get("content") or a.get("description") or a.get("title")
        if content:
            try:
                summary = summarizer(content, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
            except Exception:
                summary = content  # fallback if summarizer fails
            digest.append({
                "title": a["title"],
                "summary": summary,
                "url": a["url"]
            })

    return digest
