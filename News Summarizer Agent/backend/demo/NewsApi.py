from transformers import pipeline
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")  # Correct usage with quotes
if not API_KEY:
    raise ValueError("❌ NEWS_API_KEY not found in .env file")

# Fetch news
url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}"
response = requests.get(url)
articles = response.json().get("articles", [])

# Summarizer pipeline
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

digest = []
for a in articles[:5]:   # summarize top 5
    content = a.get("content") or a.get("description")
    if content:
        summary = summarizer(content, max_length=60, min_length=20, do_sample=False)
        digest.append({
            "title": a["title"],
            "summary": summary[0]["summary_text"],
            "url": a["url"]
        })

# Print digest
for i, d in enumerate(digest, 1):
    print(f"\n{i}️⃣ {d['title']}\n👉 {d['summary']}\n🔗 {d['url']}")
