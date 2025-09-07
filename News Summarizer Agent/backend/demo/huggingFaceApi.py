import os
import requests
from dotenv import load_dotenv

# Optional: local fallback
from transformers import pipeline

# Load API keys
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not NEWS_API_KEY:
    raise ValueError("❌ NEWS_API_KEY not found in .env")
if not HF_API_KEY:
    raise ValueError("❌ HF_API_KEY not found in .env")

# Hugging Face API setup
HF_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}

# Local fallback summarizer
local_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    try:
        response = requests.post(
            HF_API_URL,
            headers=HEADERS,
            json={"inputs": text, "parameters": {"max_length": 60, "min_length": 20}},
            timeout=60  # avoid hanging forever
        )

        if response.status_code == 200:
            return response.json()[0]["summary_text"]
        else:
            print("⚠️ HuggingFace API error:", response.status_code, response.text)
    except Exception as e:
        print("⚠️ HuggingFace request failed:", e)

    # Fallback: local summarization
    print("🔄 Falling back to local summarizer...")
    summary = local_summarizer(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]["summary_text"]

# Fetch top headlines from NewsAPI
url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWS_API_KEY}"
res = requests.get(url).json()
articles = res.get("articles", [])

# Process and summarize top 5 articles
digest = []
for a in articles[:5]:
    content = a.get("content") or a.get("description")
    if content:
        summary = summarize_text(content)
        if summary:
            digest.append({
                "title": a["title"],
                "summary": summary,
                "url": a["url"]
            })

# Print digest
print("\n📰 Daily News Digest\n" + "="*40)
for i, d in enumerate(digest, 1):
    print(f"\n{i}️⃣ {d['title']}\n👉 {d['summary']}\n🔗 {d['url']}")
