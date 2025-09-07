import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from transformers import pipeline

# --- Summarization Pipeline ---
print("⚡ Loading summarization model...")
summarizer = pipeline("summarization", model="Falconsai/text_summarization")

def summarize_text(text):
    """Summarize text using local Hugging Face model"""
    try:
        summary = summarizer(text, max_length=125, min_length=55, do_sample=False)
        return summary[0]["summary_text"]
    except Exception as e:
        print("❌ Summarization error:", e)
        return None

# --- Selenium Setup ---
options = webdriver.ChromeOptions()
options.add_argument("--headless=new")
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# --- Site-specific Scrapers ---
def scrape_google_news(driver, max_articles=5):
    driver.get("https://news.google.com/topstories?hl=en-IN&gl=IN&ceid=IN:en")
    time.sleep(3)
    digest = []
    for h in driver.find_elements(By.TAG_NAME, "h3")[:max_articles]:
        try:
            title = h.text
            a_tag = h.find_element(By.TAG_NAME, "a")
            link = a_tag.get_attribute("href") if a_tag else "No link found"
            digest.append({"title": title, "url": link})
        except:
            continue
    return digest

def scrape_bbc_news(driver, max_articles=5):
    driver.get("https://www.bbc.com/news")
    time.sleep(3)
    digest = []
    headlines = driver.find_elements(By.CSS_SELECTOR, "h3 a")
    for h in headlines[:max_articles]:
        try:
            title = h.text
            link = h.get_attribute("href")
            digest.append({"title": title, "url": link})
        except:
            continue
    return digest

def scrape_ndtv_news(driver, max_articles=5):
    driver.get("https://www.ndtv.com/latest")
    time.sleep(3)
    digest = []
    headlines = driver.find_elements(By.CSS_SELECTOR, "h2 a")
    for h in headlines[:max_articles]:
        try:
            title = h.text
            link = h.get_attribute("href")
            digest.append({"title": title, "url": link})
        except:
            continue
    return digest

def scrape_reuters_news(driver, max_articles=5):
    driver.get("https://www.reuters.com/world/")
    time.sleep(3)
    digest = []
    headlines = driver.find_elements(By.CSS_SELECTOR, "h3 a")
    for h in headlines[:max_articles]:
        try:
            title = h.text
            link = h.get_attribute("href")
            digest.append({"title": title, "url": link})
        except:
            continue
    return digest

def scrape_hindu_news(driver, max_articles=5):
    driver.get("https://www.thehindu.com/news/")
    time.sleep(3)
    digest = []
    headlines = driver.find_elements(By.CSS_SELECTOR, "h3 a, h2 a")
    for h in headlines[:max_articles]:
        try:
            title = h.text
            link = h.get_attribute("href")
            digest.append({"title": title, "url": link})
        except:
            continue
    return digest

# --- Scrape and summarize all sites ---
all_digests = []
for site_scraper in [scrape_google_news, scrape_bbc_news, scrape_ndtv_news, scrape_reuters_news, scrape_hindu_news]:
    site_digest = site_scraper(driver)
    for article in site_digest:
        article['summary'] = summarize_text(article['title'])
    all_digests += site_digest

driver.quit()

# --- Print final digest ---
print("\n📰 Multi-site News Digest\n" + "="*60)
for i, d in enumerate(all_digests, 1):
    print(f"\n{i}️⃣ {d['title']}\n👉 {d.get('summary', 'No summary')}\n🔗 {d['url']}")
    print("-"*60)