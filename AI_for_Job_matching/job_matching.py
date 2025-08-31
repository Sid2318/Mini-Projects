import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure Gemini API
genai.configure(api_key="AIzaSyD3IHgDTHiHU3QffVML_P2qBXkQN8Zd-mY")

def extract_skills(text):
    """Extracts skills from teacher's bio and certifications."""
    if not text:
        return []

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(f"Extract key teaching skills from this text:\n{text}")

    return response.text.split(", ")  # Convert to list

def match_teacher(teacher_skills, job_listings):
    """Matches teacher skills with job requirements using cosine similarity."""
    if not teacher_skills or not job_listings:
        return []

    all_texts = [", ".join(teacher_skills)] + [", ".join(job["skills_required"]) for job in job_listings]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Sort jobs by similarity score
    sorted_jobs = sorted(zip(job_listings, scores), key=lambda x: x[1], reverse=True)

    return [{"job": job, "score": round(score, 2)} for job, score in sorted_jobs[:5]]
