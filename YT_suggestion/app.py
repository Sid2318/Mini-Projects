from flask import Flask, render_template, request
import requests

app = Flask(__name__)

# Your YouTube API Key
API_KEY_YT = "AIzaSyBBvAoqs0h5ZmtPSbUmzXEIG_GF1yzclLU"

def fetch_youtube_videos(query):
    url = f"https://www.googleapis.com/youtube/v3/search?key={API_KEY_YT}&q={query}&part=snippet&type=video"
    response = requests.get(url)
    data = response.json()
    
    videos = []
    for item in data.get("items", []):
        video_info = {
            "title": item["snippet"]["title"],
            "video_url": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            "thumbnail_url": item["snippet"]["thumbnails"]["high"]["url"]
        }
        videos.append(video_info)
    
    return videos

@app.route("/", methods=["GET", "POST"])
def index():
    videos = []
    if request.method == "POST":
        query = request.form["query"]
        videos = fetch_youtube_videos(query)
    
    return render_template("index.html", videos=videos)

if __name__ == "__main__":
    app.run(debug=True)
