from flask import Flask, request, jsonify, render_template
# from flask_sqlalchemy import SQLAlchemy
from flask_pymongo import PyMongo
from job_matching import extract_skills, match_teacher

app = Flask(__name__)

# Configure MongoDB
app.config["MONGO_URI"] = "mongodb+srv://OweeMirajkar:r.3y_rP_Ne6xAmX@cluster0.abxbzkz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
mongo = PyMongo(app)

# class Teacher(db.Model):
#     __tablename__ = "teachers"
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     resume_text = db.Column(db.Text, nullable=False)

# class Job(db.Model):
#     __tablename__ = "jobs"
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(100), nullable=False)
#     skills_required = db.Column(db.Text, nullable=False)  # Store as comma-separated values
#     salary = db.Column(db.String(50), nullable=True)
#     location = db.Column(db.String(100), nullable=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/match_teacher', methods=['POST'])
def match_teacher_jobs():
    data = request.json
    teacher_id = data.get('teacher_id')

    if not teacher_id:
        return jsonify({"error": "Teacher ID is required"}), 400

    # Fetch teacher details
    teacher = mongo.db.teachers.find_one({"_id": teacher_id})
    if not teacher:
        return jsonify({"error": "Teacher not found"}), 404

    # Extract skills from bio & certifications
    extracted_skills = extract_skills(teacher.get("bio", "") + " " + " ".join(cert["certification_name"] for cert in mongo.db.certifications.find({"teacher_id": teacher_id})))

    # Fetch job listings
    job_listings = list(mongo.db.jobs.find({}))
    
    formatted_jobs = []
    for job in job_listings:
        formatted_jobs.append({
            "id": job["_id"],
            "title": job["job title"],
            "skills_required": job["required skills"].split(","),
            "location": job["location"],
            "company": job["company name"],
            "description": job["job description"]
        })

    # Match teacher with jobs
    matched_jobs = match_teacher(extracted_skills, formatted_jobs)

    return jsonify({"matched_jobs": matched_jobs})

if __name__ == "__main__":
    app.run(debug=True, port=8000)