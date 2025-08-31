from flask import Flask, render_template, request, redirect, url_for, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'secretkey'  # Secret key for session management

# In-memory storage for users' profiles, credentials, and job listings
users_data = {
    "priya.k@email.com": {
        "name": "Priya K.",
        "email": "priya.k@email.com",
        "password": "password123",
        "phoneNumber": "+911234567890",
        "region": "Maharashtra, India",
        "languagePreference": "Hindi, English",
        "bio": "Passionate about typing, tailoring and baking. Looking to expand my skills and connect with like-minded professionals.",
        "profilePic": "https://randomuser.me/api/portraits/women/45.jpg",
        "dateJoined": "2023-12-01",
        "skills": ["fast typing","tailoring","baking"]
    },
    "rahul.s@email.com": {
        "name": "Rahul S.",
        "email": "rahul.s@email.com",
        "password": "password123",
        "phoneNumber": "+919876543210",
        "region": "Gujarat, India",
        "languagePreference": "Gujarati, English",
        "bio": "Skilled in data entry and office work. Always looking for new opportunities to grow.",
        "profilePic": "https://randomuser.me/api/portraits/men/32.jpg",
        "dateJoined": "2024-01-15",
        "skills": []
    },
    "aisha.m@email.com": {
        "name": "Aisha M.",
        "email": "aisha.m@email.com",
        "password": "securePass",
        "phoneNumber": "+919812345678",
        "region": "Karnataka, India",
        "languagePreference": "Kannada, English",
        "bio": "Experienced in graphic design and social media marketing.",
        "profilePic": "https://randomuser.me/api/portraits/women/36.jpg",
        "dateJoined": "2024-02-10",
        "skills": []
    },
    "vikram.t@email.com": {
        "name": "Vikram T.",
        "email": "vikram.t@email.com",
        "password": "vikram2025",
        "phoneNumber": "+918765432109",
        "region": "Delhi, India",
        "languagePreference": "Hindi, English",
        "bio": "Software engineer with expertise in Python and backend development.",
        "profilePic": "https://randomuser.me/api/portraits/men/24.jpg",
        "dateJoined": "2024-03-05",
        "skills": []
    },
    "neha.p@email.com": {
        "name": "Neha P.",
        "email": "neha.p@email.com",
        "password": "neha2024",
        "phoneNumber": "+919623456789",
        "region": "Maharashtra, India",
        "languagePreference": "Marathi, English",
        "bio": "Certified yoga instructor and wellness coach.",
        "profilePic": "https://randomuser.me/api/portraits/women/29.jpg",
        "dateJoined": "2024-04-12",
        "skills": []
    },
    "arjun.v@email.com": {
        "name": "Arjun V.",
        "email": "arjun.v@email.com",
        "password": "arjun2024",
        "phoneNumber": "+918912345678",
        "region": "Tamil Nadu, India",
        "languagePreference": "Tamil, English",
        "bio": "Electrician with 5 years of experience in wiring and installation.",
        "profilePic": "https://randomuser.me/api/portraits/men/40.jpg",
        "dateJoined": "2024-05-20",
        "skills": []
    },
    "sonali.k@email.com": {
        "name": "Sonali K.",
        "email": "sonali.k@email.com",
        "password": "sonali789",
        "phoneNumber": "+919734567890",
        "region": "West Bengal, India",
        "languagePreference": "Bengali, English",
        "bio": "Passionate about teaching and child development.",
        "profilePic": "https://randomuser.me/api/portraits/women/50.jpg",
        "dateJoined": "2024-06-18",
        "skills": []
    }
}


jobs_data = [
    {
        "companyName": "Tailoring Emporium",
        "jobTitle": "Tailoring Assistant",
        "jobDescription": "Assist in tailoring garments, maintaining the workshop, and assisting customers in choosing fabric.",
        "requiredSkills": ["Tailoring", "Customer Service", "Fabric Knowledge"],
        "location": "Mumbai, Maharashtra",
        "postedDate": "2025-03-25",
        "expiryDate": "2025-04-25"
    },
    {
        "companyName": "Baking Delights",
        "jobTitle": "Baker",
        "jobDescription": "Prepare baked goods, decorate cakes, and assist in creating recipes for new products.",
        "requiredSkills": ["Baking", "Cake Decoration", "Customer Interaction"],
        "location": "Pune, Maharashtra",
        "postedDate": "2025-03-28",
        "expiryDate": "2025-04-28"
    },
    {
        "companyName": "Tech Innovations",
        "jobTitle": "Python Developer",
        "jobDescription": "Develop scalable web applications using Python and Flask.",
        "requiredSkills": ["Python", "Flask", "Backend Development"],
        "location": "Bangalore, Karnataka",
        "postedDate": "2025-03-30",
        "expiryDate": "2025-05-01"
    },
    {
        "companyName": "FitWell Studio",
        "jobTitle": "Yoga Instructor",
        "jobDescription": "Conduct yoga sessions for beginners and advanced learners.",
        "requiredSkills": ["Yoga", "Wellness Coaching", "Health Management"],
        "location": "Nagpur, Maharashtra",
        "postedDate": "2025-04-01",
        "expiryDate": "2025-05-02"
    },
    {
        "companyName": "Power Electric",
        "jobTitle": "Electrician",
        "jobDescription": "Install and maintain electrical systems in residential and commercial buildings.",
        "requiredSkills": ["Electrical Wiring", "Installation", "Maintenance"],
        "location": "Chennai, Tamil Nadu",
        "postedDate": "2025-04-05",
        "expiryDate": "2025-05-10"
    },
    {
        "companyName": "Bright Minds Academy",
        "jobTitle": "Primary School Teacher",
        "jobDescription": "Teach students foundational subjects with engaging methods.",
        "requiredSkills": ["Teaching", "Child Development", "Classroom Management"],
        "location": "Kolkata, West Bengal",
        "postedDate": "2025-04-08",
        "expiryDate": "2025-05-15"
    },
    {
        "companyName": "Digital Creatives",
        "jobTitle": "Graphic Designer",
        "jobDescription": "Design marketing materials, social media content, and brand assets.",
        "requiredSkills": ["Graphic Design", "Adobe Photoshop", "Illustration"],
        "location": "Delhi, India",
        "postedDate": "2025-04-12",
        "expiryDate": "2025-05-20"
    },
    {
        "companyName": "Data Entry Solutions",
        "jobTitle": "Data Entry Operator",
        "jobDescription": "Enter and manage data accurately in the company database.",
        "requiredSkills": ["Data Entry", "Typing", "Microsoft Excel"],
        "location": "Ahmedabad, Gujarat",
        "postedDate": "2025-04-15",
        "expiryDate": "2025-05-25"
    }
]


# Function to extract skills from bio (a simple approach can be checking for keywords)
def extract_skills_from_bio(bio):
    # This is a simple placeholder method; you'd ideally use NLP to identify skills more effectively
    possible_skills = ["tailoring", "baking", "data entry", "customer service", "fabric knowledge", "cake decoration"]
    extracted_skills = [skill for skill in possible_skills if skill.lower() in bio.lower()]
    return extracted_skills

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if email exists and password is correct
        user = users_data.get(email)
        if user and user['password'] == password:
            session['user'] = email  # Save email in session
            return redirect(url_for('index'))  # Redirect to home page
        else:
            return "Invalid credentials", 401  # Return error message for incorrect login

    return render_template('login.html')

# Route to display the home page after login
@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    
    user_email = session['user']
    user = users_data.get(user_email)

    # Extract skills from bio and add them to the user's explicit skills
    bio_skills = extract_skills_from_bio(user['bio'])
    all_skills = list(set(user['skills'] + bio_skills))  # Combine explicit skills and bio skills, avoiding duplicates

    return render_template('index.html', user_email=user_email, skills=all_skills)

# Route to update the user's profile (add skills)
@app.route('/update_profile', methods=['GET', 'POST'])
def update_profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    user = users_data[user_email]

    if request.method == 'POST':
        new_skills = request.form.getlist('skills')  # Get multiple skills as a list
        user['skills'].extend(new_skills)  # Append new skills

        # Extract skills from bio and merge with explicit skills
        bio_skills = extract_skills_from_bio(user['bio'])
        user['skills'] = list(set(user['skills'] + bio_skills))  # Remove duplicates

        return redirect(url_for('index'))  # Redirect to home page after updating profile

    # Extract current skills (explicit + bio)
    bio_skills = extract_skills_from_bio(user['bio'])
    all_skills = list(set(user['skills'] + bio_skills))

    return render_template('update_profile.html', skills=all_skills)


# Route to get job recommendations
@app.route('/recommend', methods=['GET'])
def recommend():
    if 'user' not in session:
        return redirect(url_for('login'))

    user_email = session['user']
    user = users_data.get(user_email)

    if not user:
        return f"User {user_email} not found!", 404

    recommended_jobs = recommend_jobs(user['skills'])
    return render_template('recommend.html', user_email=user_email, recommended_jobs=recommended_jobs)

# Function to recommend jobs based on skills
def recommend_jobs(user_skills):
    job_descriptions = [job['jobDescription'] for job in jobs_data]

    # TF-IDF Vectorizer to convert job descriptions to vectors
    vectorizer = TfidfVectorizer()
    job_vectors = vectorizer.fit_transform(job_descriptions)

    # Create a vector for the user's skills
    user_skill_text = " ".join(user_skills)
    user_vector = vectorizer.transform([user_skill_text])

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(user_vector, job_vectors)

    # Rank jobs by similarity
    ranked_jobs = sorted(zip(similarity_scores[0], jobs_data), reverse=True, key=lambda x: x[0])

    # Get the top 3 jobs
    top_jobs = [{"companyName": job[1]['companyName'],
                 "jobTitle": job[1]['jobTitle'],
                 "jobDescription": job[1]['jobDescription'],
                 "location": job[1]['location'],
                 "postedDate": job[1]['postedDate']}
                for job in ranked_jobs[:3]]

    return top_jobs

if __name__ == '__main__':
    app.run(debug=True)
