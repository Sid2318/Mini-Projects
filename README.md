# Mini-Projects Collection

This repository contains a collection of mini-projects developed for various purposes.

## Projects Included

### Chatbot

An interactive chatbot application built using:

- React for the frontend
- Express.js for the API backend
- FastAPI with LangGraph and Groq for the AI integration

#### Features

- Real-time chat interface
- Connection to Groq LLM API for intelligent responses
- Modern Bootstrap UI

#### Setup Instructions

1. Clone the repository
2. Set up the environment:

**Backend Setup**

```bash
cd Chatbot/backend
pip install -r requirements.txt
# Create a .env file with your Groq API key
echo "GROQ_API_KEY=your_api_key_here" > .env
npm install
```

**Frontend Setup**

```bash
cd Chatbot/frontend
npm install
```

3. Run the application:

**Start the Python FastAPI Backend**

```bash
cd Chatbot/backend
python app.py
```

**Start the Node.js Express Backend**

```bash
cd Chatbot/backend
npm start
```

**Start the React Frontend**

```bash
cd Chatbot/frontend
npm run dev
```

4. Access the application at http://localhost:5173

### AI for Job Matching

Details about the job matching project.

### ML for Job Recommendation

Details about the job recommendation project.

### YouTube Suggestion System

Details about the YouTube suggestion system.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
