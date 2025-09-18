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

### Movie Recommendation System

A modern movie recommendation application built using:

- Next.js for the frontend and API routes
- Tailwind CSS for styling
- TypeScript for type safety

#### Features

- Browse movie recommendations
- Parallel card display for movie comparison
- Responsive design for all devices

#### Setup Instructions

1. Navigate to the movie-recommendation directory:

```bash
cd movie-recommendation
```

2. Install dependencies:

```bash
npm install
```

3. Run the development server:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### DNS Traffic Analysis Tools

A collection of tools for capturing and analyzing DNS traffic using Python and Scapy.

#### Features

- DNS packet capture
- Query logging
- Traffic analysis

#### Files

- `dns_capture.py`: Basic DNS capture tool
- `dns_sniffer.py`: DNS traffic sniffing with detailed output
- `dns_sniffer_add_in_csv.py`: DNS traffic logging to CSV format

#### Setup Instructions

1. Install dependencies:

```bash
pip install scapy
```

2. Run the DNS sniffer:

```bash
sudo python dns_sniffer.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
