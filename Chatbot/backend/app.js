import express from "express";
import axios from "axios";
import cors from "cors";

const app = express();

app.use(cors());
app.use(express.json());

app.use((req, res, next) => {
  console.log(`Request received: ${req.method} ${req.url}`);
  next();
});

// Health check endpoint
app.get("/", (req, res) => {
  res.json({ status: "ok", message: "Node.js API server is running" });
});

app.post("/api/chat", async (req, res) => {
  try {
    const { message } = req.body;
    console.log(`Processing message: "${message}"`);

    // Call Python chatbot backend
    console.log("Sending request to Python backend...");
    const response = await axios.post("http://localhost:5000/chat", {
      message,
    });

    console.log("Received response from Python backend:", response.data);
    res.json(response.data);
  } catch (err) {
    console.error("Error communicating with Python backend:", err.message);
    if (err.code === "ECONNREFUSED") {
      console.error(
        "Connection refused - Is the Python server running on port 5000?"
      );
      res.status(503).json({
        error: "Brain service unavailable",
        response:
          "Sorry, my brain is offline right now. Please try again later.",
      });
    } else {
      console.error("Full error:", err);
      res.status(500).json({
        error: "Error communicating with Python backend",
        response: "I'm having trouble thinking right now. Please try again.",
      });
    }
  }
});

app.listen(process.env.PORT || 3000, () => {
  console.log(
    `Node.js server running on port http://localhost:${
      process.env.PORT || 3000
    }`
  );
  console.log("API endpoint available at: http://localhost:3000/api/chat");
});
