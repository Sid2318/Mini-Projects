import express from "express";
import cors from "cors";
import fetch from "node-fetch"; // Add fetch import

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.send("Backend is running 🚀");
});

app.post("/api/data", async (req, res) => {
  const { number } = req.body;
  console.log("Number received from frontend:", number);

  try {
    // Forward number to Python server
    const pyRes = await fetch("http://localhost:7000/set-glass", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ number }),
    });
    const result = await pyRes.json();
    res.json(result);
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
