import React, { useState } from "react";
import axios from "axios";

function App() {
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [context, setContext] = useState("");

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      alert("Please select at least one file.");
      return;
    }

    const formData = new FormData();
    files.forEach((file) => {
      formData.append("files", file);
    });

    try {
      setUploading(true);
      const res = await axios.post("http://localhost:8000/upload/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      alert("✅ Files uploaded: " + res.data.files_processed.join(", "));
      setFiles([]);
    } catch (err) {
      console.error(err);
      alert("❌ Upload failed!");
    } finally {
      setUploading(false);
    }
  };

  const handleAsk = async () => {
    if (!question.trim()) {
      alert("Please enter a question.");
      return;
    }

    try {
      const res = await axios.get("http://localhost:8000/ask/", {
        params: { q: question },
      });
      setAnswer(res.data.answer);
      setContext(res.data.context);
    } catch (err) {
      console.error(err);
      alert("❌ Error fetching answer.");
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f9fafb",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        padding: "40px 20px",
        fontFamily: "Arial, sans-serif",
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: "800px",
          background: "white",
          padding: "30px",
          borderRadius: "12px",
          boxShadow: "0 6px 18px rgba(0,0,0,0.1)",
        }}
      >
        <h1 style={{ color: "#1e3a8a", marginBottom: "20px" }}>
          📚 RAG Agent (Teacher Mode)
        </h1>

        {/* File Upload Section */}
        <section style={{ marginBottom: "40px" }}>
          <h2 style={{ marginBottom: "10px", color: "#111827" }}>
            Upload Documents
          </h2>
          <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
            <input
              type="file"
              multiple
              onChange={handleFileChange}
              style={{
                flex: "1",
                padding: "8px",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
              }}
            />
            <button
              onClick={handleUpload}
              disabled={uploading}
              style={{
                padding: "10px 20px",
                backgroundColor: uploading ? "#9ca3af" : "#2563eb",
                color: "white",
                border: "none",
                borderRadius: "6px",
                cursor: uploading ? "not-allowed" : "pointer",
                fontWeight: "bold",
              }}
            >
              {uploading ? "Uploading..." : "Upload"}
            </button>
          </div>

          {files.length > 0 && (
            <ul style={{ marginTop: "15px", color: "#374151" }}>
              {files.map((file, index) => (
                <li key={index} style={{ fontSize: "14px" }}>
                  📄 {file.name}
                </li>
              ))}
            </ul>
          )}
        </section>

        {/* Question Answer Section */}
        <section>
          <h2 style={{ marginBottom: "10px", color: "#111827" }}>
            Ask a Question
          </h2>
          <div style={{ display: "flex", gap: "10px" }}>
            <input
              type="text"
              placeholder="Type your question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              style={{
                flex: "1",
                padding: "10px",
                border: "1px solid #d1d5db",
                borderRadius: "6px",
              }}
            />
            <button
              onClick={handleAsk}
              style={{
                padding: "10px 20px",
                backgroundColor: "#16a34a",
                color: "white",
                border: "none",
                borderRadius: "6px",
                cursor: "pointer",
                fontWeight: "bold",
              }}
            >
              Ask
            </button>
          </div>

          {answer && (
            <div
              style={{
                marginTop: "25px",
                padding: "20px",
                backgroundColor: "#f0fdf4",
                border: "1px solid #bbf7d0",
                borderRadius: "10px",
                boxShadow: "0 3px 8px rgba(0,0,0,0.05)",
              }}
            >
              <h3
                style={{
                  color: "#166534",
                  marginBottom: "10px",
                  fontSize: "18px",
                }}
              >
                ✅ Answer
              </h3>
              <p
                style={{
                  fontSize: "16px",
                  lineHeight: "1.6",
                  color: "#065f46",
                  margin: 0,
                }}
              >
                {answer}
              </p>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

export default App;
