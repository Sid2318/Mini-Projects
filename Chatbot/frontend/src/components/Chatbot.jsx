import { useState, useEffect, useRef } from "react";
import axios from "axios";

const Chatbot = () => {
  const [messages, setMessages] = useState([
    {
      text: "Hello! How can I assist you today?",
      sender: "bot",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);

  const handleSend = async () => {
    if (input.trim()) {
      // Store input text as we'll clear the input immediately
      const userInput = input.trim();

      // Add user message
      setMessages([
        ...messages,
        { text: userInput, sender: "user", timestamp: new Date() },
      ]);
      setInput("");

      // Show typing indicator
      setIsTyping(true);

      try {
        // Send message to backend
        const response = await axios.post("http://localhost:3000/api/chat", {
          message: userInput,
        });

        // Add bot response
        setIsTyping(false);
        setMessages((prev) => [
          ...prev,
          {
            text:
              response.data.response ||
              "Sorry, I couldn't process that request.",
            sender: "bot",
            timestamp: new Date(),
          },
        ]);
      } catch (error) {
        console.error("Error communicating with backend:", error);

        // Show error message
        setIsTyping(false);
        setMessages((prev) => [
          ...prev,
          {
            text: "Sorry, I'm having trouble connecting to my brain right now. Please try again later.",
            sender: "bot",
            timestamp: new Date(),
          },
        ]);
      }
    }
  };

  // Auto-scroll to the bottom when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <>
      <div className="container d-flex justify-content-center align-items-center min-vh-100 py-4">
        <div className="card shadow w-100" style={{ maxWidth: "500px" }}>
          {/* Header */}
          <div className="card-header bg-primary text-white py-3">
            <h5 className="mb-0">Chatbot Assistant</h5>
          </div>

          {/* Messages */}
          <div
            className="card-body p-3"
            style={{ height: "400px", overflowY: "auto" }}
          >
            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`d-flex mb-3 ${
                  msg.sender === "user"
                    ? "justify-content-end"
                    : "justify-content-start"
                }`}
              >
                {msg.sender === "bot" && (
                  <div className="avatar me-2 align-self-end">
                    <div
                      className="rounded-circle bg-info text-white d-flex align-items-center justify-content-center"
                      style={{ width: "32px", height: "32px" }}
                    >
                      <i className="bi bi-robot"></i>
                    </div>
                  </div>
                )}
                <div className="message-container">
                  <div
                    className={`p-3 rounded-3 ${
                      msg.sender === "user"
                        ? "bg-primary text-white"
                        : "bg-light"
                    }`}
                    style={{ maxWidth: "280px" }}
                  >
                    {msg.text}
                  </div>
                  <div
                    className={`text-muted small mt-1 ${
                      msg.sender === "user" ? "text-end" : ""
                    }`}
                  >
                    {msg.timestamp.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                </div>
                {msg.sender === "user" && (
                  <div className="avatar ms-2 align-self-end">
                    <div
                      className="rounded-circle bg-secondary text-white d-flex align-items-center justify-content-center"
                      style={{ width: "32px", height: "32px" }}
                    >
                      <i className="bi bi-person"></i>
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Typing indicator */}
            {isTyping && (
              <div className="d-flex mb-3 justify-content-start">
                <div className="avatar me-2 align-self-end">
                  <div
                    className="rounded-circle bg-info text-white d-flex align-items-center justify-content-center"
                    style={{ width: "32px", height: "32px" }}
                  >
                    <i className="bi bi-robot"></i>
                  </div>
                </div>
                <div
                  className="p-3 rounded-3 bg-light"
                  style={{ maxWidth: "100px" }}
                >
                  <div className="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="card-footer border-top-0 bg-white p-3">
            <div className="input-group">
              <input
                type="text"
                className="form-control rounded-start"
                placeholder="Type a message..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSend()}
              />
              <button
                onClick={handleSend}
                className="btn btn-primary"
                disabled={!input.trim()}
              >
                <i className="bi bi-send-fill"></i>
              </button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default Chatbot;
