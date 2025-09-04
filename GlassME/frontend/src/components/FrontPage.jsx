import { useState } from "react";

function FrontPage() {
  const [number, setNumber] = useState("");
  const [videoStarted, setVideoStarted] = useState(false);

  const sendNumber = async () => {
    const response = await fetch("http://localhost:5000/api/data", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ number: Number(number) }),
    });

    const data = await response.json();
    console.log("Backend response:", data);
    if (data.success) {
      setVideoStarted(true);
    }
  };

  return (
    <div>
      <h1>Glasses Try-On</h1>
      <input
        type="number"
        value={number}
        onChange={(e) => setNumber(e.target.value)}
        placeholder="Enter glasses number"
      />
      <button onClick={sendNumber}>Send</button>

      {videoStarted && (
        <div>
          <h2>Video Stream</h2>
          <img src="http://localhost:7000/video" alt="Video Stream" />
        </div>
      )}
    </div>
  );
}

export default FrontPage;
