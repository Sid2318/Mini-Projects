import React, { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";

import Loader from "./components/Loader";
import DateSelector from "./components/DateSelector";
import NewsCard from "./components/NewsCard";

function App() {
  const [news, setNews] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedDate, setSelectedDate] = useState(new Date());

  // Format date as YYYY-MM-DD
  const formatDate = (date) => {
    const d = new Date(date);
    const month = `${d.getMonth() + 1}`.padStart(2, "0");
    const day = `${d.getDate()}`.padStart(2, "0");
    return `${d.getFullYear()}-${month}-${day}`;
  };

  const fetchNews = async (date) => {
    setLoading(true);
    try {
      const formattedDate = formatDate(date);
      const res = await axios.get(
        `http://localhost:8000/news?date=${formattedDate}`
      );
      setNews(res.data);
    } catch (error) {
      console.error("Error fetching news:", error);
      setNews([]);
    }
    setLoading(false);
  };

  // Fetch news on initial load
  useEffect(() => {
    fetchNews(selectedDate);
  }, []);

  return (
    <div className="App">
      <header>
        <h1>📰 News Summarizer</h1>
        <DateSelector
          selectedDate={selectedDate}
          setSelectedDate={setSelectedDate}
          onDateChange={fetchNews} // only fetch when user finishes picking date
        />
      </header>

      {loading ? (
        <Loader />
      ) : (
        <div className="news-container">
          {news.length === 0 ? (
            <p>No news found for this date.</p>
          ) : (
            news.map((item, index) => (
              <NewsCard
                key={index}
                title={item.title}
                summary={item.summary}
                url={item.url}
              />
            ))
          )}
        </div>
      )}
    </div>
  );
}

export default App;
