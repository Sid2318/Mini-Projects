import React from "react";
import "./NewsCard.css";

const NewsCard = ({ title, summary, url }) => {
  const handleClick = () => {
    window.open(url, "_blank");
  };

  return (
    <div onClick={handleClick} className="news-card">
      <h2>{title}</h2>
      <p>{summary}</p>
    </div>
  );
};

export default NewsCard;
