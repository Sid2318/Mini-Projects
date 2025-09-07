import React from "react";
import "./Loader.css"; // optional spinner styling

const Loader = () => {
  return (
    <div className="loader">
      <div className="spinner"></div>
      <p>Loading news...</p>
    </div>
  );
};

export default Loader;
