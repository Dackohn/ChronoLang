import React from "react";
import { Link } from "react-router-dom";
import BannerImage from "../assets/aa.jpg";
import "../styles/Home.css";

function Home() {
  return (
    <div className="home" style={{ backgroundImage: `url(${BannerImage})` }}>
      <div className="headerContainer">
        <h1> ChronoLang </h1>
        <p> DSL for time-series analytics, forecasting, and real-time data processing</p>
        <Link to="/work">
          <button> GET STARTED </button>
        </Link>
      </div>
    </div>
  );
}

export default Home;