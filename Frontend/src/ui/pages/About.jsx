import React from "react";
import contact1 from "../assets/contact.png";
import "../styles/About.css";
import Contact from "./Contact";
function About() {
  return (
    <div className="about">
      <div
        className="aboutTop"
        style={{ backgroundImage: `url(${contact1})` }}
      ></div>
      <div className="aboutBottom">
        <h1> ABOUT US</h1>
        <p>
         Time-series data is critical in domains such as financial forecasting,
          climate modeling, industrial automation, and IoT analytics. Existing tools
          like Pandas, SQL, Apache Spark, and R's statistical libraries require verbose
          scripts, complex syntax, and additional dependencies, making them difficult for
         non-programmers to use. Furthermore, these tools often lack built-in support for
         real-time streaming data and forecasting capabilities without integrating external 
          machine learning libraries. This gap in usability and performance creates challenges 
          for time-series analytics. ChronoLang is proposed as a domain-specific language (DSL)
         designed to simplify time-series analytics, forecasting, and real-time data processing.
         It aims to provide an intuitive, declarative syntax with built-in forecasting models and
          real-time streaming support, addressing the limitations of traditional tools.
        </p>
      </div>
    </div>
  );
}

export default About;