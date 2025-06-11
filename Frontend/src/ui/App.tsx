import './App.css'
import Footer from "./components/Footer.jsx";
import Home from "./pages/Home";
import Work from "./pages/Work";

import About from "./pages/About";
import Contact from "./pages/Contact";
import Navbar from "./components/Navbar";
import { HashRouter as Router, Routes, Route } from 'react-router-dom'

function App() {
  return (
    <div className="App">
      <Router>
        <Navbar />
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/work" element={<Work />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
        </Routes>
        <Footer />
      </Router>
    </div>
  )
}

export default App