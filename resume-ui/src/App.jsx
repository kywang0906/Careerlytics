import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Header from './components/Header';
import LandingPage from './components/LandingPage';
import PredictionAndWordCloud from './components/PredictionAndWordCloud'
import RewriteSuggestions from './components/RewriteSuggestions'

function App() {
  return (
    <Router>
      <Header />
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/analysis" element={<PredictionAndWordCloud />} />
        <Route path="/rewrite" element={<RewriteSuggestions />} />
      </Routes>
    </Router>
  )
}

export default App
