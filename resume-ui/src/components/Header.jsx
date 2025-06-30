import React from 'react';
import { Link } from 'react-router-dom';
import '../App.css'

const Header = () => {
  return (
    <header className="header">
      <div className="container">
        <img src="/logo.png" alt="Logo" style={{ height: '40px', marginRight: '10px' }} />  
        <Link to="/" className="header-brand">
          Careerlytics
        </Link>
      </div>
    </header>
  );
};

export default Header;