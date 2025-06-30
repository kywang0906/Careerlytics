import React from 'react';
import ResumeInputForm from './ResumeInputForm';

const LandingPage = () => {
  return (
    <div>
      <div className="hero">
        <h1 className="hero-title">Unlock Your Tech Career Potential</h1>
        <p className="hero-subtitle">
          Paste your resume details below. Careerlytics will analyze your profile and match you with the perfect tech job.
        </p>
      </div>

      <ResumeInputForm />
    </div>
  );
};

export default LandingPage;