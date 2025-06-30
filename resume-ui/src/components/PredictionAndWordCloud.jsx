import React, { useEffect, useRef, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import WordCloud from 'wordcloud';

const HARD_SKILLS = {
  'Software Engineer': [
    ['Operating Systems', 0.9851102115],
    ['Reinforcement Learning', 0.9803669996],
    ['Multivariable Calculus', 0.9803503152],
    ['Virtual Reality', 0.9795895506],
    ['Random Forest', 0.9789653342],
    ['Supply Chain', 0.978205813],
    ['Augmented Reality', 0.9765214698],
    ['Data Structures', 0.97601578],
    ['TCP/IP', 0.9756414368],
    ['Logistic Regression', 0.9749559964],
    ['Programming Languages', 0.9743882053],
    ['Natural Language Processing', 0.9738262859],
    ['Linear Algebra', 0.9730389301],
    ['Differential Equations', 0.9721081995],
    ['Convolutional Neural Networks', 0.9699182022],
    ['Machine Learning', 0.9675232181],
    ['Deep Learning', 0.9672925359],
    ['Sentiment Analysis', 0.9671902602],
    ['PL/SQL', 0.966363402],
    ['Quality Assurance', 0.9663525468],
  ],
  'Data Analyst': [
    ['Linear Algebra', 0.9717710693],
    ['Predictive Analytics', 0.9711620276],
    ['Neural Networks', 0.9710017973],
    ['Human Resources', 0.9701375094],
    ['Google Adwords', 0.9695682124],
    ['Giac Certifications', 0.9686989094],
    ['Artificial Intelligence', 0.9684273425],
    ['Cloud Computing', 0.9678758909],
    ['Data Mining', 0.9663676372],
    ['Supply Chain', 0.9661982447],
    ['Machine Learning', 0.9659802469],
    ['Mental Health', 0.9659750614],
    ['Exploratory Data Analysis', 0.9652226162],
    ['Accounts Payable', 0.9641924095],
    ['Conflict Resolution', 0.9635999021],
    ['Criminal Justice', 0.9628792387],
    ['Graphic Design', 0.9627949154],
    ['Deep Learning', 0.9625776991],
    ['Logistic Regression', 0.9617274845],
    ['Digital Marketing', 0.9612168628],
  ],
  'Data Scientist': [
    ['Raspberry PI', 0.9852167955],
    ['Differential Equations', 0.9839343842],
    ['Signal Processing', 0.9823108347],
    ['Random Forest', 0.9816096851],
    ['Reinforcement Learning', 0.9806895715],
    ['Remote Sensing', 0.9795775728],
    ['Partial Differential Equations', 0.9791707952],
    ['Supply Chain', 0.9782974643],
    ['Virtual Reality', 0.9762357805],
    ['Operating Systems', 0.9752830997],
    ['Augmented Reality', 0.9751315891],
    ['Cognitive Science', 0.975018655],
    ['Linear Algebra', 0.9743969861],
    ['Quantum Computing', 0.9739102836],
    ['Machine Translation', 0.9732057792],
    ['Multivariable Calculus', 0.9729522122],
    ['Computational Linguistics', 0.9727190887],
    ['Sentiment Analysis', 0.9723105509],
    ['Fluid Mechanics', 0.9716693046],
    ['Statistical Inference', 0.9708942044],
  ],
  'Project/Product/Program Manager': [
    ['Linear Algebra', 0.9718836949],
    ['Criminal Justice', 0.9705338729],
    ['Public Relations', 0.9675066339],
    ['Neural Networks', 0.9657552825],
    ['Active Directory', 0.9649877854],
    ['Natural Language Processing', 0.9621256961],
    ['Accounts Payable', 0.9617900121],
    ['Programming Languages', 0.9614347362],
    ['Machine Learning', 0.9605178945],
    ['Public Speaking', 0.9599423572],
    ['Augmented Reality', 0.9598905408],
    ['Human Resources', 0.9572893361],
    ['Cloud Computing', 0.9572787368],
    ['Disaster Recovery', 0.956299067],
    ['Data Structures', 0.9561336375],
    ['Operating Systems', 0.9560520441],
    ['Deep Learning', 0.9555578401],
    ['Supply Chain', 0.9552513994],
    ['Lean Manufacturing', 0.9546955682],
    ['Big Data', 0.9545338302],
  ],
};

const PredictionAndWordCloud = () => {
  const navigate = useNavigate();
  const { state } = useLocation();
  const { label } = state;
  const canvasRef = useRef(null);

  useEffect(() => {
    const skillsRaw = HARD_SKILLS[label] || HARD_SKILLS['Data Scientist'];
    // Map 0.95~0.99 to [20~80]
    const list = skillsRaw.map(([skill, score]) => {
      const size = Math.max(20, Math.round((score - 0.95) / 0.05 * 60));
      return [skill, size];
    });

    if (canvasRef.current) {
      WordCloud(canvasRef.current, {
        list,
        gridSize: 12,
        weightFactor: s => s,
        fontFamily: 'sans-serif',
        rotateRatio: 0.5,
        backgroundColor: '#fff',
        shrinkToFit: true,
      });
    }
  }, [label]);

  return (
    <div className="container mt-5">
      <h2 className="mb-4 text-center">Step 2: Resume Analysis Result</h2>
      <div className="mb-4">
        <p className="h5">🎯 Predicted Target Position:</p>
        <p className="h4 text-primary fw-bold">{label}</p>
      </div>
      <div className="mb-4">
        <p className="h5 mb-3">📊 Recommended Skills Word Cloud:</p>
        <div className="d-flex justify-content-center">
          <canvas
            ref={canvasRef}
            width={600}
            height={400}
            className="border border-secondary rounded"
          />
        </div>
      </div>
      <div className="text-center">
        <button
          className="btn btn-primary btn-lg px-5 py-2 btn-submit"
          onClick={() => navigate('/rewrite', { state })}
        >
          See Resume Suggestions
        </button>
      </div>
    </div>
  );
};

export default PredictionAndWordCloud;
