import React, { useEffect, useState, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';

const RewriteSuggestions = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Get state from first page's payload
  const { payload } = location.state || {};

  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Add a state to store the task_id returned from the backend
  const [taskId, setTaskId] = useState(null);

  // Use useRef to store interval ID so it can be cleared when needed
  const intervalRef = useRef(null);

  // First useEffect: runs only once on component mount, used to submit the task
  useEffect(() => {
    // Guard clause: show error if there's no payload
    if (!payload) {
      setError('Missing resume data. Please go back and try again.');
      setLoading(false);
      return;
    }

    const startRewriteTask = async () => {
      try {
        console.log('Step 1: Starting rewrite task...');
        const res = await fetch('/start-rewrite', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(`Failed to start task: ${res.status} ${text}`);
        }

        const data = await res.json();
        // Save the task_id returned from backend into state
        setTaskId(data.task_id);
        console.log('Task started with ID:', data.task_id);

      } catch (err) {
        console.error(err);
        setError(err.message);
        setLoading(false);
      }
    };

    startRewriteTask();

    // Ensure any existing interval is cleared on component unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [payload]); // Depends on payload, only run once

  // Second useEffect: start polling once taskId is set
  useEffect(() => {
    // Only start polling if taskId is set and still in loading state
    if (!taskId || !loading) {
      return;
    }

    console.log(`Step 2: Polling status for task ${taskId}...`);

    // Set a timer to poll status every 5 seconds
    intervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/rewrite-status/${taskId}`);
        if (!res.ok) {
          // Treat failed status check as task failure
          throw new Error(`Failed to poll status: ${res.status}`);
        }
        const data = await res.json();

        console.log('Current task status:', data.status);

        if (data.status === 'COMPLETED') {
          // Task is completed!
          clearInterval(intervalRef.current); // Clear the timer
          setItems(data.result.items || []);
          setLoading(false);
          console.log('Task completed!', data.result);
        } else if (data.status === 'FAILED') {
          // Task failed!
          clearInterval(intervalRef.current); // Clear the timer
          setError('Rewrite task failed on the server. Please try again.');
          setLoading(false);
        }
        // Do nothing if status is PENDING or PROCESSING; wait for next poll
      } catch (err) {
        // Catch network errors during polling
        console.error(err);
        setError(err.message);
        setLoading(false);
        clearInterval(intervalRef.current); // Clear the timer
      }
    }, 5000); // Poll every 5000ms (5 seconds)

    // Cleanup function for this useEffect
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [taskId, loading]); // Depends on taskId and loading

  if (loading) {
    return <div className="text-center mt-5">
      <div className="spinner-border text-primary" role="status">
        <span className="visually-hidden">Loading...</span>
      </div>
      <p className="mt-3">AI is generating suggestions, this may take several minutes...</p>
    </div>;
  }
  if (error) {
    return (
      <div className="container text-center mt-5 alert alert-danger">
        <h4 className="alert-heading">An Error Occurred</h4>
        <p>{error}</p>
        <button className="btn btn-secondary" onClick={() => navigate('/')}>
          Back Home
        </button>
      </div>
    );
  }

  return (
    <div style={{ width: '100%', maxWidth: '800px', margin: '0 auto', padding: '2rem' }}>
      <h2 className="text-center mb-4">Step 3: Rewrite Suggestions</h2>
      {items.length === 0 && (
          <div className="alert alert-warning text-center">
            No suggestions were returned. This could be due to empty descriptions in your resume.
          </div>
      )}
      {items.map((item, idx) => (
        <div key={idx} className="card mb-3">
          <div className="card-body">
            <h6 className="card-subtitle mb-2 text-muted">Original:</h6>
            <p className="card-text">{item.original}</p>
            <hr />
            <h6 className="card-subtitle mb-2 text-success">Suggestion:</h6>
            <p className="card-text fw-bold">{item.suggestion}</p>
          </div>
        </div>
      ))}
      <div className="text-center mt-4">
        <button
          className="btn btn-success px-4"
          onClick={() => navigate('/')}
        >
          Done
        </button>
      </div>
    </div>
  );
};

export default RewriteSuggestions;