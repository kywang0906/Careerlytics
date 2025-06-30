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

  // 新增一個 state 來儲存從後端拿到的 task_id
  const [taskId, setTaskId] = useState(null);

  // 使用 useRef 來儲存 interval ID，以便在需要時清除它
  const intervalRef = useRef(null);

  // 第一個 useEffect：只在組件載入時執行一次，用於提交任務
  useEffect(() => {
    // 防呆：如果沒有 payload，直接報錯
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
        // 將後端回傳的 task_id 存到 state 中
        setTaskId(data.task_id);
        console.log('Task started with ID:', data.task_id);

      } catch (err) {
        console.error(err);
        setError(err.message);
        setLoading(false);
      }
    };

    startRewriteTask();

    // 在組件卸載時，確保清除任何可能存在的 interval
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [payload]); // 依賴 payload，確保只執行一次

  // 第二個 useEffect：當 taskId 被設定後，開始輪詢任務狀態
  useEffect(() => {
    // 只有當 taskId 有值，且還在 loading 狀態時，才開始輪詢
    if (!taskId || !loading) {
      return;
    }

    console.log(`Step 2: Polling status for task ${taskId}...`);

    // 設定一個計時器，每 5 秒鐘查詢一次狀態
    intervalRef.current = setInterval(async () => {
      try {
        const res = await fetch(`/rewrite-status/${taskId}`);
        if (!res.ok) {
          // 如果查詢失敗，也當作任務失敗處理
          throw new Error(`Failed to poll status: ${res.status}`);
        }
        const data = await res.json();

        console.log('Current task status:', data.status);

        if (data.status === 'COMPLETED') {
          // 任務完成！
          clearInterval(intervalRef.current); // 清除計時器
          setItems(data.result.items || []);
          setLoading(false);
          console.log('Task completed!', data.result);
        } else if (data.status === 'FAILED') {
          // 任務失敗！
          clearInterval(intervalRef.current); // 清除計時器
          setError('Rewrite task failed on the server. Please try again.');
          setLoading(false);
        }
        // 如果狀態是 PENDING 或 PROCESSING，則什麼都不做，等待下一次輪詢
      } catch (err) {
        // 捕獲輪詢過程中的網路錯誤
        console.error(err);
        setError(err.message);
        setLoading(false);
        clearInterval(intervalRef.current); // 清除計時器
      }
    }, 5000); // 每 5000 毫秒 (5秒) 查詢一次

    // 這個 useEffect 的清理函式
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [taskId, loading]); // 依賴 taskId 和 loading

  // --- 以下的 JSX 渲染部分維持不變 ---

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