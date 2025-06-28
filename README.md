# Careerlytics - AI-Powered Resume Analysis System

Careerlytics is an intelligent resume analysis system designed to help tech professionals optimize their resumes and identify their ideal job roles. By inputting their resume details, users receive AI-driven job predictions, a relevant skill word cloud, and professional rewrite suggestions for their experience descriptions.

## ✨ Features

* **AI-Powered Job Role Prediction**: Utilizes an ONNX model to predict the most suitable tech job role (e.g., Software Engineer, Data Scientist, Data Analyst, Project Manager) based on the resume content.
* **Dynamic Skill Word Cloud**: Dynamically generates a word cloud of key skills based on the predicted job role, helping users identify skill gaps.
* **FAANG-Style Resume Rewrites**: Leverages a locally-hosted Large Language Model (Gemma-2) to transform standard experience descriptions into impactful, professional bullet points.
* **Asynchronous Task Handling**: Employs an asynchronous polling mechanism for the time-consuming resume rewrite task, preventing request timeouts and improving user experience.
* **Interactive Web Interface**: A modern, responsive frontend built with React and Vite.

## ⚙️ Architecture

This project uses a decoupled frontend and backend architecture.

* **Frontend**: Built with React (Vite), responsible for data input and results visualization.
* **Backend**: A RESTful API service built with Python and FastAPI.
* **Models**:
    * **Classification Model**: Deployed with `ONNX Runtime` for efficient job role prediction.
    * **Generative Model**: Runs a GGUF-formatted Gemma-2 model on the CPU using `llama-cpp-python` for text generation.
