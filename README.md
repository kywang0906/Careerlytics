# Careerlytics: AI-Powered Resume Analysis System

Careerlytics is an intelligent resume analysis system designed to help tech professionals optimize their resumes and identify their ideal job roles. By inputting their resume details, users receive AI-driven job predictions, a relevant skill word cloud, and professional rewrite suggestions for their experience descriptions.

## ✨ Features

* **AI-Powered Job Role Prediction**: Utilizes an ONNX model to predict the most suitable tech job role (Software Engineer, Data Scientist, Data Analyst, Project Manager) based on the resume content.
* **Dynamic Skill Word Cloud**: Uses the AutoPhrase algorithm to extract high-quality skill keywords from thousands of aggregated LinkedIn profiles.
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

## 🗂️ Project Structure
```
deployed-files/resume-app/
│
├── fetch-data/ # Scripts for crawling and merging resume data
│ ├── crawl_urls.py # Crawls LinkedIn URLs using search queries
│ ├── merge_data.py # Merges multiple datasets into a unified format
│ └── scrape_bright_data.py # Fetches resume data via Bright Data SERP API
│
├── preprocess/ # Preprocessing and text embedding workflows
│ ├── clean_for_BERT.ipynb # Cleans and formats data for BERT-based models
│ ├── clean_for_LGBM.ipynb # Prepares structured data for LightGBM training
│ ├── combine_files.ipynb # Combines cleaned datasets from different sources
│ └── embed_text.ipynb # Converts textual data into vector embeddings for LightGBM model
│
├── resume-ui/ # Frontend client built with React and Vite
│
├── resume_suggestion_api/ # Backend service for resume rewriting and suggestion generation
│
├── train-model/ # Scripts for training classification and generative models
│ ├── convert_BERT_to_ONNX.ipynb # Convert BERT-based model into ONNX format
│ ├── fine_tune_BERT.ipynb # Fine-tune BERT models as a role classifier
│ ├── train_gemma.ipynb # Fine-tune Gemma 2 model to generate resume content rewrite suggestions
│ ├── gemma_to_gguf.ipynb # Convert Gemma 2 model into GGUF format
│ └── train_lightgbm.ipynb # Train LightGBM model
│
└── README.md # This project documentation
```

## 🧠 Model Overview

* **ONNX Classification Model**: Takes embedded resume data and predicts job roles (Software Engineer, Data Scientist, Data Analyst, & Project Manager).
* **Gemma-2 Generative Model**: A locally hosted large language model (GGUF format), used to rewrite experience descriptions into FAANG-style resume bullets.
* **Text Embedding**: Supports BERT-based embeddings depending on the model training task.

## 💡 How to Use

1. Start the FastAPI backend by navigating to the root directory of the backend (e.g., `resume_suggestion_api/`) and run:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
This launches the FastAPI server at http://localhost:8000, serving endpoints for job prediction and resume rewriting.
Make sure your classification model (ONNX) and LLM (Gemma-2 in GGUF format) are already downloaded and properly referenced in the backend script.

2. Launch the frontend (React + Vite) and open http://localhost:5173 in your browser.
From the `resume-ui/` folder, install dependencies and run the Vite development server:
```bash
npm install
npm run dev
```

3. Upload your resume content, and receive job predictions, key skills, and rewritten suggestions instantly.
<img width="1438" alt="image" src="https://github.com/user-attachments/assets/1cc61f7b-e5c3-4b47-9803-a82fd70f00b3" />

## 🚀 Demo Preview
Here's a quick look at the main features:

![Demo](https://github.com/user-attachments/assets/21eda0b2-13cb-40e6-aa79-5975e0b92dd1)
