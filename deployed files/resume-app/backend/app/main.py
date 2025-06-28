from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Path, BackgroundTasks
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM, pipeline as hf_pipeline
import onnxruntime as ort
import numpy as np
from collections import Counter
import re
from pathlib import Path as PathLib
from llama_cpp import Llama
import uuid # 引入 uuid 來生成唯一的任務 ID
import logging # 引入 logging 來更好地追蹤背景任務
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic models ---
class EducationEntry(BaseModel):
    school: str
    major: str
    start_year: str
    end_year: str

class ExperienceEntry(BaseModel):
    company: str
    title: str
    description: str
    start_year: str
    end_year: str

class ProjectEntry(BaseModel):
    name: str
    description: str

class PublicationEntry(BaseModel):
    name: str
    description: str

class ClassificationRequest(BaseModel):
    about: str = Field(..., description="Brief self-introduction")
    experience: Optional[List[ExperienceEntry]] = Field(default_factory=list)
    education: Optional[List[EducationEntry]] = Field(default_factory=list)
    projects: Optional[List[ProjectEntry]] = Field(default_factory=list)
    publications: Optional[List[PublicationEntry]] = Field(default_factory=list)
    courses: Optional[List[str]] = Field(default_factory=list)
    certifications: Optional[List[str]] = Field(default_factory=list)

class ClassificationResponse(BaseModel):
    label: str
    score: float

class KeywordsResponse(BaseModel):
    keywords: List[List[int]]

class SkillEntry(BaseModel):
    skill: str
    score: float

class SkillListResponse(BaseModel):
    skills: List[SkillEntry]

class RewriteItem(BaseModel):
    original: str
    suggestion: str

class RewriteResponse(BaseModel):
    items: List[RewriteItem]

# --- Pydantic models of Assync tasks ---
class RewriteTaskSubmitResponse(BaseModel):
    task_id: str

class RewriteStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[RewriteResponse] = None


# --- Mappings ---
LABEL_MAPPING = {
    'LABEL_0': 'Data Analyst',
    'LABEL_1': 'Data Scientist',
    'LABEL_2': 'Project/Product/Program Manager',
    'LABEL_3': 'Software Engineer'
}

# --- 任務管理與鎖 ---
task_results: Dict[str, dict] = {}
llm_lock = threading.Lock() # 2. 在全域建立一個鎖

# --- App initialization ---
app = FastAPI(title="Resume API")

# --- 用於儲存背景任務結果的記憶體字典 ---
task_results: Dict[str, dict] = {}

# --- Load ONNX classification model ---
tokenizer = AutoTokenizer.from_pretrained("/models/onnx-model", use_fast=False)
session = ort.InferenceSession("/models/onnx-model/model.onnx")

# -------- Load Llama.cpp --------
MODEL_GGUF_PATH = "/models/gemma/gemma2_merged-q8_0.gguf"
llm = Llama(
    model_path=MODEL_GGUF_PATH,
    n_ctx=512,
    n_threads=8,
)

def rewrite_bullet(bullet: str) -> str:
    prompt = (
        "Rewrite this resume bullet in FAAMG style:\n"
        f"'{bullet}'\n\n"
        "Answer:"
    )
    resp = llm(
        prompt=prompt,
        max_tokens=32,
        temperature=0.8,
        top_p=0.9
    )
    return resp["choices"][0]["text"].strip()

def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(logits - np.max(logits, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

def build_input_text(req: ClassificationRequest) -> str:
    parts = [f"About: {req.about}"]
    if req.experience:
        exps = [f"{e.company} | {e.title} | {e.description} | {e.start_year} | {e.end_year}[EXP]" for e in req.experience]
        parts.append("Experience: " + " ".join(exps) + "[SEP]")
    if req.education:
        edus = [f"{ed.school} | {ed.major} | {ed.start_year} | {ed.end_year}[EDU]" for ed in req.education]
        parts.append("Education: " + " ".join(edus) + "[SEP]")
    if req.projects:
        projs = [f"{p.name} | {p.description}[PRO]" for p in req.projects]
        parts.append("Projects: " + " ".join(projs) + "[SEP]")
    if req.publications:
        pubs = [f"{p.name} | {p.description}[PUB]" for p in req.publications]
        parts.append("Publications: " + " ".join(pubs) + "[SEP]")
    if req.certifications:
        certs = [f"{c}[CER]" for c in req.certifications]
        parts.append("Certifications: " + " ".join(certs) + "[SEP]")
    if req.courses:
        cous = [f"{c}[COU]" for c in req.courses]
        parts.append("Courses: " + " ".join(cous) + "[SEP]")
    return " ".join(parts)

def collect_bullet(classification):
    bullets = []
    if classification:
        for item in classification:
            if item.description:
                item.description.split("\n")
                for sub in item.description:
                    if sub:
                        bullets.append(sub)
    return bullets

# --- Implement in background ---
def rewrite_in_background(task_id: str, req: ClassificationRequest):
    logger.info(f"Task {task_id}: Rewrite task started.")
    with llm_lock:
        logger.info(f"Task {task_id}: Lock acquired, starting inference.")
        try:
            # Task processing
            task_results[task_id] = {"status": "PROCESSING", "result": None}
            
            # Collect to-be-rewritten bullets
            bullets = collect_bullet(req.experience) + collect_bullet(req.projects) + collect_bullet(req.publications)
            
            # bullets = [e.description for e in req.experience if e.description] + \
            #           [p.description for p in req.projects if p.description] + \
            #           [q.description for q in req.publications if q.description]
            
            items: List[RewriteItem] = []
            for i, b in enumerate(bullets):
                logger.info(f"Task {task_id}: Rewriting bullet {i+1}/{len(bullets)}")
                suggestion = rewrite_bullet(b)
                items.append(RewriteItem(original=b, suggestion=suggestion))
            
            # 任務完成，儲存結果
            final_result = RewriteResponse(items=items)
            task_results[task_id] = {"status": "COMPLETED", "result": final_result}
            logger.info(f"Task {task_id}: Rewrite task completed successfully.")

        except Exception as e:
            logger.error(f"Task {task_id}: An error occurred: {e}", exc_info=True)
            # 任務失敗
            task_results[task_id] = {"status": "FAILED", "result": None}


# --- Endpoints (部分修改) ---
@app.post("/predict", response_model=ClassificationResponse)
def predict(req: ClassificationRequest):
    # ... (此路由不變)
    try:
        text = build_input_text(req)
        inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='np')
        ort_outs = session.run(None, {k:v for k,v in inputs.items()})
        logits = ort_outs[0]
        probs = softmax(logits, axis=1)
        idx = int(np.argmax(probs, axis=1)[0])
        label = LABEL_MAPPING.get(f"LABEL_{idx}", f"LABEL_{idx}")
        return ClassificationResponse(label=label, score=float(probs[0][idx]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 1. 發起改寫任務的路由
@app.post("/start-rewrite", response_model=RewriteTaskSubmitResponse, status_code=202)
def start_rewrite(req: ClassificationRequest, background_tasks: BackgroundTasks):
    """
    這個路由會立即返回一個任務 ID，並在背景開始執行耗時的改寫工作。
    """
    task_id = str(uuid.uuid4())
    task_results[task_id] = {"status": "PENDING", "result": None}
    background_tasks.add_task(rewrite_in_background, task_id, req)
    return RewriteTaskSubmitResponse(task_id=task_id)

# 2. 查詢任務狀態與結果的路由
@app.get("/rewrite-status/{task_id}", response_model=RewriteStatusResponse)
def get_rewrite_status(task_id: str):
    """
    用上一步拿到的 task_id 來查詢任務的目前狀態。
    """
    task = task_results.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return RewriteStatusResponse(task_id=task_id, status=task["status"], result=task.get("result"))
