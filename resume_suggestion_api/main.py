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
import uuid
import logging
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

SKILL_FILES = {
    'da': 'skills/da_skills.txt',
    'ds': 'skills/ds_skills.txt',
    'pm': 'skills/pm_skills.txt',
    'swe': 'skills/swe_skills.txt'
}

# --- 任務管理與鎖 ---
task_results: Dict[str, dict] = {}
llm_lock = threading.Lock() # 2. 在全域建立一個鎖

# --- App initialization ---
app = FastAPI(title="Resume API")

# --- 用於儲存背景任務結果的記憶體字典 ---
task_results: Dict[str, dict] = {}


# --- Load ONNX classification model ---
tokenizer = AutoTokenizer.from_pretrained("onnx-model")
session = ort.InferenceSession("onnx-model/model.onnx")

# -------- Load Llama.cpp --------
MODEL_GGUF_PATH = "gemma2_merged-q8_0.gguf"
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
            task_results[task_id] = {"status": "PROCESSING", "result": None}
            # 1. 先收集所有原始的、未經處理的文字區塊
            all_description_blocks = [e.description for e in req.experience if e.description] + \
                                     [p.description for p in req.projects if p.description] + \
                                     [q.description for q in req.publications if q.description]
            
            # 2. 建立一個新的 list 來存放所有被拆分和清理過的獨立 bullet points
            all_individual_bullets: List[str] = []
            for block in all_description_blocks:
                # 根據換行符號進行切割
                lines = block.split('\n')
                for line in lines:
                    # 去除每行前後的空白
                    cleaned_line = line.strip()
                    # 使用正規表示式去除行首可能存在的項目符號 (•, *, -) 和之後的空白
                    # ^[•*-] 表示以 • 或 * 或 - 開頭
                    # \s* 表示零個或多個空白字元
                    cleaned_line = re.sub(r'^[•*-]\s*', '', cleaned_line)
                    
                    # 如果清理後該行還有內容，則將其加入到最終的列表中
                    if cleaned_line:
                        all_individual_bullets.append(cleaned_line)

            # --- 修改結束 ---

            items: List[RewriteItem] = []
            # 現在，我們遍歷的是被拆分好的 all_individual_bullets 列表
            for i, bullet in enumerate(all_individual_bullets):
                logger.info(f"Task {task_id}: Rewriting bullet {i+1}/{len(all_individual_bullets)}")
                suggestion = rewrite_bullet(bullet)
                # original 欄位現在儲存的是清理過後的單一句子
                items.append(RewriteItem(original=bullet, suggestion=suggestion))
            
            final_result = RewriteResponse(items=items)
            task_results[task_id] = {"status": "COMPLETED", "result": final_result}
            logger.info(f"Task {task_id}: Rewrite task completed successfully.")

        except Exception as e:
            logger.error(f"Task {task_id}: An error occurred: {e}", exc_info=True)
            task_results[task_id] = {"status": "FAILED", "result": None}


# --- Endpoints ---
@app.post("/predict", response_model=ClassificationResponse)
def predict(req: ClassificationRequest):
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

@app.post("/start-rewrite", response_model=RewriteTaskSubmitResponse, status_code=202)
def start_rewrite(req: ClassificationRequest, background_tasks: BackgroundTasks):
    """
    這個路由會立即返回一個任務 ID，並在背景開始執行耗時的改寫工作。
    """
    task_id = str(uuid.uuid4())
    task_results[task_id] = {"status": "PENDING", "result": None}
    background_tasks.add_task(rewrite_in_background, task_id, req)
    return RewriteTaskSubmitResponse(task_id=task_id)

@app.get("/rewrite-status/{task_id}", response_model=RewriteStatusResponse)
def get_rewrite_status(task_id: str):
    """
    用上一步拿到的 task_id 來查詢任務的目前狀態。
    """
    task = task_results.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return RewriteStatusResponse(task_id=task_id, status=task["status"], result=task.get("result"))