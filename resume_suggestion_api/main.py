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

# --- Pydantic models of Async tasks ---
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

# --- Task management and lock ---
task_results: Dict[str, dict] = {}
llm_lock = threading.Lock()  # 2. Create a global lock

# --- App initialization ---
app = FastAPI(title="Resume API")

# --- In-memory dictionary for storing background task results ---
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

# Implement in background
def rewrite_in_background(task_id: str, req: ClassificationRequest):
    logger.info(f"Task {task_id}: Rewrite task started.")
    
    with llm_lock:
        logger.info(f"Task {task_id}: Lock acquired, starting inference.")
        try:
            task_results[task_id] = {"status": "PROCESSING", "result": None}
            # Collect all the original, unprocessed text blocks
            all_description_blocks = [e.description for e in req.experience if e.description] + \
                                     [p.description for p in req.projects if p.description] + \
                                     [q.description for q in req.publications if q.description]
            
            # Create a new list to hold all split and cleaned individual bullet points
            all_individual_bullets: List[str] = []
            for block in all_description_blocks:
                # Split based on newline characters
                lines = block.split('\n')
                for line in lines:
                    # Remove leading and trailing whitespace from each line
                    cleaned_line = line.strip()
                    # Use regex to remove any leading bullets (•, *, -) and following whitespace
                    # ^[•*-] indicates a line starting with • or * or -
                    # \s* matches zero or more whitespace characters
                    cleaned_line = re.sub(r'^[•*-]\s*', '', cleaned_line)
                    
                    # If the cleaned line still contains text, add it to the final list
                    if cleaned_line:
                        all_individual_bullets.append(cleaned_line)

            items: List[RewriteItem] = []
            # Iterate over the cleaned individual bullets
            for i, bullet in enumerate(all_individual_bullets):
                logger.info(f"Task {task_id}: Rewriting bullet {i+1}/{len(all_individual_bullets)}")
                suggestion = rewrite_bullet(bullet)
                items.append(RewriteItem(original=bullet, suggestion=suggestion))
            
            final_result = RewriteResponse(items=items)
            task_results[task_id] = {"status": "COMPLETED", "result": final_result}
            logger.info(f"Task {task_id}: Rewrite task completed successfully.")

        except Exception as e:
            logger.error(f"Task {task_id}: An error occurred: {e}", exc_info=True)
            task_results[task_id] = {"status": "FAILED", "result": None}


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
    This route immediately returns a task ID and starts the time-consuming rewrite work in the background.
    """
    task_id = str(uuid.uuid4())
    task_results[task_id] = {"status": "PENDING", "result": None}
    background_tasks.add_task(rewrite_in_background, task_id, req)
    return RewriteTaskSubmitResponse(task_id=task_id)

@app.get("/rewrite-status/{task_id}", response_model=RewriteStatusResponse)
def get_rewrite_status(task_id: str):
    """
    Use the previously obtained task_id to check the current status of the task.
    """
    task = task_results.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return RewriteStatusResponse(task_id=task_id, status=task["status"], result=task.get("result"))