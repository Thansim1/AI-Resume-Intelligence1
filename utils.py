"""Curated skill extraction and contact-information helpers."""
from __future__ import annotations

import re
from typing import Dict, List, Set

SKILL_CATEGORIES: Dict[str, List[str]] = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "c",
        "r", "scala", "kotlin", "swift", "go", "golang", "rust", "ruby",
        "php", "matlab", "bash", "shell", "perl", "haskell", "lua",
    ],
    "ML / AI": [
        "machine learning", "deep learning", "neural network", "nlp",
        "natural language processing", "computer vision", "reinforcement learning",
        "transfer learning", "bert", "gpt", "llm", "large language model",
        "transformers", "hugging face", "langchain", "rag", "generative ai",
        "diffusion model", "object detection", "yolo",
    ],
    "ML Frameworks & Libraries": [
        "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
        "xgboost", "lightgbm", "catboost", "fastai", "jax", "opencv",
        "nltk", "spacy", "gensim", "sentence transformers",
    ],
    "Data & Analytics": [
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "tableau", "power bi", "looker", "excel", "sql", "nosql",
        "data analysis", "data analytics", "data visualization", "etl",
        "data pipeline", "statistics", "hypothesis testing", "a/b testing",
    ],
    "Databases": [
        "mysql", "postgresql", "postgres", "sqlite", "mongodb", "redis",
        "cassandra", "dynamodb", "elasticsearch", "neo4j", "oracle",
        "firebase", "supabase", "pinecone", "chroma", "weaviate",
    ],
    "Cloud & MLOps": [
        "aws", "azure", "gcp", "google cloud", "heroku", "docker",
        "kubernetes", "k8s", "mlflow", "airflow", "dvc", "wandb",
        "weights & biases", "sagemaker", "vertex ai", "databricks",
        "spark", "hadoop", "kafka", "terraform", "ci/cd", "jenkins",
        "github actions", "fastapi", "flask", "django", "streamlit",
    ],
    "Tools & Practices": [
        "git", "github", "gitlab", "bitbucket", "linux", "unix",
        "rest api", "graphql", "microservices", "agile", "scrum",
        "jira", "confluence", "notion", "latex", "jupyter", "vscode",
        "postman", "unit testing", "pytest", "tdd", "oop",
    ],
    "Soft Skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "critical thinking", "project management", "collaboration",
        "presentation", "research", "analytical thinking",
    ],
}

ALL_SKILLS: Set[str] = {skill for group in SKILL_CATEGORIES.values() for skill in group}


def _normalize(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return re.sub(r"\s+", " ", (text or "").lower().strip())


def _skill_pattern(skill: str) -> re.Pattern:
    """Match a skill as a phrase, without matching it inside a larger word."""
    escaped = re.escape(skill)
    # Lookarounds handle punctuation-bearing skills such as C++, C# and CI/CD.
    return re.compile(r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])", re.IGNORECASE)


def extract_skills(text: str) -> Dict[str, List[str]]:
    """Extract detected skills grouped by category."""
    normalized = _normalize(text)
    found: Dict[str, List[str]] = {}
    for category, skills in SKILL_CATEGORIES.items():
        matched = [skill for skill in skills if _skill_pattern(skill).search(normalized)]
        if matched:
            found[category] = matched
    return found


def extract_skills_flat(text: str) -> Set[str]:
    """Return all detected skills as a set."""
    normalized = _normalize(text)
    return {skill for skill in ALL_SKILLS if _skill_pattern(skill).search(normalized)}


def get_missing_skills(resume_skills: Set[str], job_skills: Set[str]) -> Set[str]:
    return job_skills - resume_skills


def get_matching_skills(resume_skills: Set[str], job_skills: Set[str]) -> Set[str]:
    return resume_skills & job_skills


def skill_match_rate(resume_skills: Set[str], job_skills: Set[str]) -> float:
    """Percentage of detected job-description skills also found in the resume."""
    if not job_skills:
        return 0.0
    return round(len(resume_skills & job_skills) / len(job_skills) * 100, 1)


def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract a likely email address and phone number (best-effort only)."""
    info: Dict[str, str] = {}
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text or "")
    if email_match:
        info["email"] = email_match.group(0)
    phone_match = re.search(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)", text or "")
    if phone_match:
        info["phone"] = phone_match.group(1).strip()
    return info


def extract_section(text: str, section_name: str) -> str:
    """Best-effort section extraction based on a heading; not a full resume parser."""
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(section_name)}\s*$\n(.*?)(?=^\s*[A-Z][A-Z &/]{2,}\s*$|\Z)"
    )
    match = pattern.search(text or "")
    return match.group(1).strip() if match else ""
