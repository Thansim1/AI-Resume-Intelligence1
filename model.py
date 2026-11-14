"""TF-IDF similarity scoring and rule-based improvement suggestions."""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "as", "if", "then", "than",
    "that", "this", "these", "those", "i", "you", "he", "she", "it", "we",
    "they", "me", "him", "her", "us", "them", "my", "your", "his", "its",
    "our", "their", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "each", "every", "more", "most", "other", "such", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "up", "about", "against", "while", "per", "etc", "also",
}


def preprocess(text: str) -> List[str]:
    """Lowercase, tokenize, preserve technical tokens such as C++ and C#."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s+#.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    # Keep one-character technical language tokens C and R, but remove other
    # one-character fragments caused by punctuation or formatting.
    return [t for t in tokens if t not in STOPWORDS and (len(t) > 1 or t in {"c", "r"})]


def build_vocabulary(docs: List[List[str]]) -> List[str]:
    """Build a deterministic sorted vocabulary."""
    vocab: Set[str] = set()
    for doc in docs:
        vocab.update(doc)
    return sorted(vocab)


def tf(tokens: List[str], vocab: List[str]) -> List[float]:
    """Normalized term-frequency vector."""
    counts = Counter(tokens)
    total = max(len(tokens), 1)
    return [counts.get(word, 0) / total for word in vocab]


def idf(vocab: List[str], docs: List[List[str]]) -> List[float]:
    """Smoothed inverse document frequency."""
    n_docs = len(docs)
    if n_docs == 0:
        return [0.0 for _ in vocab]
    return [
        math.log((n_docs + 1) / (sum(word in set(doc) for doc in docs) + 1)) + 1
        for word in vocab
    ]


def tfidf_vector(tokens: List[str], vocab: List[str], idf_scores: List[float]) -> List[float]:
    """Multiply term frequency by inverse document frequency."""
    return [term_freq * inverse_freq
            for term_freq, inverse_freq in zip(tf(tokens, vocab), idf_scores)]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Return cosine similarity in the range 0.0–1.0."""
    if len(vec_a) != len(vec_b):
        raise ValueError("Vectors must have the same dimensions.")
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (mag_a * mag_b)))


def compute_match_score(resume_text: str, job_text: str) -> float:
    """Return TF-IDF cosine similarity as a percentage from 0 to 100."""
    resume_tokens, job_tokens = preprocess(resume_text), preprocess(job_text)
    if not resume_tokens or not job_tokens:
        return 0.0
    vocab = build_vocabulary([resume_tokens, job_tokens])
    idf_scores = idf(vocab, [resume_tokens, job_tokens])
    resume_vec = tfidf_vector(resume_tokens, vocab, idf_scores)
    job_vec = tfidf_vector(job_tokens, vocab, idf_scores)
    return round(cosine_similarity(resume_vec, job_vec) * 100, 1)


SUGGESTIONS: Dict[str, List[str]] = {
    "ML / AI": [
        "Add a practical ML project (for example, a classifier or recommendation system) and explain your contribution.",
        "Describe model training, evaluation metrics and validation methods you have actually used.",
        "Link relevant research, coursework or open-source contributions.",
    ],
    "ML Frameworks & Libraries": [
        "List the ML libraries you have genuinely used, such as scikit-learn, TensorFlow or PyTorch.",
        "Add a small project demonstrating one framework and link its repository.",
    ],
    "Programming Languages": [
        "Add a Technical Skills section listing languages you can confidently explain.",
        "Link code samples or projects that demonstrate your programming experience.",
    ],
    "Data & Analytics": [
        "Add a data analysis project with a clear question, dataset, methods and findings.",
        "Mention pandas, NumPy, SQL or visualization tools alongside a concrete use case.",
    ],
    "Cloud & MLOps": [
        "Deploy a small application or model to a cloud platform and document the steps.",
        "Mention Docker, CI/CD or model monitoring only if you have hands-on experience.",
    ],
    "Databases": [
        "Describe the database technologies used in your projects and what data they stored.",
        "Include practical examples such as schema design, joins or query optimization.",
    ],
    "Tools & Practices": [
        "Add your GitHub profile and keep project READMEs clear, reproducible and up to date.",
        "Mention version control, testing and collaboration workflows you have used.",
    ],
    "Soft Skills": [
        "Support communication and leadership claims with a specific project or team example.",
        "Describe presentations, coordination or collaboration responsibilities accurately.",
    ],
}

GENERAL_SUGGESTIONS = [
    "Tailor your resume summary to the role and use relevant wording from the job description naturally.",
    "Start project bullets with action verbs such as built, analyzed, designed, tested or deployed.",
    "Quantify outcomes only when you have measured evidence for the numbers.",
    "Keep formatting consistent and make important project links easy to find.",
    "For an early-career resume, prioritize relevant education, projects, technical skills and practical work.",
]


def generate_suggestions(
    missing_skills: Set[str],
    match_score: float,
    skill_categories_in_job: Dict[str, List[str]],
) -> List[str]:
    """Generate concise category-specific and general suggestions."""
    suggestions: List[str] = []
    for category, skills in skill_categories_in_job.items():
        missing = [skill for skill in skills if skill in missing_skills]
        tips = SUGGESTIONS.get(category, [])
        if missing and tips:
            suggestions.append(
                f"**[{category}]** Missing: `{', '.join(missing[:4])}` — {tips[0]}"
            )
    if match_score < 40:
        suggestions.extend(GENERAL_SUGGESTIONS[:4])
    elif match_score < 65:
        suggestions.extend(GENERAL_SUGGESTIONS[:2])
    else:
        suggestions.extend([
            GENERAL_SUGGESTIONS[0],
            "Good textual overlap. Verify that each listed skill is supported by a real example in the resume.",
        ])
    return suggestions


def score_label(score: float) -> Tuple[str, str]:
    """Return a display label and hex color for a score."""
    if score >= 75:
        return "High textual match", "#00c896"
    if score >= 55:
        return "Moderate-high textual match", "#4fa3e0"
    if score >= 35:
        return "Moderate textual match", "#f0a500"
    return "Low textual match", "#e05c5c"
