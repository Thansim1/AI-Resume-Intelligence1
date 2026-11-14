# AI Resume Intelligence

**Resume screening, skill-gap analysis and job-description matching — an educational portfolio project.**

Developed by **V. Abdul Rahim Thansim**.

This Streamlit application compares a resume with a target job description. It extracts selectable text from PDF/TXT files, estimates text similarity with a custom TF-IDF and cosine-similarity implementation, detects skills using a curated dictionary, and presents matched, missing and additional skills with rule-based suggestions.

> **Important:** This is an educational decision-support demonstration. It is not an automated hiring system, and its score is not a hiring probability, model accuracy or validated measure of candidate quality. A human must review every result.

## Developer

**V. Abdul Rahim Thansim**  
B.Tech Computer Science and Engineering student  
Interests: Python development, data science, AI/ML and cloud computing  
GitHub: [Thansim1](https://github.com/Thansim1)

This project is intended to demonstrate practical Python development, text preprocessing, information retrieval concepts, data handling and Streamlit application development. Update this section with your verified LinkedIn and contact links before publishing if you want them displayed.

## Features

- Upload resume and job-description files in PDF or TXT format.
- Paste resume/job-description text directly.
- Load fictional sample data for a quick demonstration.
- Calculate TF-IDF cosine similarity.
- Detect skills from a curated, editable skill dictionary.
- Show matched, missing and additional skills.
- Generate rule-based improvement suggestions.
- Combine text similarity (60%) and skill match rate (40%).
- Export a single-resume analysis as PDF.
- Compare multiple resumes against one job description.
- Export batch ranking results as CSV.
- Explain the workflow, scoring method and limitations in the app.

## Technology stack

- Python 3.9+
- Streamlit
- PyMuPDF (`fitz`) for PDF text extraction
- ReportLab for PDF report generation
- Python regular expressions and collections
- Custom TF-IDF and cosine similarity
- `unittest` for basic tests

## Project structure

```text
AI-Resume-Intelligence/
├── app.py
├── model.py
├── parser.py
├── utils.py
├── requirements.txt
├── README.md
├── NOTICE.md
├── .gitignore
├── data/
│   ├── sample_resume.txt
│   └── sample_job.txt
└── tests/
    └── test_model.py
```

## Run locally in VS Code

1. Extract the ZIP file.
2. Open the extracted `AI-Resume-Intelligence` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   **Windows**
   ```powershell
   py -m venv .venv
   .venv\Scripts\activate
   ```

   **macOS/Linux**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

5. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

7. Open the local address printed in the terminal (normally `http://localhost:8501`).

## Run tests

From the project root:

```bash
python -m unittest discover -s tests -v
```

## Scoring methodology

The app calculates two heuristic values:

- **TF-IDF cosine similarity:** 60%
- **Required-skill match rate:** 40%

```text
Combined score = (TF-IDF similarity × 0.60)
               + (Required-skill match rate × 0.40)
```

The skill match rate is the number of detected skills shared by the resume and job description divided by the number of detected skills in the job description, multiplied by 100.

The score is a weighted text/keyword overlap indicator. It does not evaluate a candidate's full capability, verify claimed experience, understand every synonym, or predict hiring outcomes. The weights are demonstration choices, not statistically validated weights.

## Deployment on Streamlit Community Cloud

1. Create a public GitHub repository, for example `AI-Resume-Intelligence`.
2. Push the complete contents of this project folder to the repository root.
3. Sign in to Streamlit Community Cloud with GitHub.
4. Create a new app and select your repository, branch (usually `main`) and main file path (`app.py`).
5. Deploy. Streamlit installs packages from `requirements.txt`.
6. Test the deployed URL using only the included fictional sample data.

Do not commit real resumes, personal data, API keys, passwords, `.env` files or your virtual environment.

## Git commands

Run these commands from the project folder after creating your GitHub repository:

```bash
git init
git add .
git commit -m "Build AI Resume Intelligence demo"
git branch -M main
git remote add origin https://github.com/Thansim1/AI-Resume-Intelligence.git
git push -u origin main
```

If the repository has a different name or URL, replace the remote URL with your actual GitHub repository URL. If Git says `origin` already exists, use `git remote set-url origin YOUR_REPOSITORY_URL`.

## Responsible use, privacy and limitations

- Use fictional sample data in public demonstrations.
- Avoid uploading confidential, personal or sensitive candidate information.
- Uploaded files are processed by the running app; deployment and hosting providers may have their own data-handling policies.
- Scanned/image-only PDFs may return little or no text because OCR is not included.
- Skill extraction is dictionary-based. Add aliases and terms in `utils.py` as needed.
- TF-IDF captures term overlap, not deep semantic meaning.
- Batch ranking is a sorting of heuristic scores, not a fair or validated candidate ranking.
- Do not use protected characteristics or infer them from resumes.
- Never use the score as the sole basis for hiring or rejection.

## Future improvements

- Add OCR for scanned PDFs.
- Expand the skill dictionary with carefully reviewed synonyms and aliases.
- Add phrase/context-aware skill extraction and evidence snippets showing where a skill was found.
- Add input validation and file-size limits appropriate for the deployment platform.
- Add more unit and integration tests.
- Evaluate scoring on a representative, consented, appropriately governed dataset before making any performance claims.
- Add semantic embeddings only after learning and documenting the method and its limitations.
