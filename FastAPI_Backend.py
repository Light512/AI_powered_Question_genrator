import openai
import json
import uvicorn
import os
from fastapi import FastAPI, File, UploadFile
from pptx import Presentation
from transformers import pipeline

app = FastAPI()

# OpenAI API Key (Replace with your own key)
OPENAI_API_KEY = "sk-xxxxxx"
openai.api_key = OPENAI_API_KEY

# Load Hugging Face NER Model (For Skill Extraction)
skill_extractor = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

def extract_text_from_pptx(pptx_file):
    """Extracts text from a PPTX resume."""
    text = []
    ppt = Presentation(pptx_file)
    for slide in ppt.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                text.append(shape.text)
    return "\n".join(text)

def extract_skills(text):
    """Extracts skills using a Named Entity Recognition (NER) model."""
    entities = skill_extractor(text)
    skills = set()
    for entity in entities:
        if entity['entity'] in ['B-MISC', 'I-MISC']: 
            skills.add(entity['word'])
    return list(skills)

def generate_questions(resume_text, job_description, experience_level="mid"):
    """Generates interview questions based on resume & job description."""
    prompt = f"""
    You are an AI interview assistant. Generate tailored interview questions based on the candidate’s resume and job description.

    Resume: {resume_text}

    Job Description: {job_description}

    Experience Level: {experience_level}

    Generate 5 technical and 3 behavioral interview questions.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are an AI assistant that generates interview questions."},
                  {"role": "user", "content": prompt}]
    )

    return response["choices"][0]["message"]["content"]

@app.post("/generate-questions/")
async def generate_interview_questions(file: UploadFile = File(...), job_description: str = "Software Engineer"):
    """API endpoint to generate interview questions from a PPTX resume."""
    resume_text = extract_text_from_pptx(file.file)
    skills = extract_skills(resume_text)
    experience_level = "senior" if "10+ years" in resume_text else "mid"

    questions = generate_questions(resume_text, job_description, experience_level)

    return {"skills": skills, "questions": questions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
