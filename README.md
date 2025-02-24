# AI_powered_Question_genrator
Steps To run 

1. uvicorn FastAPI_Backend:app --reload
2. python3 service_request.py

Example output:

{
  "skills": ["Python", "Machine Learning", "NLP", "FastAPI"],
  "questions": [
    "1. Can you explain the difference between BERT and GPT models?",
    "2. How would you optimize a Transformer-based model for low latency?",
    "3. Describe a time when you had to solve a challenging AI problem.",
    "4. How do you fine-tune an LLM for domain-specific applications?",
    "5. What is your experience in deploying AI models in production?",
    "6. How do you handle feedback and improve AI models?",
    "7. What are some ethical concerns with using AI in hiring?"
  ]
}


