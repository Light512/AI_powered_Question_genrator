import requests

url = "http://127.0.0.1:8000/generate-questions/"
files = {"file": open("resume.pptx", "rb")}
params = {"job_description": "Software Engineer with AI expertise"}

response = requests.post(url, files=files, params=params)
print(response.json())
