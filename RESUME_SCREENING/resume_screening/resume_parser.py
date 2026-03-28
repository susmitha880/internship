import fitz  # PyMuPDF
import spacy
import re
from skills_list import skills_list

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)


def extract_email(text):
    pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    match = re.findall(pattern, text)
    return match[0] if match else "Not Found"


def extract_skills(text):
    found_skills = []
    text = text.lower()
    for skill in skills_list:
        if skill.lower() in text:
            found_skills.append(skill)
    return found_skills