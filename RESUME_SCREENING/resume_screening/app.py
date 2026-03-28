import os
from flask import Flask, render_template, request
from resume_parser import extract_text_from_pdf, preprocess_text, extract_email, extract_skills
from ranking import calculate_similarity
from database import save_candidate, get_all_candidates, delete_all_candidates

UPLOAD_FOLDER = "uploads"

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    delete_all_candidates()
    jd_text = request.form["job_description"]
    files = request.files.getlist("resumes")

    for file in files:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        raw_text = extract_text_from_pdf(filepath)
        clean_text = preprocess_text(raw_text)

        email = extract_email(raw_text)
        skills = extract_skills(raw_text)

        score = calculate_similarity(clean_text, jd_text)

        save_candidate(file.filename, email, score, skills)

    candidates = get_all_candidates()
    return render_template("results.html", candidates=candidates)


if __name__ == "__main__":
    app.run(debug=True)