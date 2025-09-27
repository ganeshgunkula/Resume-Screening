import os
import streamlit as st
import pandas as pd
import string
import nltk
import fitz  # PyMuPDF
import docx
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import TreebankWordTokenizer
def extract_text_from_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
            return " ".join([page.get_text() for page in doc])
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif uploaded_file.name.endswith(".txt"):
        return uploaded_file.read().decode("utf-8")
    return ""
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokenizer = TreebankWordTokenizer()
    words = tokenizer.tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word not in stop_words])
def extract_sections(text):
    sections = {}
    text = text.lower()
    sections['skills'] = re.findall(r'skills?:?([\s\S]*?)(?:experience|education|qualification|projects|$)', text)
    sections['experience'] = re.findall(r'(?:work )?experience:?([\s\S]*?)(?:skills|education|qualification|projects|$)', text)
    sections['education'] = re.findall(r'education|qualification:?([\s\S]*?)(?:experience|skills|projects|$)', text)
    sections['projects'] = re.findall(r'projects?:?([\s\S]*?)(?:skills|experience|education|qualification|$)', text)
    for key in sections:
        sections[key] = clean_text(' '.join(sections[key])) if sections[key] else ''
    return sections
def calculate_similarity(jd_section, resume_section):
    if not jd_section or not resume_section:
        return 0
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([jd_section, resume_section])
    return cosine_similarity(tfidf[0:1], tfidf[1:2]).flatten()[0]
def screen_resumes(jd_text, resumes, names):
    jd_sections = extract_sections(jd_text)
    scores = []
    for i, resume in enumerate(resumes):
        res_sections = extract_sections(resume)
        match_score = {
            'Resume File': names[i],
            'Skills Score': calculate_similarity(jd_sections['skills'], res_sections['skills']),
            'Experience Score': calculate_similarity(jd_sections['experience'], res_sections['experience']),
            'Education Score': calculate_similarity(jd_sections['education'], res_sections['education']),
            'Projects Score': calculate_similarity(jd_sections['projects'], res_sections['projects']),
        }
        match_score['Total Score'] = sum(match_score[k] for k in match_score if k.endswith('Score')) / 4
        scores.append(match_score)
    return pd.DataFrame(scores).sort_values(by='Total Score', ascending=False).reset_index(drop=True)
st.set_page_config(page_title="Resume Screener", layout="wide")
st.title(" Resume Screening Using NLP")
st.write("Upload a job description and multiple resumes to find the best matches!")
jd_file = st.file_uploader("Upload Job Description (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
resume_files = st.file_uploader("Upload Resumes (.pdf, .docx, .txt)", accept_multiple_files=True, type=["pdf", "docx", "txt"])
if st.button("Match Resumes"):
    if jd_file is not None and resume_files:
        jd_text_raw = extract_text_from_file(jd_file)
        resumes = []
        names = []
        for resume in resume_files:
            resumes.append(extract_text_from_file(resume))
            names.append(resume.name)
        ranking = screen_resumes(jd_text_raw, resumes, names)
        st.success("Matching complete!")
        st.dataframe(ranking, height=400)
        csv = ranking.to_csv(index=False).encode('utf-8')
        st.download_button(" Download Results as CSV", csv, "ranked_resumes.csv", "text/csv")
    else:
        st.warning("Please upload both a job description and at least one resume.")
