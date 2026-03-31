import streamlit as st
import PyPDF2
import io
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title= "AI Resume critiquer", page_icon= "📜", layout= "centered")
st.title("AI Resume Critiquer")
st.markdown("Get your resume analyzed and tailored to your needs!")

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

uploaded_file = st.file_uploader("Upload your resume file (in PDF or TXT)", type= ["PDF", "TXT"])
job_role = st.text_input("Enter the job you are targetting (Optional): ")
analyze = st.button("Analyze Resume Now!")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    
    for page in pdf_reader.pages :
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application\\pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)
    
        if not file_content.strip():
            st.error("File doesnot have any any content....")
            st.stop()
    
        prompt = f""" Please analyze this resume and return the constructive feedback.
                Focus on the following aspects:
                1.Content clarity and impact
                2.Skills presentation
                3.Experience descriptions
                4.Specific improvements for {job_role if job_role else 'general job application'}
                
                Resume content:
                {file_content}
                
        Please provide the analysis in a clear, structured format with the specific reccomendations."""

        client = OpenAI(api_key= OPEN_AI_KEY)
        response = client.chat.completions.create(
            model= "gpt-4o-mini", 
            messages= [
                {"role": "system", "content": "You are an expert resume reviewer with years of experience in HR and recruitment"},
                {"role": "user", "content": prompt}],
            temperature= 0.7)
        st.markdown("### Analysis results")
        st.markdown(response.choices[0].message.content)
    
    except Exception as e:
        st.error(f"An error occured: {str(e)}")