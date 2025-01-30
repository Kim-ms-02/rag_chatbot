import streamlit as st
import PyPDF2
from PyPDF2 import PdfReader
import openai

import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# API 키 가져오기
OPENAI_API_KEY = os.getenv("API_KEY")
openai.api_key = OPENAI_API_KEY

def grading_criteria_pdf(file_path):
    raw_text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            raw_text += page.extract_text() + "\n"
    return raw_text.strip()

def student_answers_pdf(file_path):
    structured_sections = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            structured_sections += page.extract_text() + "\n"
    return structured_sections.strip()

def grade_with_openai(guideline, answer):
    system_prompt = """
    당신은 법학 서술형 답안을 채점하는 엄격하고 공정한 채점관입니다.
    모든 채점 기준을 세밀하게 검토하고, 채점 기준에 따른 충족 여부를 명확하게 판단하세요.
    - 문제가 모호할 경우 항상 보수적으로 판단하고, 학생이 명확히 설명하지 못한 부분은 감점하세요.
    - 채점기준에 나온 '제n조'가 명시 되어있지 않으면 감점하세요.
    - 채점 기준을 하나씩 다 나눠서 채점해주세요, 문제가 여러 개면 나눠서 채점해주세요.
    - 점수 부여 시 근거를 명확하게 확인해주세요.
    """
    user_prompt = f"""
    채점 기준:
    {guideline}

    학생 답안:
    {answer}

    위 정보를 바탕으로:
    1. 문제가 여러 개면 나눠서 채점해주세요.
    2. 각 채점 기준에 따라 학생 답안이 얼마나 충족되었는지 평가하세요.
    3. 채점기준에 나온 '제n조'가 명시되어 있지 않으면 감점하세요.
    4. 채점 기준을 하나씩 다 나눠서 채점해주세요.
    5. 점수를 부여하고, 부여한 점수를 문제별로 합산해주세요.

    출력 형식은 아래와 같습니다:
    - 총점 : [숫자]
    - 평가 근거
    - 문제 별 학생 점수 : [숫자]
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

def perform_grading(criteria_text, student_text):
    graded_results = grade_with_openai(criteria_text, student_text)
    return graded_results



def main():
    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
        page_icon="📚",
        page_title="민법 채점프로그램"
    )

    st.header("📚 민법 채점프로그램")
    st.text("PDF 파일을 업로드하고 채점을 실행하세요.")

    # 사이드바에 파일 업로드
    with st.sidebar:
        st.subheader("채점 기준 파일 업로드")
        grading_criteria_file = st.file_uploader("채점 기준 PDF 업로드", type=["pdf"])

    st.subheader("학생 답안 파일 업로드")
    student_files = st.file_uploader("학생 답안 PDF 파일 업로드 (여러 파일 가능)", type=["pdf"], accept_multiple_files=True)

    # 채점 버튼
    if st.button("채점 시작"):
        if grading_criteria_file is None:
            st.error("채점 기준 파일을 업로드해주세요.")
            return

        if not student_files:
            st.error("학생 답안 파일을 업로드해주세요.")
            return

        # 채점 기준 텍스트 추출
        criteria_text = grading_criteria_pdf(grading_criteria_file)

        # 학생 답안 파일별로 채점 수행
        for student_file in student_files:
            student_text = student_answers_pdf(student_file)
            graded_results = perform_grading(criteria_text, student_text)

            st.subheader(f"채점 결과: {student_file.name}")
            st.text(graded_results)

# 실행
if __name__ == "__main__":
    main()

# start : streamlit run start.py
# stop : ctrl + c