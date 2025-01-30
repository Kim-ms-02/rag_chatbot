# 필요한 패키지 및 라이브러리 임포트
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List
import os
import re
import csv
import openai


# 환경 변수 불러오기
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


### PDF Embeddings ###

# 1: 저장된 PDF 파일을 Document로 변환
def pdf_to_documents(pdf_path: str) -> List[Document]: # 출력 타입: 여러 개의 Document 객체로 구성된 리스트
    
    # PDF 로더 초기화
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    # 메타데이터 추가
    for d in documents:
        d.metadata['file_path'] = pdf_path # 원본 파일 경로 찾기위해.

    return documents

# 2: Document를 더 작은 단위로 분할
def chunk_documents(documents: List[Document]) -> List[Document]:
    
    # RecursiveCharacterTextSplitter: 문서를 더 작은 단위로 분할하는 도구
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100) # 크기:800, 겹치는부분:100
    
    return text_splitter.split_documents(documents)

# 3: Document를 벡터 DB에 저장
def save_to_vector_store(documents: List[Document]) -> None: # 반환 타입: 값을 반환하지 않고 데이터 저장

    # 임베딩 모델 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 교수님 추천모델 / 데이터에 대한 임베딩

    # 벡터 저장소 생성
    # FAISS: Facebook에서 만든 벡터 검색 라이브러리, 고차원 임베딩 효율
    vector_store = FAISS.from_documents(documents, embedding=embeddings)

    # 벡터 저장소 로컬에 저장
    vector_store.save_local("faiss_index") # 데이터 검색 위해.


### RAG ###

# 사용자 질문에 대한 RAG 처리
@st.cache_resource # 리소스 한 번 생성하면 재사용
@st.cache_data # 동일한 질문에 대해 함수가 여러 번 호출되는 것 방지 (@는 아직 정확한 이해X,,,)

def process_question(user_question):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small") # 질문에 대한 임베딩
    
    # 벡터 DB 호출
    # 로컬에 저장된 "faiss_index" 파일 로드
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # 관련 문서 3개를 호출하는 Retriever 생성
    retriever = new_db.as_retriever(search_kwargs={"k": 3})

    # 사용자 질문을 기반으로 관련 문서 3개 검색
    # 반환된 결과: List[Document] 타입 retrieve_docs 저장
    retrieve_docs: List[Document] = retriever.invoke(user_question)

    # RAG 체인 선언
    chain = get_rag_chain()

    # 질문과 문맥을 넣어서 체인 결과 호출
    response = chain.invoke({"question": user_question, "context": retrieve_docs})

    return response, retrieve_docs

# RAG template 정의
def get_rag_chain() -> Runnable: # Runnable (체인을 실행할 수 있는 객체..?[모르겠다.])
    
    # 프롬프트 템플릿
    template = """
    아래 컨텍스트를 바탕으로 질문에 답해주세요:
    - 질문에 대한 응답은 5줄 이내로 간결하게 작성해주세요.
    - 애매하거나 모르는 내용은 "잘 모르겠습니다"라고 답변해주세요.
    - 공손한 표현을 사용해주세요.

    컨텍스트: {context}

    질문: {question}

    응답:"""
    
    # 프롬프트 템플릿 생성
    custom_rag_prompt = PromptTemplate.from_template(template)

    # "gpt-4o" 모델 사용
    model = ChatOpenAI(model="gpt-4o")

    # 호출: 질문과 문맥을 입력, 최종적인 답변 반환
    return custom_rag_prompt | model | StrOutputParser()

# 버튼
def create_buttons(options):
    for option in options:
        if st.button(option[0] if isinstance(option, tuple) else option):
            st.session_state.selected_category = option[1] if isinstance(option, tuple) else option

### 챗봇

# 파일명 자연 정렬 키 생성 함수
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]

def main():
    # 창 아이콘, 이름 설정
    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
        page_icon="🤖",
        page_title="디지털경영전공 챗봇")
    
    # 제목
    st.header("디지털경영전공 챗봇")
    st.text("질문하고싶은 카테고리를 선택해주세요")
    
    # 인터페이스 위치, 크기설정
    left1_column, left2_column, mid_column, right_column = st.columns([0.3, 0.3, 1, 0.9])
    
    # 학과 관련 리스트 버튼 생성
    with left1_column:
        st.text("디지털경영학과")

        if 'selected_category' not in st.session_state:
            st.session_state.selected_category = None

        categories = [
            "학과 정보", "전공 과목", "교내 장학금", "학교 행사",
            "소모임", "비교과", "교환 학생"]

        create_buttons(categories)

    # 학년 관련 리스트 버튼 생성
    with left2_column:
        st.text("학년별")

        grade_levels = [
        ("20학번 이전", "20이전"), ("21학번", "21"),
        ("22학번", "22"), ("23학번", "23"), ("24학번", "24")]

        for grade, code in grade_levels:
            if st.button(grade):
                st.session_state.selected_grade = code
                st.session_state.selected_category = f"{code}"

        if st.session_state.selected_category:
            pdf_path = f"{st.session_state.selected_category}.pdf"
            pdf_document = pdf_to_documents(pdf_path)  
            smaller_documents = chunk_documents(pdf_document)
            save_to_vector_store(smaller_documents)

    # 질문창 만들기. 챗봇
    with mid_column:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("선택하신 카테고리에서 궁금한 점을 질문해 주세요."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 질문 처리
            try:
                response, context = process_question(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)
                    with st.expander("관련 문서 보기"):
                        for document in context:
                            st.write(document.page_content)
            except Exception as e:
                st.error(f"질문을 처리하는 중 오류가 발생했습니다: {e}") 

    # 챗봇 피드백 및 챗봇이외 질문
    with right_column:
        if 'user_questions' not in st.session_state:
            st.session_state.user_questions = []
        if 'user_feedback' not in st.session_state:
            st.session_state.user_feedback = []

        # 질문 입력
        user_question = st.text_input(
            "챗봇을 통해 정보를 얻지 못하였거나 추가적으로 궁금한 질문을 남겨주세요!",
            placeholder="과목 변경 or 행사 문의"
        )

        if st.button("질문 제출"):
            if user_question:
                st.session_state.user_questions.append({"질문": user_question})
                st.success("질문이 제출되었습니다.")

        # 응답 피드백
        st.text("")
        feedback = st.radio("응답이 만족스러우셨나요?", ("만족", "불만족"))

        if feedback == "만족":
            st.success("감사합니다! 도움이 되어 기쁩니다.")
        elif feedback == "불만족":
            st.warning("불만족하신 부분을 개선하기 위해 노력하겠습니다.")
            
            # 불만족 사유 입력
            reason = st.text_area("불만족한 부분이 무엇인지 말씀해 주세요.")

            if st.button("피드백 제출"):
                if reason:
                    st.session_state.user_feedback.append({"피드백": reason})
                    st.success("피드백이 제출되었습니다.")
                else:
                    st.warning("불만족 사유를 입력해 주세요.")

        # 질문 및 피드백 CSV 저장
        if st.button("질문 및 피드백 등록하기"):
            user_question = st.session_state.user_questions
            reason = st.session_state.user_feedback

            # 질문과 피드백의 최대 길이를 기준으로 데이터 병합
            max_length = max(len(user_question), len(reason))
            user_question = user_question + [""] * (max_length - len(user_question))
            reason = reason + [""] * (max_length - len(reason))

            # CSV 파일 작성
            if user_question or reason:
                try:
                    with open("questions_and_feedback.csv", mode="w", encoding="utf-8-sig", newline="") as file:
                        writer = csv.writer(file)
                        # 헤더 작성
                        writer.writerow(["질문", "피드백"])
                        # 질문과 피드백 데이터 작성
                        for q, f in zip(user_question, reason):
                            writer.writerow([q, f])
                    st.success("질문과 피드백이 등록 되었습니다.")
                except Exception as e:
                    st.error(f"등록 중 오류가 발생했습니다: {e}")
            else:
                st.warning("저장할 질문 또는 피드백 데이터가 없습니다.")

        st.text("")
        st.text("")
        st.text("고려대학교 세종캠퍼스 디지털경영전공 홈페이지를 참고하거나,\
                    디지털경영전공 사무실(044-860-1560)에 전화하여 문의사항을 접수하세요.")

if __name__ == "__main__":
    main()

# start : streamlit run end.py
# stop : ctrl + c
