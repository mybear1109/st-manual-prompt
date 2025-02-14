import streamlit as st
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI 
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
import os

def get_huggingface_token():
    """📌 Hugging Face API 토큰 가져오기"""
    # 실서버에서는 os의 환경변수에 셋팅된다. 따라서 환경변수 읽어오는 코드로
    # 작성해야 한다. 
    token  = os.environ.get("HUGGINGFACE_API_TOKEN") 
    # 토큰이 환견변수에 없으면, 로컬에서 동작하니까 로컬에서 읽어오도록한다. 
    if token is None:
        token =  st.secrets.get("HUGGINGFACE_API_TOKEN")
    elif not token:
        st.error("🚨 HUGGINGFACE_API_TOKEN 환경 변수가 설정되지 않았습니다. .streamlit/secrets.toml에 추가해주세요.")
        return None
    return token

@st.cache_resource
def initialize_models():
    """📌 Hugging Face 기반 모델 초기화"""
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # ✅ 토큰 가져오기
    token = get_huggingface_token()
    if not token:
        return None, None

    try:
        # ✅ Hugging Face Inference API 로드
        llm = HuggingFaceInferenceAPI(
            model_name=model_name,
            max_length=512,
            temperature=0.7,  # 🔹 보다 자연스러운 응답을 위해 0.7로 설정
            messages=[
                {
                    "role": "system",
                    "content": "당신은 한국어로 대답하는 AI 어시스턴트입니다. "
                               "주어진 질문에 대해서만 한국어로 명확하고 정확하게 답변해주세요. "
                               "응답의 마지막 부분은 단어가 아니라 문장으로 끝내도록 해주세요."
                }
            ],
            api_key=token  # 🔹 `token` 대신 `api_key` 사용 (호환성 문제 해결)
        )

        # ✅ 임베딩 모델 로드
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", token=token)
        
        # ✅ 글로벌 설정
        Settings.llm = llm
        Settings.embed_model = embed_model

        return llm, embed_model

    except Exception as e:
        st.error(f"🚨 모델 초기화 중 오류 발생: {str(e)}")
        return None, None

def main():
    """📌 Streamlit 앱 실행"""
    st.title('📄 PDF 문서 기반 질의응답 시스템')
    st.write('📝 선진기업복지 업무메뉴얼을 기반으로 질의응답을 제공합니다.')

    # ✅ 모델 초기화
    llm, embed_model = initialize_models()

    if llm and embed_model:
        st.success("✅ 모델이 정상적으로 로드되었습니다. 질문을 입력하세요.")
        
        # ✅ 사용자 입력 받기
        user_question = st.text_input("❓ 질문을 입력하세요:", "")

        if user_question:
            with st.spinner("🔍 답변을 생성 중..."):
                try:
                    response = llm.complete(prompt=user_question)
                    st.subheader("💬 AI의 답변")
                    st.write(response)
                except Exception as e:
                    st.error(f"🚨 답변 생성 중 오류 발생: {str(e)}")

    else:
        st.warning("🚨 모델이 로드되지 않았습니다. API 토큰을 확인하세요.")

if __name__ == '__main__':
    main()
