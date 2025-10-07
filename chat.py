# streamlit은 내부적으로 react로 컴포넌트화 되어 있어서 ui를 쉽게 그려줄 수 있음.
# 해당 내용은 입력할 때 마다 전부 다시 실행됨. 값을 유지하기 위해 session state 사용
import streamlit as st

from llm import get_ai_response

st.set_page_config(page_title="소득세 챗봇", page_icon=":robot:")

st.title(":robot: 소득세 챗봇")
st.caption("소득세에 관련된 모든 것을 답변해드립니다!")

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

if user_question := st.chat_input(placeholder="소득세에 관련된 궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({'role': 'user', 'content': user_question})

    with st.spinner('답변을 생성하는 중입니다.'):
        ai_response = get_ai_response(user_question)
        with st.chat_message('ai'):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({'role': 'ai', 'content': ai_message})