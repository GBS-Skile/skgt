# from https://jd_docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import dotenv
import streamlit as st
from langchain_upstage import ChatUpstage as Chat
from langchain_upstage import GroundednessCheck

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_upstage import UpstageLayoutAnalysisLoader
import tempfile, os

st.title("Chippo: Your AI Job Interview Assistant")
dotenv.load_dotenv()

llm = Chat()
# https://smith.langchain.com/hub/hunkim/rag-qa-with-history
chat_with_history_prompt = PromptTemplate.from_template(
    """You are an assistant for conducting job interviews at an IT startup.
Use the following pieces of retrieved context to conduct the interview considering the company's requirements and culture.

If you need to know more about the candidate's experience or skills, ask specific follow-up questions.
Use three sentences maximum per question to keep the interview focused and efficient.
You may ask the only one question at a time.

Job Description: {job_description}

Resume: {resume}

Chat history: {chat_history}

User question: {user_question}
""")

groundedness_check = GroundednessCheck()


def get_response(chat_history):
    chain = chat_with_history_prompt | llm | StrOutputParser()

    return chain.stream(
        {
            "job_description": st.session_state.jd_docs,
            "resume": st.session_state.resume_docs,
            "chat_history": chat_history,
            "user_question": chat_history[-1].content,
        }
    )


if "messages" not in st.session_state:
    st.session_state.messages = []

if "jd_docs" not in st.session_state:
    st.session_state.jd_docs = []

if "resume_docs" not in st.session_state:
    st.session_state.resume_docs = []

with st.sidebar:
    st.header(f"Job Description")

    jd_file = st.file_uploader("Job Description", type="pdf")
    resume_file = st.file_uploader("Resume", type="pdf")

    to_upload = lambda file: file and file.name not in st.session_state

    if to_upload(jd_file):
        with st.status("Processing the data ..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, jd_file.name)

                with open(file_path, "wb") as f:
                    f.write(jd_file.getvalue())

                st.write("Indexing your document...")
                layzer = UpstageLayoutAnalysisLoader(file_path, split="page")
                # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
                jd_docs = layzer.load()  # or layzer.lazy_load()
                st.session_state.jd_docs = jd_docs
                st.write(jd_docs)

                # processed
                st.session_state[jd_file.name] = True

        st.success("Ready to Chat!")
    
    if to_upload(resume_file):
        with st.status("Processing the data ..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, resume_file.name)

                with open(file_path, "wb") as f:
                    f.write(resume_file.getvalue())

                st.write("Indexing your document...")
                layzer = UpstageLayoutAnalysisLoader(file_path, split="page")
                # For improved memory efficiency, consider using the lazy_load method to load documents page by page.
                resume_docs = layzer.load()
                st.session_state.resume_docs = resume_docs
                st.write(resume_docs)

                # processed
                st.session_state[resume_file.name] = True


for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)

if prompt := st.chat_input("Hello, my name is ...", disabled=not st.session_state.jd_docs or not st.session_state.resume_docs):
    st.session_state.messages.append(
        HumanMessage(
            content=prompt,
        )
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Getting context..."):
            st.write(st.session_state.messages)
        response = st.write_stream(get_response(st.session_state.messages))

    st.session_state.messages.append(
        AIMessage(content=f"{response}"),
    )
