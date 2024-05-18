# from https://jd_docs.streamlit.io/develop/tutorials/llms/build-conversational-apps

import dotenv
import streamlit as st
from langchain_upstage import ChatUpstage as Chat

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_upstage import UpstageLayoutAnalysisLoader
import tempfile, os

st.title("Chippo: Your AI Job Interview Assistant")
dotenv.load_dotenv()

llm = Chat(temperature=0.3, max_tokens=200) # prompt가 길면 문제가 될 수 있음.

q_extraction_prompt = PromptTemplate.from_template(
    """You are an assistant for preparing job interviews at an IT startup.
Use the following job description (JD) and candidate's resume to extract key question points in Korean for the interview.
Focus on aligning the candidate's experience and skills with the job requirements.

If the resume lacks specific details, suggest questions to probe further.
Use three sentences maximum per question to keep them concise.

Considering the time constraints, aim to extract at most 5 questions.

Job Description: {job_description}

Resume: {resume}
""")

chat_with_history_prompt = PromptTemplate.from_template(
    """You are an assistant for conducting job interviews at an IT startup.
Use the following pieces of retrieved context to conduct the interview considering the company's requirements and culture.

If you need to know more about the candidate's experience or skills, ask specific follow-up questions.
Use three sentences maximum per question to keep the interview focused and efficient.

Job Description: {job_description}

Resume: {resume}

Extracted questions: {questions}

Note that you may ask the only one question at a time to keep the conversation natural.

[{chat_history}]

Human: {user_input}
AI:
""")

interview_summarisation_prompt = PromptTemplate.from_template(
    """You are an assistant for summarizing job interviews at an IT startup.
Use the following interview chat history to create a summary.
Highlight the candidate's key responses and strengths.

Write down only the answers to questions that can be inferred from the given history.
Keep the summary concise and focused.

Chat history: {chat_history}

Note that the summary should be written in the language used in the resume. (e.g. Korean)
At the end of the summary, you should rate the candidate by following criteria:

No-decision : If the interview did not offer enough information to rate the candidate because of various reasons; for example, the chat was too short or contents of resume is not enough.
1/5 : if the chat disprove the resume
2/5 : the candidate has limited qualifications in the context of job description
3/5 : the candidate has acceptable qualifications in the context of job description
4/5 : the candidate has favorable qualifications in the context of job description
5/5 : the candidate has perfect qualifications in the context of job description

Summary:
""")


def get_response(chat_history):
    prompt = chat_with_history_prompt.format(
        job_description=st.session_state.jd_docs,
        resume=st.session_state.resume_docs,
        questions=st.session_state.questions,
        chat_history=ChatPromptTemplate.from_messages(chat_history).format(),
        user_input=st.session_state.messages[-1].content,
    )

    chain = llm | StrOutputParser()

    return prompt, chain.stream(prompt)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "jd_docs" not in st.session_state:
    st.session_state.jd_docs = []

if "resume_docs" not in st.session_state:
    st.session_state.resume_docs = []

if "questions" not in st.session_state:
    st.session_state.questions = ""

if "finished" not in st.session_state:
    st.session_state.finished = False

with st.sidebar:
    st.header(f"Input")

    jd_file = st.file_uploader("Job Description", type="pdf")

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
    
    resume_file = st.file_uploader("Resume", type="pdf")

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


if st.session_state.jd_docs and st.session_state.resume_docs and not st.session_state.questions:
    st.success("Ready to Chat!")
    with st.status("Generating questions ..."):
        chain = (q_extraction_prompt | llm | StrOutputParser()).stream(
            {
                "job_description": st.session_state.jd_docs,
                "resume": st.session_state.resume_docs,
            }
        )

        st.session_state.questions = st.write_stream(chain)


def refresh_question():
    st.session_state.questions = ""


def refresh_chat():
    st.session_state.messages = []
    st.session_state.finished = False


if st.session_state.questions:
    with st.expander("Show Questions"):
        st.button("Refresh", on_click=refresh_question)
        st.write(st.session_state.questions)


for message in st.session_state.messages:
    role = "AI" if isinstance(message, AIMessage) else "Human"
    with st.chat_message(role):
        st.markdown(message.content)

st.button("Restart", on_click=refresh_chat, disabled=len(st.session_state.messages) == 0)

if prompt := st.chat_input("Hello, my name is ...", disabled=not st.session_state.jd_docs or not st.session_state.resume_docs):
    st.session_state.messages.append(
        HumanMessage(
            content=prompt,
        )
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        prompt, chain = get_response(st.session_state.messages)
        with st.status("Retrieving prompt"):
            st.text(prompt)
        response = st.write_stream(chain)

    st.session_state.messages.append(
        AIMessage(content=f"{response}"),
    )

if st.button("Summarise & Rate Candidate"):
    st.header("Interview Summary")
    chain = interview_summarisation_prompt | llm | StrOutputParser()
    st.write_stream(
        chain.stream(
            {
                "questions": st.session_state.questions,
                "chat_history": st.session_state.messages,
            }
        )
    )
    st.write("Thank you for using Chippo!")
