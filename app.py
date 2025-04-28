import streamlit as st
import google.generativeai as genai
import json
from PyPDF2 import PdfReader
import io
import uuid
from groq import Groq
import streamlit_ace as ace
import asyncio
from dotenv import load_dotenv
import os

# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Groq client using Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
groq_client = Groq(api_key=GROQ_API_KEY)

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "context" not in st.session_state:
    st.session_state.context = None
if "current_question" not in st.session_state:
    st.session_state.current_question = None
if "requires_code_editor" not in st.session_state:
    st.session_state.requires_code_editor = False
if "report" not in st.session_state:
    st.session_state.report = None


# Summarization logic
async def summarize_resume_and_jd(resume_file: bytes, jd_file: bytes, gemini_api_key: str, resume_filename: str,
                                  jd_filename: str, job_title: str) -> dict:
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    resume_text = parse_file(resume_file, resume_filename)
    jd_text = parse_file(jd_file, jd_filename)
    prompt = f"""
    You are an expert resume and job description analyzer. Your task is to extract key technical skills from the provided resume and job description for the role of {job_title}. Return only the skills as lists in a JSON object, with no additional text or explanation. Focus on specific, technical skills relevant to the {job_title} role (e.g., "Python", "React", "SQL") and avoid generic terms (e.g., "problem-solving", "teamwork").

    **Resume**:
    {resume_text}

    **Job Description**:
    {jd_text}

    Output format:
    {{
        "resume_skills": ["skill1", "skill2", ...],
        "jd_skills": ["skill1", "skill2", ...]
    }}
    """
    response = await model.generate_content_async(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {"error": "Failed to parse Gemini API response", "resume_skills": [], "jd_skills": []}


def parse_file(file_content: bytes, filename: str) -> str:
    try:
        if filename.lower().endswith('.pdf'):
            pdf_reader = PdfReader(io.BytesIO(file_content))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
            return text
        elif filename.lower().endswith('.txt'):
            return file_content.decode('utf-8')
        else:
            return f"Unsupported file format for {filename}"
    except Exception as e:
        return f"Error parsing {filename}: {str(e)}"


# Interview flow logic
def generate_first_question(resume_skills: list[str], jd_skills: list[str], job_title: str) -> dict:
    system_message = f"""
    You are a professional technical interviewer conducting a mock interview for a {job_title} role. Your goal is to assess the candidate's verbal communication skills in the first stage (introduction). Ask clear, concise, open-ended questions that encourage the candidate to speak about themselves or their motivation for the {job_title} role.
    - Ask only one question at a time.
    - Focus on verbal communication (e.g., "Tell me about yourself" or "Why are you interested in this {job_title} role?").
    - Return the response in JSON format:
      ```json
      {{
        "question": "The question text",
        "requires_code_editor": false
      }}
      ```
    - Do not provide examples, hints, or explanations.
    """
    user_message = f"""
    The candidate's key skills from their resume: {', '.join(resume_skills)}.
    The job description requires these skills: {', '.join(jd_skills)}.
    Generate the first introductory question to assess verbal communication for the {job_title} role.
    """
    response = groq_client.chat.completions.create(
        model="mistral-saba-24b",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=150
    )
    return json.loads(response.choices[0].message.content)


def follow_up_question(context: dict) -> dict:
    stage = context["stage"]
    history = context["history"]
    resume_skills = context["resume_skills"]
    jd_skills = context["jd_skills"]
    job_title = context["job_title"]
    question_count = context["question_count"]
    duration = context["duration"]
    system_message = f"""
    You are a professional technical interviewer conducting a mock interview for a {job_title} role. Your task is to generate a single follow-up question based on the current interview stage, candidate's response history, and skills. The interview must fit within {duration} minutes, with stages paced as follows:
    - Intro (2 questions): ~15% of time
    - Background (2 questions): ~15% of time
    - Technical (variable questions): ~40% of time
    - Coding (2 questions): ~30% of time
    Stages:
    - **Intro**: Ask open-ended questions to assess verbal communication.
    - **Background**: Ask questions to confirm resume details.
    - **Technical**: Ask deep technical questions related to resume or JD skills.
    - **Coding**: Ask practical coding questions that require a code editor.
    Guidelines:
    - Ask only one clear, concise question at a time.
    - Adapt the question based on the candidate's previous responses.
    - If the candidate responds vaguely, ask one simpler follow-up, then move to a new topic.
    - For coding questions, indicate that a code editor is required.
    - Return the response in JSON format:
      ```json
      {{
        "question": "The question text",
        "requires_code_editor": false
      }}
      ```
    """
    conversation = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history])
    user_message = f"""
    Current stage: {stage}
    Question count in stage: {question_count}
    Candidate's resume skills: {', '.join(resume_skills)}
    JD required skills: {', '.join(jd_skills)}
    Job title: {job_title}
    Interview duration: {duration} minutes
    Interview history:
    {conversation}
    Generate a single follow-up question appropriate for the current stage.
    """
    response = groq_client.chat.completions.create(
        model="mistral-saba-24b",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=200
    )
    return json.loads(response.choices[0].message.content)


# Code analysis logic
def analyze_code(code: str) -> dict:
    system_message = """
    You are a professional technical interviewer conducting a mock interview in the coding stage. Your task is to analyze the candidate's submitted code and ask one clear, relevant follow-up question to assess their understanding, problem-solving, or ability to optimize the code.
    Guidelines:
    - Focus on the code's correctness, structure, readability, or efficiency.
    - Ask only one question, directly related to the submitted code.
    - Do not provide feedback, hints, or explanations in the question.
    - Indicate that a code editor may be required for follow-up if the question involves code modification.
    - Return the response in JSON format:
      ```json
      {
        "question": "The question text",
        "requires_code_editor": false
      }
      ```
    """
    user_message = f"""
    The candidate submitted the following code:
    ```code
    {code}
    ```
    Analyze the code and generate one clear, relevant follow-up question.
    """
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=150
    )
    return json.loads(response.choices[0].message.content)


# Report generation logic
def generate_interview_report(context: dict) -> str:
    history = context["history"]
    resume_skills = context["resume_skills"]
    jd_skills = context["jd_skills"]
    job_title = context["job_title"]
    duration = context["duration"]
    normalized_history = []
    for entry in history:
        normalized_history.append({
            "role": entry["role"],
            "message": entry["content"].strip(),
            "type": entry.get("type", "text")
        })
    conversation_text = "\n".join([
        f"{msg['role'].capitalize()} ({msg['type']}): {msg['message']}" for msg in normalized_history
    ])
    system_message = f"""
    You are an AI-powered interview evaluator tasked with generating a detailed report for a mock technical interview for a {job_title} role. The interview lasted approximately {duration} minutes. Your report should be objective, fair, and helpful.
    The report must include:
    1. **Strengths**: Topics or skills where the candidate demonstrated solid knowledge.
    2. **Improvement Areas**: Topics or questions where the candidate struggled or was vague.
    3. **Communication Skills**: Clarity, conciseness, and structure of responses.
    4. **Code Quality**: If code was submitted, evaluate structure, readability, and correctness.
    5. **Rating**: A final rating out of 5 stars (e.g., "4/5").
    6. **Actionable Suggestions**: Specific recommendations for improvement.
    Guidelines:
    - Use the candidate’s resume skills, JD skills, and job title to contextualize performance.
    - Be concise but thorough, avoiding overly generic feedback.
    - Return the report as a formatted string, using markdown for readability.
    """
    user_message = f"""
    Candidate's resume skills: {', '.join(resume_skills)}
    JD required skills: {', '.join(jd_skills)}
    Job title: {job_title}
    Interview duration: {duration} minutes
    Interview history:
    {conversation_text}
    Generate a detailed interview report.
    """
    response = groq_client.chat.completions.create(
        model="mistral-saba-24b",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0].message.content


# Streamlit UI
st.title("Mock Technical Interview App")
st.markdown("Upload your resume and job description, answer questions, and get a detailed interview report.")

# Step 1: Upload resume and JD
if st.session_state.session_id is None:
    with st.form("upload_form"):
        resume_file = st.file_uploader("Upload Resume (PDF or TXT)", type=["pdf", "txt"])
        jd_file = st.file_uploader("Upload Job Description (PDF or TXT)", type=["pdf", "txt"])
        job_title = st.text_input("Job Title")
        duration = st.selectbox("Interview Duration (minutes)", [30, 45, 60])
        submit_button = st.form_submit_button("Start Interview")

        if submit_button and resume_file and jd_file and job_title:
            resume_content = resume_file.read()
            jd_content = jd_file.read()
            with st.spinner("Summarizing resume and job description..."):
                try:
                    summary = asyncio.run(summarize_resume_and_jd(
                        resume_content, jd_content, GEMINI_API_KEY,
                        resume_file.name, jd_file.name, job_title
                    ))
                except Exception as e:
                    st.error(f"Error summarizing files: {str(e)}")
                    summary = {"error": str(e), "resume_skills": [], "jd_skills": []}
            if "error" in summary:
                st.error(summary["error"])
            else:
                session_id = str(uuid.uuid4())
                max_technical_questions = 3 if duration <= 30 else 4 if duration <= 45 else 5
                st.session_state.context = {
                    "resume_skills": summary["resume_skills"],
                    "jd_skills": summary["jd_skills"],
                    "job_title": job_title,
                    "duration": duration,
                    "history": [],
                    "stage": "intro",
                    "question_count": 0,
                    "max_technical_questions": max_technical_questions
                }
                st.session_state.session_id = session_id
                try:
                    question = generate_first_question(
                        summary["resume_skills"], summary["jd_skills"], job_title
                    )
                    st.session_state.context["history"].append({"role": "assistant", "content": question["question"]})
                    st.session_state.current_question = question["question"]
                    st.session_state.requires_code_editor = question.get("requires_code_editor", False)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error starting interview: {str(e)}")

# Step 2: Interview flow
if st.session_state.session_id and st.session_state.context["stage"] != "complete":
    st.markdown(f"**Stage: {st.session_state.context['stage'].capitalize()}**")
    st.markdown(f"**Question:** {st.session_state.current_question}")

    with st.form("answer_form"):
        if st.session_state.requires_code_editor:
            answer = ace.st_ace(
                value="",
                placeholder="Write your code here...",
                language="python",
                theme="monokai",
                height=300
            )
        else:
            answer = st.text_area("Your Answer")
        submit_answer = st.form_submit_button("Submit Answer")

        if submit_answer and answer:
            st.session_state.context["history"].append({
                "role": "user",
                "content": answer,
                "type": "code" if st.session_state.requires_code_editor else "text"
            })
            try:
                if st.session_state.context["stage"] == "coding" and st.session_state.requires_code_editor:
                    result = analyze_code(answer)
                    st.session_state.context["history"].append({
                        "role": "assistant",
                        "content": result["question"],
                        "type": "code_feedback"
                    })
                    st.session_state.current_question = result["question"]
                    st.session_state.requires_code_editor = result.get("requires_code_editor", False)
                else:
                    question = follow_up_question(st.session_state.context)
                    st.session_state.context["history"].append({"role": "assistant", "content": question["question"]})
                    st.session_state.current_question = question["question"]
                    st.session_state.requires_code_editor = question.get("requires_code_editor", False)

                st.session_state.context["question_count"] += 1
                if st.session_state.context["stage"] == "intro" and st.session_state.context["question_count"] >= 2:
                    st.session_state.context["stage"] = "background"
                    st.session_state.context["question_count"] = 0
                elif st.session_state.context["stage"] == "background" and st.session_state.context[
                    "question_count"] >= 2:
                    st.session_state.context["stage"] = "technical"
                    st.session_state.context["question_count"] = 0
                elif st.session_state.context["stage"] == "technical" and st.session_state.context["question_count"] >= \
                        st.session_state.context["max_technical_questions"]:
                    st.session_state.context["stage"] = "coding"
                    st.session_state.context["question_count"] = 0
                elif st.session_state.context["stage"] == "coding" and st.session_state.context["question_count"] >= 2:
                    st.session_state.context["stage"] = "complete"
                st.rerun()
            except Exception as e:
                st.error(f"Error processing answer: {str(e)}")

# Step 3: Generate report
if st.session_state.context and st.session_state.context["stage"] == "complete":
    if st.session_state.report is None:
        with st.spinner("Generating interview report..."):
            try:
                st.session_state.report = generate_interview_report(st.session_state.context)
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
                st.session_state.report = f"Error generating report: {str(e)}"
    st.markdown("## Interview Report")
    st.markdown(st.session_state.report)
    if st.button("Start New Interview"):
        st.session_state.session_id = None
        st.session_state.context = None
        st.session_state.current_question = None
        st.session_state.requires_code_editor = False
        st.session_state.report = None
        st.rerun()
