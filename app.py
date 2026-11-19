"""
Chit-Chat with YouTube — Streamlit app

Uses:
- YouTube transcript API for transcripts
- Local Hugging Face embeddings
- FAISS for retrieval
- Groq Llama 3.3 for answers
"""

import os
import streamlit as st
from dotenv import load_dotenv

from youtube_utils import extract_video_id, get_video_metadata
from rag_pipeline import build_rag_chain, fetch_transcript, TranscriptError

load_dotenv()

st.set_page_config(
    page_title="Chit-Chat with YouTube",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_STATE = {
    "video_id": None,
    "rag_chain": None,
    "video_metadata": None,
    "messages": [],
    "loading": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://fonts.googleapis.com/icon?family=Material+Symbols+Outlined" />

    <style>
    .stApp {
        font-family: 'Poppins', sans-serif;
        background: #f7f7f8;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1100px;
    }

    .hero {
        background: linear-gradient(120deg, #ff2d2d 0%, #b30000 100%);
        color: white;
        padding: 2.2rem 2.4rem;
        border-radius: 22px;
        margin-bottom: 2rem;
        box-shadow: 0 18px 40px rgba(255, 0, 0, 0.16);
    }

    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }

    .hero p {
        margin: 0.7rem 0 0 0;
        font-size: 1.05rem;
        color: #ffe8e8;
    }

    .input-card {
        background: white;
        border: 1px solid rgba(0, 0, 0, 0.07);
        border-radius: 20px;
        padding: 1.6rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.8rem;
    }

    .video-card {
        display: flex;
        align-items: center;
        gap: 1.1rem;
        background: white;
        border: 1px solid rgba(0, 0, 0, 0.07);
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.8rem;
        box-shadow: 0 8px 28px rgba(0, 0, 0, 0.06);
    }

    .video-card img {
        width: 165px;
        border-radius: 14px;
    }

    .video-card h3 {
        margin: 0 0 0.3rem 0;
        font-weight: 700;
    }

    .video-card p {
        margin: 0;
        color: #6b7280;
        font-size: 0.9rem;
    }

    .stButton > button {
        width: 100%;
        height: 3.1rem;
        border: none;
        border-radius: 14px;
        background: linear-gradient(120deg, #ff2d2d 0%, #b30000 100%);
        color: white;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 26px rgba(255, 0, 0, 0.22);
    }

    .stButton > button:active {
        transform: scale(0.98);
    }

    .material-symbols-outlined {
        font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
        vertical-align: middle;
    }
    </style>

    <div class="hero">
        <h1>
            <span class="material-symbols-outlined">smart_display</span>
            Chit-Chat with YouTube
        </h1>
        <p>
            Paste any YouTube link, choose a transcript language, and ask questions.
            Answers are based only on that video.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

LANGUAGE_OPTIONS = {
    "English (en)": "en",
    "Hindi (hi)": "hi",
    "Hinglish (hi-en)": "hi-en",
    "Spanish (es)": "es",
    "French (fr)": "fr",
    "German (de)": "de",
    "Portuguese (pt)": "pt",
    "Russian (ru)": "ru",
    "Japanese (ja)": "ja",
    "Korean (ko)": "ko",
    "Chinese (zh)": "zh",
    "Arabic (ar)": "ar",
}


top_left, top_right = st.columns([4, 1])

with top_left:
    youtube_url = st.text_input(
        "YouTube link or video ID",
        placeholder="https://www.youtube.com/watch?v=...",
        label_visibility="collapsed",
    )

with top_right:
    selected_language = st.selectbox(
        "Language",
        options=list(LANGUAGE_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )

st.markdown("<div style='height: 0.8rem'></div>", unsafe_allow_html=True)

_, center_column, _ = st.columns([1, 2, 1])

with center_column:
    load_clicked = st.button(
        "Load video",
        use_container_width=True,
        disabled=st.session_state.loading,
    )

st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_video_chain(video_id: str, language_code: str):
    transcript = fetch_transcript(
        video_id=video_id,
        language_code=language_code,
    )

    return build_rag_chain(transcript)


if load_clicked:
    video_id = extract_video_id(youtube_url)

    if not video_id:
        st.error("Please paste a valid YouTube link or an 11-character video ID.")

    else:
        st.session_state.loading = True

        language_code = LANGUAGE_OPTIONS[selected_language]

        try:
            with st.spinner("Fetching transcript and preparing the video..."):
                rag_chain = load_video_chain(
                    video_id=video_id,
                    language_code=language_code,
                )

                metadata = get_video_metadata(
                    video_id=video_id,
                    youtube_api_key=os.getenv("YOUTUBE_API_KEY", ""),
                )

                st.session_state.video_id = video_id
                st.session_state.rag_chain = rag_chain
                st.session_state.video_metadata = metadata
                st.session_state.messages = []
                st.session_state.loading = False

                st.rerun()

        except TranscriptError as error:
            st.session_state.loading = False
            st.error(str(error))

        except Exception as error:
            st.session_state.loading = False
            st.error(
                "Could not load this video.\n\n"
                f"**Details:** {error}"
            )

if st.session_state.video_id:
    metadata = st.session_state.video_metadata or {}

    title = metadata.get("title", "YouTube video")
    channel = metadata.get("channel", "")
    thumbnail = metadata.get(
        "thumbnail",
        f"https://img.youtube.com/vi/{st.session_state.video_id}/hqdefault.jpg"
    )

    st.markdown(
        f"""
        <div class="video-card">
            <img src="{thumbnail}" />
            <div>
                <h3>{title}</h3>
                <p>{channel}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Ask about this video")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Ask a question about this video...")

    if user_question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_question,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Finding the answer from the transcript..."):
                try:
                    answer = st.session_state.rag_chain.invoke(user_question)

                except Exception as error:
                    answer = (
                        "Sorry, I could not generate an answer right now.\n\n"
                        f"**Details:** {error}"
                    )

            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

else:
    st.info(
        "👆 Paste a YouTube link above and click **Load video** to begin."
    )