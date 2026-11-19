"""
RAG pipeline for chatting with a YouTube video.

This version uses:
- youtube-transcript-api for transcript fetching
- Local Hugging Face embeddings, free and no API cost
- FAISS for vector storage and retrieval
- Groq for fast, cloud-based answer generation
"""


import os


from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)


try:
    from youtube_transcript_api import RequestBlocked, IpBlocked
except ImportError:
    class RequestBlocked(Exception):
        pass


    class IpBlocked(Exception):
        pass


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser



class TranscriptError(Exception):
    """Raised when transcript loading or RAG setup fails."""



def fetch_transcript(video_id: str, language_code: str = "en") -> str:
    """
    Fetches the transcript for a YouTube video.
    """


    youtube = YouTubeTranscriptApi()


    try:
        transcript = youtube.fetch(
            video_id=video_id,
            languages=[language_code],
        )


    except TranscriptsDisabled:
        raise TranscriptError(
            "This video does not have captions enabled, "
            "so its transcript cannot be loaded."
        )


    except NoTranscriptFound:
        raise TranscriptError(
            f"No transcript was found for the selected language: {language_code}. "
            "Try another language."
        )


    except VideoUnavailable:
        raise TranscriptError(
            "This video is unavailable. It may be private, deleted, "
            "or restricted in your region."
        )


    except (RequestBlocked, IpBlocked):
        raise TranscriptError(
            "YouTube blocked the transcript request. "
            "Try running the app from your own computer."
        )


    except Exception as error:
        raise TranscriptError(
            f"Could not fetch the transcript.\n\n**Details:** {error}"
        )


    transcript_text = " ".join(
        snippet.text
        for snippet in transcript
    )


    if not transcript_text.strip():
        raise TranscriptError(
            "The transcript is empty, so this video cannot be indexed."
        )


    return transcript_text



def build_rag_chain(transcript: str):
    """
    Builds the complete RAG chain:


    user question
    -> retrieve relevant transcript chunks
    -> send context and question to Groq
    -> return answer
    """


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )


    chunks = splitter.create_documents([transcript])


    if not chunks:
        raise TranscriptError(
            "The transcript is empty, so this video cannot be indexed."
        )


    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    try:
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )


    except Exception as error:
        raise TranscriptError(
            f"Could not create the transcript index.\n\n**Details:** {error}"
        )


    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )


    prompt = PromptTemplate(
        template="""
        You are a helpful assistant that answers questions using only the
        transcript of a YouTube video.


        Rules:
        1. Use only the context below.
        2. Do not use outside knowledge.
        3. Do not make up any information.
        4. If the answer is not found in the context, reply exactly:
           "This question is not related to or answerable from this YouTube video."


        Context:
        {context}


        Question:
        {question}


        Answer:
        """,
        input_variables=["context", "question"],
    )


    def format_documents(documents):
        return "\n\n".join(
            document.page_content
            for document in documents
        )


    try:
        llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0.2,
            groq_api_key=os.getenv("GROQ_API_KEY"),
        )


    except Exception as error:
        raise TranscriptError(
            f"Could not connect to Groq.\n\n**Details:** {error}"
        )


    parallel_chain = RunnableParallel(
        {
            "context": retriever | RunnableLambda(format_documents),
            "question": RunnablePassthrough(),
        }
    )


    rag_chain = parallel_chain | prompt | llm | StrOutputParser()


    return rag_chain