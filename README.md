# 🎬 Chit-Chat with YouTube

**Chat with any YouTube video using its transcript.**  
Paste a YouTube link, load the video, and ask questions. The app answers only from that video’s transcript using Retrieval-Augmented Generation (RAG).

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-orange)
![Groq](https://img.shields.io/badge/Groq-LLM-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Live Demo

_Add your deployed app link here after deployment._

```text
[https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)
```

## Screenshots

### Home Page

![Home Page](screenshots/01-home-page.png)

### Video Loaded

![Video Loaded](screenshots/02-video-loaded.png)

### Chat Conversation

![Chat Conversation](screenshots/03-chat-conversation.png)

## Table of Contents

- [Project Goal](#project-goal)
- [Problem Statement](#problem-statement)
- [What It Does](#what-it-does)
- [Features](#features)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [RAG Fundamentals](#rag-fundamentals)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Project Structure](#project-structure)
- [Example Output](#example-output)
- [Testing](#testing)
- [Current Status](#current-status)
- [Milestones](#milestones)
- [Roadmap](#roadmap)
- [Known Limitations](#known-limitations)
- [Data and Privacy](#data-and-privacy)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)
- [Connect](#connect)
- [Acknowledgments](#acknowledgments)
- [FAQ / Troubleshooting](#faq--troubleshooting)

## Project Goal

The goal of this project is to build an AI-powered application that allows users to chat with a YouTube video.

Instead of watching an entire video to find one piece of information, a user can paste the video link and ask a question. The application retrieves the most relevant part of the video transcript and generates an answer based only on that transcript.

## Problem Statement

YouTube videos can be long, and finding a specific answer inside them takes time.

Traditional search only gives timestamps or video titles. It does not understand the meaning of the video content.

This project solves that problem by:

- Fetching the video transcript
- Breaking it into smaller meaningful parts
- Converting those parts into numerical vectors
- Storing them in a vector database
- Retrieving only the relevant parts for a user question
- Generating a grounded answer using an LLM

## What It Does

This application allows a user to:

1. Paste a YouTube video link or video ID.
2. Select the preferred transcript language.
3. Load the video.
4. Ask questions about the video.
5. Receive answers based only on the video transcript.
6. Get a clear response when a question cannot be answered from the transcript.

The app does not hallucinate outside information. If the answer is not present in the transcript, it replies:

```text
This question is not related to or answerable from this YouTube video.
```

## Features

- YouTube link and video ID support
- Supports normal YouTube links, Shorts links, `youtu.be` links, and embed links
- Transcript language selection
- Clean and modern Streamlit interface
- Video title, channel name, and thumbnail display
- Local Hugging Face embeddings
- FAISS vector database
- Retrieval-Augmented Generation
- Groq-powered answer generation
- Prevents hallucination through a strict prompt
- Chat history during the current session
- Reset conversation option
- Error messages for unavailable videos, disabled captions, and unsupported languages

## How It Works

The application follows the RAG workflow:

```text
YouTube Link
     |
     v
Video ID Extraction
     |
     v
Transcript Fetching
     |
     v
Document Creation
     |
     v
Text Chunking
     |
     v
Embedding Generation
     |
     v
FAISS Vector Store
     |
     v
User Question
     |
     v
Relevant Transcript Retrieval
     |
     v
Prompt Creation
     |
     v
Groq LLM
     |
     v
Final Answer
```

## Architecture

```text
+---------------------+
|   Streamlit UI      |
+---------------------+
          |
          v
+---------------------+
| YouTube Transcript  |
|       API           |
+---------------------+
          |
          v
+---------------------+
|  Text Splitter      |
+---------------------+
          |
          v
+---------------------+
| Hugging Face        |
|    Embeddings       |
+---------------------+
          |
          v
+---------------------+
|     FAISS           |
|   Vector Store      |
+---------------------+
          |
          v
+---------------------+
|     Retriever       |
+---------------------+
          |
          v
+---------------------+
|   Groq LLM          |
+---------------------+
          |
          v
+---------------------+
|    Final Answer     |
+---------------------+
```

## RAG Fundamentals

This project helped me learn and apply the main concepts of Retrieval-Augmented Generation, also known as RAG.

### Document Loading

Document loading means converting raw data into a format that LangChain can process.

In this project, the YouTube transcript is fetched using `youtube-transcript-api`. The transcript is converted into one large text string.

```python
transcript_text = " ".join(
    snippet.text
    for snippet in transcript
)
```

This text is then converted into LangChain `Document` objects using:

```python
chunks = splitter.create_documents([transcript])
```

### Text Chunking

Text chunking means dividing a long document into smaller pieces.

A long transcript cannot be sent directly to an LLM because it may be too large. Therefore, the transcript is divided into chunks.

This project uses `RecursiveCharacterTextSplitter`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
```

- `chunk_size=1000` means each chunk contains approximately 1000 characters.
- `chunk_overlap=200` means 200 characters are repeated between two nearby chunks.
- Overlap helps preserve context between chunks.

### Embedding

Embedding means converting text into numerical vectors.

Similar meanings produce similar vectors. For example, sentences about “machine learning” and “AI models” may have vectors that are close to each other.

This project uses a free local embedding model:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

This means the app does not use OpenAI credits for embedding generation.

### Vector Storage

A vector store stores embeddings so that similar text can be found quickly.

This project uses FAISS, which stands for **Facebook AI Similarity Search**. It is a popular library for fast similarity search.

```python
vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings,
)
```

FAISS stores the transcript chunks and their embeddings. Later, when a user asks a question, FAISS finds the most similar transcript chunks.

### Retrieval

Retrieval means finding the most relevant information from the vector store.

When a user asks a question, the question is converted into an embedding. FAISS then compares that embedding with the transcript embeddings and returns the most similar chunks.

```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)
```

Here, `k=5` means the app retrieves the top 5 most relevant transcript chunks.

### Augmentation

Augmentation means adding retrieved information to the prompt.

The retrieved transcript chunks are combined into one context string:

```python
def format_documents(documents):
    return "\n\n".join(
        document.page_content
        for document in documents
    )
```

This context is sent to the LLM along with the user question.

### Generation

Generation means creating the final answer using an LLM.

This project uses Groq’s `openai/gpt-oss-20b` model:

```python
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
)
```

The LLM receives:

- Retrieved transcript context
- User question
- Strict instructions not to hallucinate

It then generates the final answer.

## LangChain Concepts Used

### Runnables

LangChain Runnables are used to connect different steps into one pipeline.

This project uses:

```python
RunnableParallel
RunnablePassthrough
RunnableLambda
```

The main chain is:

```python
parallel_chain = RunnableParallel(
    {
        "context": retriever | RunnableLambda(format_documents),
        "question": RunnablePassthrough(),
    }
)

rag_chain = parallel_chain | prompt | llm | StrOutputParser()
```

### Sequential Chain

The RAG pipeline is a sequential chain:

```text
Retrieve -> Format Context -> Prompt -> LLM -> Output Parser
```

First, the retriever finds transcript chunks. Then, the documents are formatted. After that, the prompt is created. Finally, the LLM generates the answer.

### Output Parser

This project uses `StrOutputParser`:

```python
from langchain_core.output_parsers import StrOutputParser
```

It converts the LLM response into a simple string that can be displayed directly in Streamlit.

```python
rag_chain = parallel_chain | prompt | llm | StrOutputParser()
```

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| LangChain | RAG and chain orchestration |
| youtube-transcript-api | Transcript fetching |
| YouTube Data API | Video title, channel, and thumbnail |
| Hugging Face Embeddings | Free local embedding generation |
| FAISS | Vector storage and similarity search |
| Groq | Cloud LLM inference |
| Python Dotenv | Environment variable management |
| Requests | HTTP requests |

## Prerequisites

Before running this project, install:

- Python 3.10 or newer
- Git
- A Groq API key
- Optional: YouTube Data API key

## Quick Start

Clone the repository:

```bash
git clone [https://github.com/YOUR_USERNAME/youtube-chatbot.git](https://github.com/YOUR_USERNAME/youtube-chatbot.git)
cd youtube-chatbot
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create your `.env` file:

```bash
cp .env.example .env
```

Add your API keys to `.env`.

Run the app:

```bash
streamlit run app.py
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk-your-groq-api-key
YOUTUBE_API_KEY=your-youtube-api-key
```

| Variable | Required | Purpose |
|---|---:|---|
| `GROQ_API_KEY` | Yes | Generates answers using Groq |
| `YOUTUBE_API_KEY` | Optional | Fetches official video title, channel, and thumbnail |

Never upload `.env` to GitHub.

## Usage

1. Open the Streamlit app.
2. Paste a YouTube link.
3. Select the transcript language.
4. Click **Load video**.
5. Wait for the transcript to load.
6. Ask a question in the chat box.
7. Click **Reset conversation** to start a new chat.

## API Documentation

### YouTube Transcript API

Used to fetch captions/transcripts.

```python
youtube.fetch(
    video_id=video_id,
    languages=[language_code]
)
```

### YouTube Data API

Used to fetch video metadata.

```python
[https://www.googleapis.com/youtube/v3/videos](https://www.googleapis.com/youtube/v3/videos)
```

### Groq API

Used for answer generation.

```python
ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
```

## Project Structure

```text
youtube-chatbot/
│
├── app.py
├── rag_pipeline.py
├── youtube_utils.py
├── requirements.txt
├── .env
├── .env.example
├── .gitignore
├── README.md
│
├── screenshots/
│   ├── 01-home-page.png
│   ├── 02-video-loaded.png
│   └── 03-chat-conversation.png
│
└── venv/
```

## Example Output

```text
User:
What is the main topic of this video?

Assistant:
The main topic of this video is ...
```

If the answer is not available:

```text
User:
Who is the prime minister of India?

Assistant:
This question is not related to or answerable from this YouTube video.
```

## Testing

Test the app using:

- A normal YouTube link
- A YouTube Shorts link
- A `youtu.be` link
- A video with English captions
- A video with Hindi captions
- A video without captions
- An invalid YouTube link
- A question unrelated to the video

## Current Status

- [x] YouTube link parsing
- [x] Transcript fetching
- [x] Text chunking
- [x] Embedding generation
- [x] FAISS vector storage
- [x] Retrieval
- [x] RAG chain
- [x] Groq answer generation
- [x] Streamlit chat interface
- [x] Error handling
- [ ] User authentication
- [ ] Chat history database
- [ ] Multiple video comparison
- [ ] Video summary generation

## Milestones

1. Build YouTube link parser.
2. Fetch YouTube transcript.
3. Split transcript into chunks.
4. Generate embeddings.
5. Store embeddings in FAISS.
6. Build retrieval system.
7. Connect LLM.
8. Build Streamlit interface.
9. Add error handling.
10. Deploy application.

## Roadmap

- Add video summarization.
- Add timestamp-based answers.
- Add support for multiple videos.
- Add PDF and website chat.
- Add chat history using a database.
- Add user login.
- Improve mobile UI.
- Add multilingual answer generation.

## Known Limitations

- The video must have captions available.
- Some videos may block transcript access.
- YouTube may block requests from cloud-server IP addresses.
- The answer quality depends on transcript quality.
- The app cannot answer questions about visual-only content if it is not mentioned in captions.
- Chat history is currently stored only in the browser session.

## Data and Privacy

This project does not store user chat history on a server.

Chat messages are stored only in the current Streamlit browser session. When the user refreshes the page or resets the conversation, the chat history is removed.

API keys should never be shared or committed to GitHub.

## Deployment

### Streamlit Community Cloud

1. Push the project to GitHub.
2. Go to https://share.streamlit.io/
3. Click **New app**.
4. Select your GitHub repository.
5. Select `app.py` as the main file.
6. Open **Advanced settings**.
7. Add secrets:

```toml
GROQ_API_KEY = "gsk-your-groq-key"
YOUTUBE_API_KEY = "your-youtube-api-key"
```

8. Click **Deploy**.

### Hugging Face Spaces

1. Create a new Space.
2. Choose Streamlit as the SDK.
3. Upload the project files.
4. Add `GROQ_API_KEY` and `YOUTUBE_API_KEY` in Space secrets.
5. Deploy the app.

### Render

1. Create a new Web Service.
2. Connect your GitHub repository.
3. Use Python as the environment.
4. Add the start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

5. Add environment variables in Render settings.

## Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your changes.
5. Push to your branch.
6. Open a pull request.

## License

This project is licensed under the MIT License.

## Connect

- GitHub: [Your GitHub Profile](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn Profile](https://www.linkedin.com/in/YOUR_LINKEDIN_ID)
- Email: your-email@example.com

## Acknowledgments

- CampusX RAG Playlist
- LangChain Documentation
- Streamlit Documentation
- Groq Documentation
- Hugging Face Sentence Transformers
- FAISS by Facebook AI Research

## FAQ / Troubleshooting

### Why does it say captions are disabled?

The selected YouTube video does not have captions or a transcript available.

### Why does it say no transcript found?

Try selecting another language. The video may not have a transcript in the selected language.

### Why does YouTube block the request?

YouTube may block transcript requests from cloud-server IP addresses. Try running the app locally.

### Why do I get a Groq model error?

Use a currently supported Groq model. This project uses:

```python
model="openai/gpt-oss-20b"
```

### Why is my API key not working?

Check that your `.env` file is in the project root and contains:

```env
GROQ_API_KEY=gsk-your-key
```

Do not add quotes around the key.

### Can I use OpenAI instead?

Yes. Replace `ChatGroq` with `ChatOpenAI`, but OpenAI requires billing credits.