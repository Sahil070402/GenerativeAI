# Generative AI YouTube Chatbot

This project is a sophisticated YouTube chatbot built with Streamlit, LangChain, and various large language models (LLMs). It allows users to interact with the content of YouTube videos by providing their links. The application fetches video transcripts, processes them, and enables a conversational AI to answer questions based on the video's content. A key feature of this chatbot is its ability to accept voice input, which is transcribed into text for a seamless user experience.

## 🚀 Features

- **YouTube Video Integration**: Simply paste YouTube video links to start a conversation.
- **Voice Input**: Use your microphone to ask questions.
- **Hybrid Search**: Utilizes a combination of vector-based similarity search (FAISS) and keyword-based search (BM25) for accurate and relevant context retrieval.
- **Dual Analysis Modes**:
    - **Normal Mode**: Fast and efficient, ideal for quick questions.
    - **Deep Analysis Mode**: A more thorough and context-aware mode for complex queries, using multi-query retrieval and reranking.
- **Session Management**: Manage multiple chat sessions and switch between them.
- **Real-time Processing Flow**: Optionally, view the step-by-step process of how your query is handled.
- **Secure API Key Management**: Uses a `.env` file to keep your API keys safe.

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Core AI/LLM Framework**: LangChain
- **LLMs**: OpenAI (GPT-4.1), Google Generative AI (for embeddings)
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Search**: BM25 (for keyword search), Sentence Transformers (for reranking)
- **Speech-to-Text**: Hugging Face Inference API (Whisper)
- **YouTube Transcript**: `youtube-transcript-api`, `yt-dlp`

## ⚙️ Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Sahil070402/GenerativeAI.git
    cd GenerativeAI
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** in the root of the project and add your API keys:
    ```env
    OPEN_API_KEY="your_openai_api_key"
    BASE_URL="your_openai_base_url"  # Or the default OpenAI API endpoint
    GOOGLE_API_KEY="your_google_api_key"
    HUGGINGFACE_API_KEY="your_huggingface_api_key"
    ```

## ▶️ How to Run

Once you have completed the setup, run the Streamlit application:

```bash
streamlit run app.py
```

Your browser will open a new tab with the application running.

## 📝 How to Use

1.  **Enter YouTube Video Links**: Paste one or more YouTube video URLs into the text area and click "Submit".
2.  **Ask Questions**: Once the videos are processed, you can ask questions about their content using the text input or the voice recorder.
3.  **Switch Modes**: Use the checkboxes in the sidebar to toggle between "Normal Mode" and "Deep Analysis Mode".
4.  **Manage Sessions**: Use the sidebar to create new chat sessions or switch between existing ones.

## 📂 Project Structure

```
.
├── .gitignore
├── app.py
├── requirements.txt
└── README.md
```

-   `app.py`: The main Streamlit application file containing all the logic.
-   `requirements.txt`: A list of all the Python dependencies.
-   `.gitignore`: Specifies which files and directories to ignore for version control.
-   `README.md`: This file.
