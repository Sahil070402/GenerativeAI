import streamlit as st
import os
import re
import time
import requests
import subprocess
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from rank_bm25 import BM25Okapi  # For hybrid retrieval
from sentence_transformers import CrossEncoder  # For reranking
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from audio_recorder_streamlit import audio_recorder


# Load environment variables
load_dotenv('.env')

# Initialize session state
if 'vector_stores' not in st.session_state:
    st.session_state.vector_stores = {}
if 'bm25_indices' not in st.session_state:
    st.session_state.bm25_indices = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'chat_sessions' not in st.session_state:
    st.session_state.chat_sessions = {}
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'session_counter' not in st.session_state:
    st.session_state.session_counter = 0
if 'show_steps' not in st.session_state:
    st.session_state.show_steps = False
if 'deep_analysis_mode' not in st.session_state:
    st.session_state.deep_analysis_mode = False
if 'audio_text' not in st.session_state:
    st.session_state.audio_text = ""
if 'recording_complete' not in st.session_state:
    st.session_state.recording_complete = False
if 'last_audio_bytes' not in st.session_state:
    st.session_state.last_audio_bytes = None
if 'text_input_content' not in st.session_state:
    st.session_state.text_input_content = ""
if 'clear_counter' not in st.session_state:
    st.session_state.clear_counter = 0
# Function to transcribe audio using Hugging Face Whisper API
def transcribe_audio(audio_bytes):
    """
    Transcribe audio using Hugging Face Inference API with Whisper
    """
    try:
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        if not hf_token:
            st.error("Please set your HUGGINGFACE_API_KEY in the .env file")
            return None
        
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
        
        # Show progress
        with st.spinner("Transcribing audio... 🎤"):
            # If the audio_bytes is None or empty, return None
            if not audio_bytes:
                st.error("No audio data received")
                return None
            
            # Try different content types in order of likelihood
            content_types = ["audio/webm", "audio/wav", "audio/mpeg", "audio/ogg"]
            
            for content_type in content_types:
                headers = {
                    "Authorization": f"Bearer {hf_token}",
                    "Content-Type": content_type
                }
                
                # Send the audio bytes directly
                response = requests.post(API_URL, headers=headers, data=audio_bytes)
                
                if response.status_code == 200:
                    result = response.json()
                    if 'text' in result:
                        return result['text'].strip()
                    else:
                        st.error("No transcription text found in the response")
                        return None
                elif response.status_code == 503:
                    st.warning("Model is loading, please wait and try again in a few seconds...")
                    return None
                elif response.status_code != 400:
                    # If it's not a 400 error, break and show the error
                    break
            
            # If we get here, all content types failed
            st.error(f"Error from Hugging Face API: {response.status_code} - {response.text}")
            return None
                
    except Exception as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    try:
        pattern = r'(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})(?:&|$|[^a-zA-Z0-9_-])'
        match = re.search(pattern, url)
        return match.group(1) if match else None
    except Exception:
        return None

# Function to fetch and process transcripts
# Function to fetch and process transcripts
def process_videos(video_urls):
    """
    Sure-shot method using yt-dlp - works 99% of the time
    """
    transcripts = []
    
    # Add progress bar for better UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, url in enumerate(video_urls):
        video_id = extract_video_id(url)
        if not video_id:
            st.error(f"Invalid YouTube URL: {url}")
            continue
            
        status_text.text(f"Processing video {i+1}/{len(video_urls)}: {video_id}")
        progress_bar.progress((i) / len(video_urls))
        
        try:
            # Use yt-dlp to get video info and subtitles
            cmd = [
                'yt-dlp', 
                '--write-auto-sub',  # Get auto-generated subtitles
                '--write-sub',       # Get manual subtitles if available
                '--sub-lang', 'en',  # Prefer English
                '--skip-download',   # Don't download video
                '--sub-format', 'vtt',
                '--output', f'temp_%(id)s.%(ext)s',
                url
            ]
            
            # Add longer timeout and better error handling
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            # Wait a moment for files to be written
            import time
            time.sleep(1)
            
            # Look for generated subtitle files
            import glob
            import os
            
            subtitle_files = glob.glob(f'temp_{video_id}*.vtt')
            
            transcript_text = ""
            
            # Try multiple times if files aren't found immediately
            max_retries = 3
            for retry in range(max_retries):
                subtitle_files = glob.glob(f'temp_{video_id}*.vtt')
                if subtitle_files:
                    break
                time.sleep(2)  # Wait 2 seconds before retry
                
            for subtitle_file in subtitle_files:
                try:
                    with open(subtitle_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        transcript_text = parse_vtt_content(content)
                    os.remove(subtitle_file)  # Clean up
                    break
                except Exception as e:
                    st.warning(f"Error reading subtitle file {subtitle_file}: {str(e)}")
                    continue
            
            if transcript_text.strip():
                transcripts.append(transcript_text)
                st.success(f"✅ Successfully extracted transcript for: {url}")
            else:
                st.error(f"❌ No transcript found for: {url}")
                
        except subprocess.TimeoutExpired:
            st.error(f"⏱️ Timeout while processing: {url}")
        except Exception as e:
            st.error(f"💥 Error processing {url}: {str(e)}")
            continue

    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    
    # Clear progress indicators after a moment
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

    if not transcripts:
        st.error("❌ No valid transcripts retrieved. Please check the video URLs.")
        return None, None

    combined_transcript = " ".join(transcripts)
    
    if not combined_transcript.strip():
        st.error("❌ All retrieved transcripts are empty.")
        return None, None

    st.info("🔄 Creating vector store and search index...")

    # Split transcript into chunks (same as original)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([combined_transcript])

    # Create vector store (same as original)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        # Create BM25 index for hybrid retrieval (same as original)
        tokenized_chunks = [doc.page_content.split() for doc in chunks]
        bm25_index = BM25Okapi(tokenized_chunks)
        
        # Store both in the current session (same as original)
        if st.session_state.current_session_id:
            st.session_state.vector_stores[st.session_state.current_session_id] = vector_store
            st.session_state.bm25_indices[st.session_state.current_session_id] = (bm25_index, chunks)
        
        st.success("✅ Vector store created successfully!")
        return vector_store, (bm25_index, chunks)
    except Exception as e:
        st.error(f"💥 Error creating vector store: {str(e)}")
        return None, None

def parse_vtt_content(content):
    """Parse VTT subtitle content and extract clean text"""
    import re
    
    # Remove VTT headers
    content = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)
    
    # Remove timing lines (00:00:00.000 --> 00:00:05.000)
    content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}.*?\n', '', content)
    
    # Remove position and styling tags
    content = re.sub(r'<[^>]+>', '', content)
    content = re.sub(r'align:start position:\d+%', '', content)
    
    # Split into lines and clean
    lines = content.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, numbers, and timing info
        if line and not line.isdigit() and not re.match(r'\d{2}:\d{2}:\d{2}', line):
            clean_lines.append(line)
    
    return ' '.join(clean_lines)

# Function for hybrid retrieval with fixed deduplication
def hybrid_retrieval(query, vector_store, bm25_index, chunks, k=4):
    vector_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": k * 2})
    vector_docs = vector_retriever.invoke(query)

    tokenized_query = query.split()
    bm25_scores = bm25_index.get_scores(tokenized_query)
    bm25_sorted_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)
    bm25_docs = [chunks[i] for i in bm25_sorted_indices[:k * 2]]

    combined_docs = []
    seen_content = set()
    for doc in vector_docs + bm25_docs:
        content = doc.page_content
        if content not in seen_content:
            seen_content.add(content)
            combined_docs.append(doc)

    return combined_docs[:k * 2]

# Function to rerank documents (only used in Deep Analysis Mode)
def rerank_documents(query, docs, k=4):
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    sorted_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return sorted_docs[:k]

# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit app UI
st.title("Sahil's YT Chatbot with Speech Input 🎤")

# Input for YouTube links
st.subheader("Enter YouTube Video Links")
video_links = st.text_area("Paste one or more YouTube video links (one per line)", height=100)
submit_button = st.button("Submit")

# Process submitted videos
if submit_button and video_links:
    with st.spinner("Processing videos..."):
        video_urls = [url.strip() for url in video_links.split('\n') if url.strip()]
        if video_urls:
            if st.session_state.current_session_id is None:
                st.session_state.session_counter += 1
                st.session_state.current_session_id = f"Session {st.session_state.session_counter}"
                st.session_state.chat_sessions[st.session_state.current_session_id] = []
            vector_store, bm25_data = process_videos(video_urls)
            if vector_store:
                st.session_state.video_processed = True
                st.success("Videos processed successfully! You can now ask questions.")
            else:
                st.session_state.video_processed = False
        else:
            st.error("Please provide at least one valid YouTube URL.")

# Chat interface
if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.vector_stores:
    with st.sidebar:
        st.subheader("Chat Session Management")
        if st.button("New Chat"):
            st.session_state.session_counter += 1
            st.session_state.current_session_id = f"Session {st.session_state.session_counter}"
            st.session_state.chat_sessions[st.session_state.current_session_id] = []
            if st.session_state.current_session_id in st.session_state.vector_stores:
                del st.session_state.vector_stores[st.session_state.current_session_id]
            if st.session_state.current_session_id in st.session_state.bm25_indices:
                del st.session_state.bm25_indices[st.session_state.current_session_id]
            st.session_state.video_processed = False
            st.session_state.audio_text = ""
            st.session_state.recording_complete = False
            st.success("Started a new chat session!")
        
        session_ids = list(st.session_state.chat_sessions.keys())
        if session_ids:
            selected_session = st.selectbox("Select Previous Chat", options=session_ids, index=session_ids.index(st.session_state.current_session_id) if st.session_state.current_session_id in session_ids else 0)
            if selected_session != st.session_state.current_session_id:
                st.session_state.current_session_id = selected_session
                st.session_state.video_processed = st.session_state.current_session_id in st.session_state.vector_stores
                st.session_state.audio_text = ""
                st.session_state.recording_complete = False
                st.success(f"Switched to {selected_session}")

        # Toggles for showing steps and mode selection
        st.session_state.show_steps = st.checkbox("Show Processing Flow", value=st.session_state.show_steps)
        st.session_state.deep_analysis_mode = st.checkbox("Deep Analysis Mode (More Accurate, Slower)", value=st.session_state.deep_analysis_mode)
        if st.session_state.deep_analysis_mode:
            st.info("Deep Analysis Mode: Prioritizing accuracy and context quality.")
        else:
            st.info("Normal Mode: Prioritizing speed for quick responses.")

        # Sidebar container for Processing Flow
        steps_container = st.container()

    # Chat section now takes full width (no col2 needed)
    st.subheader("Ask Questions About the Video(s)")

    # Display chat history for the current session
    if st.session_state.current_session_id and st.session_state.current_session_id in st.session_state.chat_sessions:
        for message in st.session_state.chat_sessions[st.session_state.current_session_id]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Initialize components
    vector_store = st.session_state.vector_stores[st.session_state.current_session_id]
    bm25_index, chunks = st.session_state.bm25_indices[st.session_state.current_session_id]

    model = ChatOpenAI(
        model="gpt-4o-2024-08-06",
        openai_api_key=os.getenv("OPEN_API_KEY"),
        openai_api_base=os.getenv("BASE_URL"),
    )

    # Set up components for Deep Analysis Mode
    if st.session_state.deep_analysis_mode:
        # Set up the base retriever with MMR
        base_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})

        # Set up MultiQueryRetriever
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=model
        )

        # Set up ContextualCompressionRetriever
        compressor = LLMChainExtractor.from_llm(llm=model)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=RunnableLambda(lambda query: custom_retrieval_deep(query, steps_dict, step_placeholders))
        )

    prompt = PromptTemplate(
        template="""You are **Sahil's YT Chatbot** 🤖 — a friendly, English-only AI assistant created by Sahil to answer questions about YouTube videos using transcript data.

🛑 Strict Rules You MUST Follow:
1. Always respond **only in English**, no matter what language the transcript is in.
2. Use **only the provided transcript context** to answer questions.
3. If the transcript doesn't contain relevant info, reply politely with:
   _"Sorry, I couldn't find that in the video content."_
4. If asked **about yourself**, respond with:
   _"I'm a YT Chatbot designed by Sahil Sareen, created to answer questions based on YouTube video transcripts." 😊
5. If asked who is featured/appears in the video, look for mentions of guests, interviewees, or other people discussed in the transcript content - do not assume the video creator is the featured person.
6. Keep responses **concise, friendly**, and **use emojis** to make answers more engaging.

👇 Transcript Context:
{context}

❓ User Question:
{question}

💬 Your Answer:
""",
        input_variables=["context", "question"]
    )

    parser = StrOutputParser()

    # Step titles for display (different for each mode)
    step_titles = {
        False: {  # Normal Mode
            1: "Performing hybrid retrieval (vector + keyword search)",
            2: "Combining and deduplicating documents",
            3: "Generating prompt for the language model",
            4: "Generating response",
            5: "Parsing and formatting response"
        },
        True: {  # Deep Analysis Mode
            1: "Generating multiple query variations",
            2: "Performing hybrid retrieval (vector + keyword search)",
            3: "Combining and deduplicating documents",
            4: "Reranking documents",
            5: "Compressing retrieved documents",
            6: "Generating prompt for the language model",
            7: "Generating response",
            8: "Parsing and formatting response"
        }
    }

    # Function to initialize and display the steps (called once per query)
    def initialize_steps_display(steps_dict, steps_container):
        if st.session_state.show_steps:
            with steps_container:
                placeholder = st.empty()
                with placeholder.container():
                    st.markdown("### Processing Flow")
                    step_placeholders = {}
                    for step_num in steps_dict:
                        step_placeholders[step_num] = st.empty()
                        step_placeholders[step_num].markdown(
                            f"**Step {step_num}:** {step_titles[st.session_state.deep_analysis_mode][step_num]} - {steps_dict[step_num]}",
                            unsafe_allow_html=True
                        )
                    return step_placeholders
        return {}

    # Function to update a specific step's status
    def update_step_status(step_placeholders, step_num, status, steps_dict):
        steps_dict[step_num] = status
        if st.session_state.show_steps and step_placeholders:
            step_placeholders[step_num].markdown(
                f"**Step {step_num}:** {step_titles[st.session_state.deep_analysis_mode][step_num]} - {status}",
                unsafe_allow_html=True
            )
        time.sleep(0.5)  # Small delay for visual transition

    # Retrieval function for Deep Analysis Mode (Steps 1-4)
    def custom_retrieval_deep(query, steps_dict, step_placeholders):
        # Step 1: Multi-Query Generation
        update_step_status(step_placeholders, 1, "🔄 Running...", steps_dict)
        docs = multi_query_retriever.invoke(query)
        update_step_status(step_placeholders, 1, "✅ Completed", steps_dict)

        # Step 2: Hybrid Retrieval
        update_step_status(step_placeholders, 2, "🔄 Running...", steps_dict)
        hybrid_docs = hybrid_retrieval(query, vector_store, bm25_index, chunks, k=4)
        update_step_status(step_placeholders, 2, "✅ Completed", steps_dict)

        # Step 3: Combine and Deduplicate
        update_step_status(step_placeholders, 3, "🔄 Running...", steps_dict)
        combined_docs = []
        seen_content = set()
        for doc in docs + hybrid_docs:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                combined_docs.append(doc)
        combined_docs = combined_docs[:8]
        update_step_status(step_placeholders, 3, "✅ Completed", steps_dict)

        # Step 4: Reranking
        update_step_status(step_placeholders, 4, "🔄 Running...", steps_dict)
        reranked_docs = rerank_documents(query, combined_docs, k=4)
        update_step_status(step_placeholders, 4, "✅ Completed", steps_dict)

        return reranked_docs

    # Retrieval function for Normal Mode (Steps 1-2)
    def custom_retrieval_normal(query, steps_dict, step_placeholders):
        # Step 1: Hybrid Retrieval
        update_step_status(step_placeholders, 1, "🔄 Running...", steps_dict)
        hybrid_docs = hybrid_retrieval(query, vector_store, bm25_index, chunks, k=2)
        update_step_status(step_placeholders, 1, "✅ Completed", steps_dict)

        # Step 2: Combine and Deduplicate
        update_step_status(step_placeholders, 2, "🔄 Running...", steps_dict)
        combined_docs = []
        seen_content = set()
        for doc in hybrid_docs:
            content = doc.page_content
            if content not in seen_content:
                seen_content.add(content)
                combined_docs.append(doc)
        combined_docs = combined_docs[:4]
        update_step_status(step_placeholders, 2, "✅ Completed", steps_dict)

        return combined_docs

    # Function to compress documents (Step 5 in Deep Analysis Mode)
    def compress_documents(query, docs, step_placeholders, steps_dict):
        update_step_status(step_placeholders, 5, "🔄 Running...", steps_dict)
        compressor = LLMChainExtractor.from_llm(llm=model)
        compressed_docs = compressor.compress_documents(docs, query)
        update_step_status(step_placeholders, 5, "✅ Completed", steps_dict)
        return compressed_docs

    # RAG chain with step-by-step status updates
    def generate_response_with_steps(query, steps_dict, step_placeholders):
        if st.session_state.deep_analysis_mode:
            # Steps 1-4: Retrieval
            docs = custom_retrieval_deep(query, steps_dict, step_placeholders)

            # Step 5: Contextual Compression
            compressed_docs = compress_documents(query, docs, step_placeholders, steps_dict)

            # Step 6: Prompting
            step_num = 6
            update_step_status(step_placeholders, step_num, "🔄 Running...", steps_dict)
            context = format_docs(compressed_docs)
            update_step_status(step_placeholders, step_num, "✅ Completed", steps_dict)

            # Step 7: Generating Response
            step_num += 1
            update_step_status(step_placeholders, step_num, "🔄 Running...", steps_dict)
            parallel_chain = RunnableParallel({
                'context': lambda _: context,
                'question': lambda _: query
            })
            main_chain = parallel_chain | prompt | model | parser
            response = main_chain.invoke(query)
            update_step_status(step_placeholders, step_num, "✅ Completed", steps_dict)

            # Step 8: Parsing Response
            step_num += 1
            update_step_status(step_placeholders, step_num, "🔄 Running...", steps_dict)
            update_step_status(step_placeholders, step_num, "✅ Completed", steps_dict)

        else:  # Normal Mode
            # Steps 1-2: Retrieval
            docs = custom_retrieval_normal(query, steps_dict, step_placeholders)

            # Step 3: Prompting
            step_num = 3
            update_step_status(step_placeholders, step_num, "🔄 Running...", steps_dict)
            context = format_docs(docs)
            update_step_status(step_placeholders, step_num, "✅ Completed", steps_dict)

            # Step 4: Generating Response
            step_num += 1
            update_step_status(step_placeholders, step_num, "🔄 Running...", steps_dict)
            parallel_chain = RunnableParallel({
                'context': lambda _: context,
                'question': lambda _: query
            })
            main_chain = parallel_chain | prompt | model | parser
            response = main_chain.invoke(query)
            update_step_status(step_placeholders, step_num, "✅ Completed", steps_dict)

            # Step 5: Parsing Response
            step_num += 1
            update_step_status(step_placeholders, step_num, "🔄 Running...", steps_dict)
            update_step_status(step_placeholders, step_num, "✅ Completed", steps_dict)

        return response

    # Speech Input Section (Replace the existing speech input section with this)
    st.subheader("🎤 Voice Input")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.write("Record your question:")
        audio_bytes = audio_recorder(
            text="Click to record",
            recording_color="#e8b4f0",
            neutral_color="#6aa36f",
            icon_name="microphone",
            icon_size="2x",
        )
    
    with col2:

        if 'clear_counter' not in st.session_state:
            st.session_state.clear_counter = 0
        # Text input area that can be filled by speech or typed manually
        user_input = st.text_area(
            "Type your question or use voice input above:",
            value=st.session_state.text_input_content,
            height=100,
            key=f"text_input_area_{st.session_state.clear_counter}"
        )
           # Update the session state when user types
        if user_input != st.session_state.text_input_content:
            st.session_state.text_input_content = user_input

        # Submit button to process the text input
        submit_text_button = st.button("Submit Question", key="submit_text")
        
        # Clear button to reset the text area
        if st.button("Clear", key="clear_text"):
            st.session_state.audio_text = ""
            st.session_state.text_input_content = ""
            # Increment counter to force widget recreation
            st.session_state.clear_counter += 1
            st.rerun()

    # Add a flag to track if we should process new audio
    process_new_audio = False

    # Check if we have new audio data (different from what we've already processed)
    if 'last_audio_bytes' not in st.session_state:
        st.session_state.last_audio_bytes = None

    if audio_bytes and audio_bytes != st.session_state.last_audio_bytes:
        process_new_audio = True
        st.session_state.last_audio_bytes = audio_bytes

    # Process NEW audio when recorded
    if process_new_audio:
        st.write("🔄 Processing your audio...")
        
        # Transcribe the audio
        transcribed_text = transcribe_audio(audio_bytes)
        
        if transcribed_text:
            st.session_state.audio_text = transcribed_text
            st.session_state.text_input_content = transcribed_text
            st.session_state.recording_complete = True
            st.success(f"✅ Transcribed: {transcribed_text}")
            
            # Process the audio query automatically
            if st.session_state.current_session_id is None:
                st.session_state.session_counter += 1
                st.session_state.current_session_id = f"Session {st.session_state.session_counter}"
                st.session_state.chat_sessions[st.session_state.current_session_id] = []

            st.session_state.chat_sessions[st.session_state.current_session_id].append({"role": "user", "content": transcribed_text})
            
            with st.chat_message("user"):
                st.markdown(transcribed_text)

            # Process the query and show steps if enabled
            with st.spinner("Processing your query..."):
                # Initialize steps dictionary based on mode
                max_steps = 8 if st.session_state.deep_analysis_mode else 5
                steps_dict = {i: "⏳ Pending" for i in range(1, max_steps + 1)}
                step_placeholders = initialize_steps_display(steps_dict, steps_container)

                try:
                    response = generate_response_with_steps(transcribed_text, steps_dict, step_placeholders)
                    st.session_state.chat_sessions[st.session_state.current_session_id].append({"role": "assistant", "content": response})
                    with st.chat_message("assistant"):
                        st.markdown(response)
                        
                    # Reset audio text after processing
                    st.session_state.audio_text = ""
                    st.session_state.recording_complete = False
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        else:
            st.error("Failed to transcribe audio. Please try again.")

    # Manual text input processing (only when Submit button is clicked and no new audio is being processed)
    elif submit_text_button and user_input and user_input.strip() and not process_new_audio:
        if st.session_state.current_session_id is None:
            st.session_state.session_counter += 1
            st.session_state.current_session_id = f"Session {st.session_state.session_counter}"
            st.session_state.chat_sessions[st.session_state.current_session_id] = []

        st.session_state.chat_sessions[st.session_state.current_session_id].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Process the query and show steps if enabled
        with st.spinner("Processing your query..."):
            # Initialize steps dictionary based on mode
            max_steps = 8 if st.session_state.deep_analysis_mode else 5
            steps_dict = {i: "⏳ Pending" for i in range(1, max_steps + 1)}
            step_placeholders = initialize_steps_display(steps_dict, steps_container)

            try:
                response = generate_response_with_steps(user_input, steps_dict, step_placeholders)
                st.session_state.chat_sessions[st.session_state.current_session_id].append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
                
                # Clear the text area after successful processing by setting session state and rerunning
                st.session_state.audio_text = ""
                st.session_state.text_input_content = ""
                st.rerun()
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")


else:
    if submit_button:
        st.warning("Please process a valid YouTube video first.")
    else:
        st.info("👆 Please submit YouTube video links above to start chatting! and hit clear button after every new request")
        st.markdown("""
        ### 🚀 Features:
        - **Text Input**: Type your questions normally
        - **Voice Input**: Click the microphone to record your question
        - **Hybrid Retrieval**: Advanced search combining vector and keyword methods  
        - **Deep Analysis Mode**: More accurate but slower processing
        - **Session Management**: Multiple chat sessions with history
        - **Processing Flow**: Optional step-by-step visualization
        """)

