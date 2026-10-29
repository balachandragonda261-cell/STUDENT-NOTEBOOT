import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# Page Config
st.set_page_config(
    page_title="NoteBot - AI Tutor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

/* Custom Header with Gradient */
.header-container {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 3rem;
    margin-bottom: 0.2rem;
    animation: fadeIn 1.2s ease-out;
}

.subheader-container {
    color: #94a3b8;
    font-size: 1.15rem;
    margin-bottom: 2rem;
    font-weight: 300;
    animation: fadeIn 1.5s ease-out;
}

/* Glassmorphism card for results */
.response-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 1.8rem;
    margin-top: 1.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    border-left: 6px solid #a855f7;
    animation: slideUp 0.5s ease-out;
}

.response-header {
    font-weight: 700;
    font-size: 1.25rem;
    color: #f8fafc;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.response-content {
    color: #e2e8f0;
    font-size: 1.1rem;
    line-height: 1.6;
}

/* Sidebar styling overrides */
section[data-testid="stSidebar"] {
    background-image: linear-gradient(180deg, #090d16 0%, #111827 100%) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}

section[data-testid="stSidebar"] h1 {
    color: #f8fafc;
    font-weight: 700;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes slideUp {
    from { transform: translateY(15px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)

# ----------------- Security Helper -----------------
def sanitize_error(msg, keys_to_mask=None):
    """Masks sensitive keys in error messages to prevent leakage."""
    if not msg:
        return ""
    msg_str = str(msg)
    if keys_to_mask:
        for key in keys_to_mask:
            if key and len(key) > 4:
                msg_str = msg_str.replace(key, "********")
    return msg_str

# ----------------- Helper Cached Functions -----------------

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(uploaded_file):
    """Extracts text from the uploaded PDF file."""
    try:
        my_pdf = PdfReader(uploaded_file)
        text = ""
        for page in my_pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

@st.cache_resource(show_spinner=False)
def create_vector_store(text, embeddings_provider, api_key=None):
    """Splits the text and generates FAISS vector database embeddings using either local HF or OpenAI."""
    splitter = RecursiveCharacterTextSplitter( 
        chunk_size=300,
        chunk_overlap=50,
        length_function=len
    )
    chunks = splitter.split_text(text)
    
    if embeddings_provider == "OpenAI":
        if not api_key:
            raise ValueError("OpenAI API Key is required for OpenAI embeddings.")
        embeddings = OpenAIEmbeddings(api_key=api_key)
    elif embeddings_provider == "HuggingFace (Free & Local)":
        # Using a fast, lightweight local sentence transformer model running on CPU
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embeddings provider: {embeddings_provider}")
        
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# ----------------- App Layout & Flow -----------------

# Sidebar
with st.sidebar:
    st.title("📚 NoteBot Settings")
    st.markdown("---")
    
    # ⚙️ Configuration Selectors
    st.subheader("⚙️ Configuration")
    llm_provider = st.selectbox(
        "LLM Provider", 
        ["Google Gemini", "OpenAI"],
        help="Select the language model provider for answering questions."
    )
    
    gemini_model = "gemini-3.5-flash"  # Default
    if llm_provider == "Google Gemini":
        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.6-flash", "gemini-3.1-pro"],
            help="Select the specific Gemini model version."
        )
        
    embeddings_provider = st.selectbox(
        "Embeddings Provider", 
        ["HuggingFace (Free & Local)", "OpenAI"],
        help="Select how to generate embeddings. HuggingFace runs locally on your device for free without API keys!"
    )
    
    st.markdown("---")
    
    # 🔑 API Keys Entry
    st.subheader("🔑 API Keys")
    
    openai_api_key = None
    gemini_api_key = None
    
    if llm_provider == "OpenAI" or embeddings_provider == "OpenAI":
        openai_key_input = st.text_input(
            "OpenAI API Key", 
            type="password",
            placeholder="sk-proj-...",
            help="Enter your OpenAI API key."
        )
        openai_api_key = openai_key_input or os.environ.get("OPENAI_API_KEY")
        
    if llm_provider == "Google Gemini":
        gemini_key_input = st.text_input(
            "Gemini API Key", 
            type="password",
            placeholder="Enter Gemini API Key...",
            help="Enter your Google Gemini API key (supports legacy AIzaSy and newer AQ keys)."
        )
        gemini_api_key = gemini_key_input or os.environ.get("GEMINI_API_KEY")
        
    st.markdown("---")
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Upload notes PDF and start asking questions", 
        type="pdf"
    )

# Main Page Header
st.markdown('<div class="header-container">NoteBot</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader-container">Your AI assistant tutor. Upload a PDF to start learning!</div>', unsafe_allow_html=True)

# Validation Check for Configuration Keys
key_validation_passed = True
validation_error_message = ""

if embeddings_provider == "OpenAI" and not openai_api_key:
    key_validation_passed = False
    validation_error_message = "⚠️ Please provide an OpenAI API key in the sidebar to use OpenAI Embeddings."
elif llm_provider == "OpenAI" and not openai_api_key:
    key_validation_passed = False
    validation_error_message = "⚠️ Please provide an OpenAI API key in the sidebar to use OpenAI LLM."
elif llm_provider == "Google Gemini":
    if not gemini_api_key:
        key_validation_passed = False
        validation_error_message = "⚠️ Please provide a Gemini API key in the sidebar to use the Google Gemini LLM."

# Main Processing Flow
if uploaded_file is not None:
    if not key_validation_passed:
        st.warning(validation_error_message)
    else:
        # Step 1: Extract Text (cached)
        with st.spinner("Extracting text from PDF..."):
            text_content = extract_text_from_pdf(uploaded_file)
            
        if text_content.strip() == "":
            st.error("No text could be extracted from this PDF. Please try another file.")
        else:
            # Step 2: Create Vector Store (cached)
            with st.spinner(f"Indexing document and creating vector store ({embeddings_provider})..."):
                try:
                    # Choose key to pass based on provider
                    emb_key = openai_api_key if embeddings_provider == "OpenAI" else None
                    vector_store = create_vector_store(text_content, embeddings_provider, emb_key)
                    st.success("✅ Notes indexed successfully! You can now query them below.")
                except Exception as e:
                    st.error(f"Error creating vector store: {e}")
                    vector_store = None
            
            if vector_store:
                st.markdown("---")
                # Step 3: Get user query
                user_query = st.text_input(
                    "Type your query here", 
                    placeholder="e.g., Explain the core concepts of the first chapter."
                )
                
                # Step 4: Semantic search & response generation
                if user_query:
                    with st.spinner(f"Searching context and generating answer ({llm_provider})..."):
                        try:
                            # Retrieve matching chunks
                            matching_chunks = vector_store.similarity_search(user_query)
                            
                            # Initialize LLM based on provider choice
                            if llm_provider == "OpenAI":
                                llm = ChatOpenAI(
                                    api_key=openai_api_key,
                                    max_tokens=300,
                                    temperature=0,
                                    model="gpt-3.5-turbo",
                                    timeout=120,
                                    max_retries=3
                                )
                            elif llm_provider == "Google Gemini":
                                llm = ChatGoogleGenerativeAI(
                                    google_api_key=gemini_api_key,
                                    model=gemini_model,
                                    max_tokens=300,
                                    temperature=0,
                                    timeout=120,
                                    max_retries=3
                                )
                            
                            # Define prompt — explicitly ask for ONE complete paragraph answer
                            customized_prompt = ChatPromptTemplate.from_template(
                                """You are a helpful assistant tutor.
Using ONLY the information from the context below, write a single, complete, well-structured paragraph that fully answers the question.
Do NOT use bullet points, numbered lists, or multiple sections.
Write everything as ONE continuous, clear paragraph.
If the context does not contain enough information, say: "I don't have enough information in the uploaded document to answer this."

Context:
{context}

Question: {input}

Answer (one complete paragraph):"""
                            )

                            chain = create_stuff_documents_chain(llm, customized_prompt)

                            # Invoke once — get the full, complete answer as a single string
                            output = chain.invoke({"input": user_query, "context": matching_chunks})

                            # Render as ONE unified card using a Streamlit container
                            with st.container():
                                st.markdown("""
                                <div class="response-card">
                                    <div class="response-header">
                                        <span>&#129302; NoteBot Says:</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(output)
                            
                            # Display references / sources
                            with st.expander("📖 Show Source Context Chunks"):
                                for idx, chunk in enumerate(matching_chunks):
                                    st.markdown(f"**Source Chunk {idx+1}:**")
                                    st.info(chunk.page_content)
                                    
                        except Exception as e:
                            raw_error = str(e)
                            # Sanitize key leaks
                            keys_to_mask = [k for k in [openai_api_key, gemini_api_key] if k]
                            clean_error = sanitize_error(raw_error, keys_to_mask=keys_to_mask)
                            
                            # Check error signatures
                            if "API key not valid" in clean_error or "API_KEY_INVALID" in clean_error or "INVALID_ARGUMENT" in clean_error:
                                st.error("⚠️ Authentication failed: The provided Gemini API key is invalid. Please verify your credentials in the sidebar and try again.")
                            elif "404" in clean_error or "not found" in clean_error.lower() or "not_found" in clean_error.lower():
                                st.error(f"⚠️ Model not found (404): The selected model '{gemini_model}' is not available or supported for your API key. Please try selecting a different model version in the sidebar.")
                            elif "403" in clean_error or "permission" in clean_error.lower() or "denied" in clean_error.lower():
                                st.error("⚠️ Permission denied (403): Your API key does not have permission to access this model. Please verify your Google AI Studio configuration.")
                            elif "quota" in clean_error.lower() or "429" in clean_error or "RESOURCE_EXHAUSTED" in clean_error:
                                st.error("⚠️ Quota exceeded or rate limit reached (429): Please check your Gemini API billing details or rate limits in Google AI Studio.")
                            elif "500" in clean_error or "503" in clean_error or "internal error" in clean_error.lower() or "service unavailable" in clean_error.lower():
                                st.error("⚠️ Server error (500/503): Google Gemini API service is temporarily unavailable or returned an internal error. Please try again in a few moments.")
                            else:
                                st.error(f"⚠️ An error occurred while generating response: {clean_error}")
else:
    # Landing page state when no PDF is uploaded
    st.info("👈 Please upload a PDF file from the sidebar to begin.")
    
    st.markdown("""
    ### How it works:
    1. **Configure RAG settings**: Choose your LLM and Embeddings providers in the sidebar.
    2. **Upload your PDF**: Drop any notes or document file into the sidebar uploader.
    3. **Enter API Key**: Provide the corresponding API Key for your selected LLM/Embeddings.
    4. **Start Learning**: Type natural language questions, and get precise context-aware answers.
    
    💡 *Tip: To run the app completely for free, select **HuggingFace (Free & Local)** as the Embeddings Provider, and **Google Gemini** as the LLM Provider.*
    """)
