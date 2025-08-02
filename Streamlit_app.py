import streamlit as st
import os
import time
import json
import hashlib
import pandas as pd
from PIL import Image
from PyPDF2 import PdfReader
import google.generativeai as genai
import docx
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from datetime import datetime
import warnings
import traceback

# Suppress warnings
warnings.filterwarnings('ignore')
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Elite Document & Image AI Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==================== AI CONFIGURATION ====================

def configure_google_ai():
    """Configure Google AI with robust key handling"""
    try:
        api_key = None
        
        # Try Streamlit secrets first
        if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"].strip().strip('"').strip("'")
        
        # Fallback to environment variables
        if not api_key:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                api_key = api_key.strip().strip('"').strip("'")
        
        if not api_key:
            raise Exception("Google API Key not found. Please add GOOGLE_API_KEY to secrets.")
        
        genai.configure(api_key=api_key)
        return api_key, True, "Google AI configured successfully!"
        
    except Exception as e:
        return None, False, f"Google AI configuration error: {str(e)}"

API_KEY, AI_CONFIGURED, CONFIG_MESSAGE = configure_google_ai()

@st.cache_resource
def get_ai_models():
    """Initialize and cache AI models"""
    try:
        if not AI_CONFIGURED or not API_KEY:
            return None, None, None, False
        
        gemini_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=API_KEY
        )
        
        gemini_vision = genai.GenerativeModel('gemini-2.0-flash')
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=API_KEY
        )
        
        return gemini_model, gemini_vision, embeddings, True
    except Exception as e:
        st.error(f"Error initializing AI models: {str(e)}")
        return None, None, None, False

# ==================== AUTH MANAGER ====================

class ModernAuthManager:
    def __init__(self):
        self.users_file = "users_db.json"
        self.ensure_users_file()
    
    def ensure_users_file(self):
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def hash_password(self, password):
        salt = "elite_ai_platform_2025"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def register_user(self, username, password, email, full_name):
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                return False, "Username already exists!"
            
            if '@' not in email or '.' not in email:
                return False, "Please enter a valid email address!"
            
            users[username] = {
                "password": self.hash_password(password),
                "email": email,
                "full_name": full_name,
                "created_at": datetime.now().isoformat(),
                "login_count": 0,
                "document_queries": 0,
                "image_queries": 0,
                "last_login": None
            }
            
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=4)
            
            return True, "Account created successfully!"
        except Exception as e:
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, username, password):
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username not in users:
                return False, "Username not found!"
            
            if users[username]["password"] != self.hash_password(password):
                return False, "Invalid password!"
            
            users[username]["login_count"] += 1
            users[username]["last_login"] = datetime.now().isoformat()
            
            with open(self.users_file, 'w') as f:
                json.dump(users, f, indent=4)
            
            return True, f"Welcome back, {users[username]['full_name']}!"
        except Exception as e:
            return False, f"Login failed: {str(e)}"
    
    def get_user_info(self, username):
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            return users.get(username, {})
        except:
            return {}
    
    def update_user_stats(self, username, stat_type):
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users and stat_type in users[username]:
                users[username][stat_type] += 1
                
                with open(self.users_file, 'w') as f:
                    json.dump(users, f, indent=4)
        except Exception as e:
            print(f"Error updating user stats: {str(e)}")

# ==================== MAIN PLATFORM ====================

class EliteDocumentImagePlatform:
    def __init__(self):
        self.auth_manager = ModernAuthManager()
        self.initialize_session_state()
        self.gemini_model, self.gemini_vision, self.embeddings, self.models_ready = get_ai_models()
        self.apply_professional_css()
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'authenticated': False,
            'username': '',
            'current_page': 'documents',
            'document_messages': [],
            'image_messages': [],
            'current_image': None,
            'document_vector_store': None,
            'processed_files': [],
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def apply_professional_css(self):
        """Apply professional dark theme CSS"""
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600&display=swap');
        
        :root {
            --primary-bg: #0a0a0a;
            --secondary-bg: #111111;
            --tertiary-bg: #1a1a1a;
            --accent-bg: #222222;
            --border-color: #2a2a2a;
            --text-primary: #ffffff;
            --text-secondary: #e0e0e0;
            --text-muted: #a0a0a0;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --accent-green: #10b981;
            --accent-orange: #f59e0b;
            --accent-red: #ef4444;
            --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            --gradient-accent: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
            --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
            --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
            --border-radius: 12px;
        }
        
        /* Global Styles */
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--primary-bg) !important;
            color: var(--text-primary) !important;
            line-height: 1.6;
        }
        
        .stApp {
            background: var(--primary-bg) !important;
        }
        
        /* Hide Default Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        header[data-testid="stHeader"] {visibility: hidden;}
        
        /* Professional Header */
        .elite-header {
            background: var(--gradient-primary);
            padding: 3rem 2rem;
            border-radius: 0 0 24px 24px;
            text-align: center;
            margin: -1rem -1rem 3rem -1rem;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }
        
        .elite-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: radial-gradient(circle at 30% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
            pointer-events: none;
        }
        
        .elite-header h1 {
            color: #ffffff;
            font-size: 3.2rem;
            font-weight: 800;
            margin: 0;
            text-shadow: 2px 4px 8px rgba(0,0,0,0.3);
            position: relative;
            z-index: 2;
        }
        
        .elite-header p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.3rem;
            margin: 1rem 0 0 0;
            font-weight: 500;
            position: relative;
            z-index: 2;
        }
        
        /* Professional Buttons */
        .stButton > button {
            background: var(--tertiary-bg) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            padding: 0.875rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: var(--shadow-sm) !important;
            backdrop-filter: blur(10px) !important;
            position: relative !important;
            overflow: hidden !important;
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
            transition: left 0.5s;
        }
        
        .stButton > button:hover {
            background: var(--accent-bg) !important;
            border-color: var(--accent-blue) !important;
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-md) !important;
            color: var(--accent-blue) !important;
        }
        
        .stButton > button:hover::before {
            left: 100%;
        }
        
        /* Navigation Buttons Special Styling */
        .nav-button {
            background: var(--gradient-accent) !important;
            color: white !important;
            border: none !important;
        }
        
        .nav-button:hover {
            transform: translateY(-3px) scale(1.05) !important;
            box-shadow: var(--shadow-lg) !important;
        }
        
        /* Forms */
        .stTextInput > div > div > input {
            background: var(--tertiary-bg) !important;
            color: var(--text-primary) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            padding: 1rem 1.25rem !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--accent-blue) !important;
            box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
            outline: none !important;
            background: var(--secondary-bg) !important;
        }
        
        .stTextInput > div > div > input::placeholder {
            color: var(--text-muted) !important;
        }
        
        /* File Uploader */
        .stFileUploader > div {
            background: var(--tertiary-bg) !important;
            color: var(--text-primary) !important;
            border: 3px dashed var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            padding: 3rem 2rem !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stFileUploader > div:hover {
            border-color: var(--accent-blue) !important;
            background: var(--secondary-bg) !important;
            transform: scale(1.02) !important;
            box-shadow: var(--shadow-md) !important;
        }
        
        /* Chat Messages */
        .stChatMessage {
            background: var(--tertiary-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            color: var(--text-primary) !important;
            margin: 1rem 0 !important;
            box-shadow: var(--shadow-sm) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stChatMessage[data-testid="user-message"] {
            background: linear-gradient(135deg, var(--accent-blue) 0%, #4f46e5 100%) !important;
            border-color: var(--accent-blue) !important;
        }
        
        .stChatMessage[data-testid="assistant-message"] {
            background: var(--secondary-bg) !important;
            border-color: var(--border-color) !important;
        }
        
        /* Success/Error/Info Messages */
        .stSuccess {
            background: linear-gradient(135deg, var(--accent-green), #059669) !important;
            color: white !important;
            border-radius: var(--border-radius) !important;
            border: none !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .stError {
            background: linear-gradient(135deg, var(--accent-red), #dc2626) !important;
            color: white !important;
            border-radius: var(--border-radius) !important;
            border: none !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .stInfo {
            background: linear-gradient(135deg, var(--accent-blue), #2563eb) !important;
            color: white !important;
            border-radius: var(--border-radius) !important;
            border: none !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .stWarning {
            background: linear-gradient(135deg, var(--accent-orange), #d97706) !important;
            color: white !important;
            border-radius: var(--border-radius) !important;
            border: none !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        /* Metrics */
        .stMetric {
            background: var(--tertiary-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            padding: 1.5rem !important;
            text-align: center !important;
            transition: all 0.3s ease !important;
            box-shadow: var(--shadow-sm) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stMetric:hover {
            transform: translateY(-4px) !important;
            box-shadow: var(--shadow-md) !important;
            border-color: var(--accent-blue) !important;
        }
        
        /* Expanders */
        .stExpander {
            background: var(--tertiary-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            backdrop-filter: blur(10px) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .stExpander > div > div {
            background: var(--tertiary-bg) !important;
            color: var(--text-primary) !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: var(--secondary-bg);
            padding: 0.5rem;
            border-radius: var(--border-radius);
            border: 1px solid var(--border-color);
        }
        
        .stTabs [data-baseweb="tab"] {
            background: var(--tertiary-bg) !important;
            color: var(--text-secondary) !important;
            border-radius: var(--border-radius) !important;
            padding: 0.875rem 2rem !important;
            font-weight: 600 !important;
            border: 1px solid var(--border-color) !important;
            transition: all 0.3s ease !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: var(--accent-bg) !important;
            color: var(--text-primary) !important;
            transform: translateY(-2px) !important;
            box-shadow: var(--shadow-sm) !important;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--gradient-primary) !important;
            color: white !important;
            border-color: var(--accent-blue) !important;
            transform: scale(1.05) !important;
            box-shadow: var(--shadow-md) !important;
        }
        
        /* Text Elements */
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
        }
        
        p, div, span, li {
            color: var(--text-primary) !important;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: var(--secondary-bg);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, var(--accent-purple), var(--accent-blue));
        }
        
        /* Professional Cards */
        .feature-card {
            background: var(--tertiary-bg);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 2rem;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            box-shadow: var(--shadow-sm);
        }
        
        .feature-card:hover {
            transform: translateY(-8px) scale(1.02);
            box-shadow: var(--shadow-lg);
            border-color: var(--accent-blue);
        }
        
        /* Loading Animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .loading {
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .elite-header h1 {
                font-size: 2.5rem;
            }
            
            .elite-header p {
                font-size: 1.1rem;
            }
            
            .stButton > button {
                padding: 0.75rem 1.5rem !important;
                font-size: 0.9rem !important;
            }
        }
        
        /* Dark Mode Enhancements */
        .stSelectbox > div > div {
            background: var(--tertiary-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
        }
        
        .stMultiSelect > div > div {
            background: var(--tertiary-bg) !important;
            border: 1px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
        }
        
        /* Chat Input Enhancements */
        .stChatInputContainer {
            background: var(--tertiary-bg) !important;
            border: 2px solid var(--border-color) !important;
            border-radius: var(--border-radius) !important;
            backdrop-filter: blur(10px) !important;
        }
        
        .stChatInput > div > div > input {
            background: transparent !important;
            color: var(--text-primary) !important;
            border: none !important;
        }
        
        .stChatInput > div > div > input::placeholder {
            color: var(--text-muted) !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_landing_page(self):
        """Render professional landing page"""
        st.markdown("""
        <div class="elite-header">
            <h1>🧠 Elite Document & Image AI Platform</h1>
            <p>Advanced AI-Powered Document Processing • Computer Vision Intelligence • Professional Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            tab1, tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])
            
            with tab1:
                self.render_login_form()
            
            with tab2:
                self.render_register_form()

    def render_login_form(self):
        """Render professional login form"""
        st.markdown("### 🔐 Welcome Back")
        st.markdown("*Access your AI-powered document and image analysis platform*")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🚀 Sign In", use_container_width=True)
            
            if submitted:
                if username and password:
                    success, message = self.auth_manager.login_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success(message)
                        if self.models_ready:
                            st.success("🤖 AI Models: Initialized and Ready!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all required fields!")

    def render_register_form(self):
        """Render professional registration form"""
        st.markdown("### ✨ Join Elite Platform")
        st.markdown("*Create your account and unlock AI-powered document and image analytics*")
        
        with st.form("register_form"):
            full_name = st.text_input("👤 Full Name", placeholder="Enter your full name")
            username = st.text_input("🧑‍💼 Username", placeholder="Choose a unique username")
            email = st.text_input("📧 Email Address", placeholder="Enter your email address")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a secure password")
            confirm_password = st.text_input("🔐 Confirm Password", type="password", placeholder="Confirm your password")
            submitted = st.form_submit_button("✨ Create Account", use_container_width=True)
            
            if submitted:
                if all([full_name, username, email, password, confirm_password]):
                    if password != confirm_password:
                        st.error("Passwords don't match!")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters long!")
                    elif len(username) < 3:
                        st.error("Username must be at least 3 characters long!")
                    else:
                        success, message = self.auth_manager.register_user(username, password, email, full_name)
                        if success:
                            st.success(message)
                            st.info("🔑 Please sign in with your new account!")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all required fields!")

    def render_dashboard(self):
        """Render main professional dashboard"""
        user_info = self.auth_manager.get_user_info(st.session_state.username)
        
        st.markdown(f"""
        <div class="elite-header">
            <h1>👋 Welcome, {user_info.get('full_name', st.session_state.username)}</h1>
            <p>Professional AI Dashboard • Document Intelligence • Computer Vision Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Professional Navigation with only Documents and Images
        st.markdown("### 🚀 AI Analytics Modules")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("📄 Document Intelligence", use_container_width=True, key="nav_docs"):
                st.session_state.current_page = 'documents'
                st.rerun()
        
        with col2:
            if st.button("🖼️ Image Intelligence", use_container_width=True, key="nav_images"):
                st.session_state.current_page = 'images'
                st.rerun()
        
        with col3:
            col_stats, col_logout = st.columns([1, 1])
            with col_stats:
                # User stats
                doc_queries = user_info.get('document_queries', 0)
                img_queries = user_info.get('image_queries', 0)
                st.metric("📊 Total Queries", f"{doc_queries + img_queries}")
            
            with col_logout:
                if st.button("🚪 Logout", use_container_width=True):
                    self.logout_user()
        
        # Feature Overview Cards
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>📄 Document Intelligence</h3>
                <p>Upload PDFs and Word documents for AI-powered analysis, summarization, and intelligent Q&A.</p>
                <strong>Features:</strong>
                <ul style="text-align: left; padding-left: 20px;">
                    <li>Multi-format document processing</li>
                    <li>Intelligent content extraction</li>
                    <li>Contextual Q&A with continuous chat</li>
                    <li>Advanced text analysis</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🖼️ Image Intelligence</h3>
                <p>Upload images for computer vision analysis, OCR text extraction, and visual understanding.</p>
                <strong>Features:</strong>
                <ul style="text-align: left; padding-left: 20px;">
                    <li>Advanced computer vision</li>
                    <li>OCR text extraction</li>
                    <li>Visual content analysis</li>
                    <li>Intelligent image Q&A</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Render current page
        if st.session_state.current_page == 'documents':
            self.render_documents_page()
        elif st.session_state.current_page == 'images':
            self.render_images_page()

    def render_documents_page(self):
        """Render Document Intelligence page with perfect continuous chat"""
        st.markdown("## 📄 Document Intelligence Center")
        st.markdown("*Upload and analyze your documents with advanced AI - Perfect continuous chat experience*")
        
        if not self.models_ready:
            st.error("❌ AI models not available. Please check your Google API key configuration.")
            return
        
        # Enhanced file upload section
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            uploaded_pdfs = st.file_uploader(
                "📄 Upload PDF Documents", 
                type="pdf", 
                accept_multiple_files=True, 
                key="pdf_docs",
                help="Upload PDF files for comprehensive AI analysis"
            )
        
        with col2:
            uploaded_docs = st.file_uploader(
                "📝 Upload Word Documents", 
                type=["docx", "doc"], 
                accept_multiple_files=True, 
                key="word_docs",
                help="Upload Word documents for intelligent processing"
            )
        
        with col3:
            if st.button("🗑️ Clear Chat", key="clear_docs", use_container_width=True):
                st.session_state.document_messages = []
                st.success("✅ Chat history cleared!")
                time.sleep(1)
                st.rerun()
        
        # Process documents with enhanced feedback
        if (uploaded_pdfs or uploaded_docs):
            if st.button("🚀 Process Documents with AI", use_container_width=True):
                with st.spinner("🧠 Processing documents with advanced AI algorithms..."):
                    success = self.process_documents(uploaded_pdfs, uploaded_docs)
                    if success:
                        st.success("✅ Documents processed successfully! Ready for intelligent Q&A.")
                    else:
                        st.error("❌ Error processing documents. Please try again or check file formats.")
        
        # Show processed files with professional display
        if st.session_state.processed_files:
            with st.expander("📚 Processed Documents Overview", expanded=True):
                for i, file_info in enumerate(st.session_state.processed_files, 1):
                    st.markdown(f"**{i}.** {file_info}")
        
        # PERFECT CONTINUOUS CHAT FOR DOCUMENTS
        if st.session_state.document_vector_store is not None:
            st.markdown("### 💬 Intelligent Document Chat")
            st.markdown("*Ask anything about your documents - Advanced AI understands context and provides detailed insights*")
            
            # Display chat history with professional styling
            for message in st.session_state.document_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Enhanced chat input
            if prompt := st.chat_input("🔍 Ask detailed questions about your documents...", key="doc_chat"):
                # Add user message to chat history
                st.session_state.document_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate professional AI response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 AI is analyzing your documents with advanced intelligence..."):
                        response = self.process_document_chat(prompt)
                        
                        # Display comprehensive response
                        st.markdown(response)
                        
                        # Store complete message in chat history
                        st.session_state.document_messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        
                        # Update user statistics
                        self.auth_manager.update_user_stats(st.session_state.username, "document_queries")
            
            # Professional example questions
            with st.expander("💡 Professional Document Analysis Examples", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📋 Content Analysis:**")
                    content_examples = [
                        "Provide a comprehensive summary of all documents",
                        "What are the main themes and topics discussed?", 
                        "Extract key findings and conclusions",
                        "Identify important dates, numbers, and facts",
                        "What recommendations are mentioned?",
                        "Analyze the document structure and organization"
                    ]
                    for example in content_examples:
                        if st.button(f"📄 {example}", key=f"doc_content_{hash(example)}", use_container_width=True):
                            st.session_state.document_messages.append({"role": "user", "content": example})
                            response = self.process_document_chat(example)
                            st.session_state.document_messages.append({"role": "assistant", "content": response})
                            st.rerun()
                
                with col2:
                    st.markdown("**🔍 Advanced Analysis:**")
                    advanced_examples = [
                        "Compare arguments and viewpoints across documents",
                        "Who are the key stakeholders mentioned?",
                        "What are the potential risks and opportunities?",
                        "Provide actionable insights and next steps",
                        "Analyze sentiment and tone throughout documents",
                        "Create an executive summary with key takeaways"
                    ]
                    for example in advanced_examples:
                        if st.button(f"🔬 {example}", key=f"doc_advanced_{hash(example)}", use_container_width=True):
                            st.session_state.document_messages.append({"role": "user", "content": example})
                            response = self.process_document_chat(example)
                            st.session_state.document_messages.append({"role": "assistant", "content": response})
                            st.rerun()

    def render_images_page(self):
        """Render Image Intelligence page with perfect continuous chat"""
        st.markdown("## 🖼️ Computer Vision Intelligence Center")
        st.markdown("*Upload and analyze images with advanced AI vision - Perfect continuous chat experience*")
        
        if not self.models_ready:
            st.error("❌ AI models not available. Please check your Google API key configuration.")
            return
        
        # Enhanced image upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_image = st.file_uploader(
                "🖼️ Upload Image for Advanced AI Analysis", 
                type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                help="Upload any image format for comprehensive computer vision analysis"
            )
        
        with col2:
            if st.button("🗑️ Clear Chat", key="clear_images", use_container_width=True):
                st.session_state.image_messages = []
                st.success("✅ Image chat history cleared!")
                time.sleep(1)
                st.rerun()
        
        # Process image with professional handling
        if uploaded_image:
            if st.button("🚀 Process Image with AI Vision", use_container_width=True):
                try:
                    image = Image.open(uploaded_image)
                    
                    # Professional image validation and optimization
                    if image.size[0] > 10000 or image.size[1] > 10000:
                        st.warning("⚠️ Large image detected. Optimizing for enhanced processing...")
                        image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
                    
                    st.session_state.current_image = image
                    st.success(f"✅ Image processed successfully: **{uploaded_image.name}**")
                    
                    # Professional image information display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📏 Width", f"{image.size[0]} px")
                    with col2:
                        st.metric("📐 Height", f"{image.size[1]} px")
                    with col3:
                        st.metric("🎨 Mode", image.mode)
                    
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")

        # Professional image display and analysis tools
        if st.session_state.current_image is not None:
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.markdown("### 🖼️ Current Image Analysis")
                st.image(st.session_state.current_image, use_container_width=True, caption="AI-Ready Image")
            
            with col2:
                st.markdown("### 🔧 Quick AI Actions")
                
                if st.button("📝 Advanced OCR Text Extraction", use_container_width=True):
                    with st.spinner("🔍 Extracting text with advanced OCR..."):
                        response = self.analyze_image_professional("Perform comprehensive OCR text extraction from this image. Extract all visible text with high accuracy, maintaining formatting and structure where possible.")
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()
                
                if st.button("🔍 Comprehensive Visual Analysis", use_container_width=True):
                    with st.spinner("🧠 Performing comprehensive image analysis..."):
                        response = self.analyze_image_professional()
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()
                
                if st.button("🎨 Design & Aesthetic Analysis", use_container_width=True):
                    with st.spinner("🎨 Analyzing design elements..."):
                        response = self.analyze_image_professional("Provide a comprehensive analysis of the design elements, visual composition, color theory, lighting, aesthetic appeal, and artistic aspects of this image.")
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()
                
                if st.button("🔬 Technical Image Analysis", use_container_width=True):
                    with st.spinner("🔬 Performing technical analysis..."):
                        response = self.analyze_image_professional("Perform a technical analysis of this image including quality assessment, resolution details, potential compression artifacts, color accuracy, and technical specifications.")
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()
            
            # PERFECT CONTINUOUS CHAT FOR IMAGES
            st.markdown("### 💬 Intelligent Image Chat")
            st.markdown("*Ask anything about your image - Advanced computer vision AI provides detailed visual insights*")
            
            # Display chat history with professional styling
            for message in st.session_state.image_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Enhanced chat input for images
            if prompt := st.chat_input("🔍 Ask detailed questions about your image...", key="image_chat"):
                # Add user message to chat history
                st.session_state.image_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate professional AI response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 AI vision is analyzing your image with advanced computer vision..."):
                        response = self.analyze_image_professional(prompt)
                        
                        # Display comprehensive response
                        st.markdown(response)
                        
                        # Store complete message in chat history
                        st.session_state.image_messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        
                        # Update user statistics
                        self.auth_manager.update_user_stats(st.session_state.username, "image_queries")
            
            # Professional example questions for images
            with st.expander("💡 Professional Image Analysis Examples", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**👁️ Visual Analysis:**")
                    visual_examples = [
                        "Describe all objects and people in detail",
                        "Analyze the setting, environment, and context", 
                        "What story does this image tell?",
                        "Identify all colors and their psychological impact",
                        "Analyze the composition and visual flow",
                        "Describe the mood and emotional impact"
                    ]
                    for example in visual_examples:
                        if st.button(f"👁️ {example}", key=f"img_visual_{hash(example)}", use_container_width=True):
                            st.session_state.image_messages.append({"role": "user", "content": example})
                            response = self.analyze_image_professional(example)
                            st.session_state.image_messages.append({"role": "assistant", "content": response})
                            st.rerun()
                
                with col2:
                    st.markdown("**📝 Content & Technical:**")
                    technical_examples = [
                        "Extract and transcribe all visible text accurately",
                        "Identify all brands, logos, and signage",
                        "Analyze the technical quality and specifications", 
                        "Describe the lighting and photographic technique",
                        "Identify potential safety or compliance issues",
                        "Provide recommendations for image improvement"
                    ]
                    for example in technical_examples:
                        if st.button(f"📖 {example}", key=f"img_technical_{hash(example)}", use_container_width=True):
                            st.session_state.image_messages.append({"role": "user", "content": example})
                            response = self.analyze_image_professional(example)
                            st.session_state.image_messages.append({"role": "assistant", "content": response})
                            st.rerun()

    # ==================== PROCESSING METHODS ====================
    
    def process_documents(self, uploaded_pdfs, uploaded_docs):
        """Process uploaded documents with enhanced error handling"""
        try:
            all_text = ""
            processed_files = []
            
            # Process PDFs with comprehensive extraction
            if uploaded_pdfs:
                for pdf_file in uploaded_pdfs:
                    try:
                        pdf_reader = PdfReader(pdf_file)
                        pdf_text = ""
                        
                        for page_num, page in enumerate(pdf_reader.pages):
                            page_text = page.extract_text()
                            if page_text.strip():
                                pdf_text += f"\n--- Page {page_num + 1} of {pdf_file.name} ---\n{page_text}\n"
                        
                        if pdf_text.strip():
                            all_text += pdf_text
                            processed_files.append(f"📄 **{pdf_file.name}** ({len(pdf_reader.pages)} pages) - PDF Document")
                        
                    except Exception as e:
                        st.error(f"❌ Error reading PDF {pdf_file.name}: {str(e)}")
            
            # Process Word documents with comprehensive extraction
            if uploaded_docs:
                for doc_file in uploaded_docs:
                    try:
                        doc = docx.Document(doc_file)
                        doc_text = ""
                        para_count = 0
                        
                        # Extract paragraphs
                        for paragraph in doc.paragraphs:
                            if paragraph.text.strip():
                                doc_text += paragraph.text + "\n"
                                para_count += 1
                        
                        # Extract tables
                        table_count = 0
                        for table in doc.tables:
                            table_count += 1
                            doc_text += f"\n--- Table {table_count} ---\n"
                            for row in table.rows:
                                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                                if row_text.strip():
                                    doc_text += row_text + "\n"
                        
                        if doc_text.strip():
                            all_text += f"\n--- {doc_file.name} ---\n{doc_text}\n"
                            processed_files.append(f"📝 **{doc_file.name}** ({para_count} paragraphs, {table_count} tables) - Word Document")
                        
                    except Exception as e:
                        st.error(f"❌ Error reading Word document {doc_file.name}: {str(e)}")
            
            # Create enhanced vector store if we have content
            if all_text.strip():
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=12000,
                    chunk_overlap=1500,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                text_chunks = text_splitter.split_text(all_text)
                
                if text_chunks:
                    vector_store = FAISS.from_texts(text_chunks, self.embeddings)
                    st.session_state.document_vector_store = vector_store
                    st.session_state.processed_files = processed_files
                    return True
            
            return False
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            return False

    def process_document_chat(self, question):
        """Process document chat with enhanced AI analysis"""
        if not self.models_ready:
            return "❌ AI models not available for document analysis. Please check configuration."
        
        try:
            # Retrieve relevant document sections
            docs = st.session_state.document_vector_store.similarity_search(question, k=6)
            
            if not docs:
                return "❌ No relevant information found in the documents for your question. Please try rephrasing or ask about different topics."
            
            # Enhanced professional prompt template
            prompt_template = """
            You are an expert document analyst with advanced comprehension capabilities. Analyze the provided document context thoroughly and provide a comprehensive, professional response.
            
            Document Context:
            {context}
            
            User Question: {question}
            
            Instructions:
            - Provide a detailed, well-structured analysis
            - Use specific examples and quotes from the documents when relevant
            - Organize your response with clear headings and bullet points where appropriate
            - If the question requires synthesis across multiple documents, provide comparative insights
            - Include actionable recommendations when applicable
            - Maintain professional tone and comprehensive coverage
            
            Professional Analysis:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create and execute enhanced analysis chain
            chain = load_qa_chain(self.gemini_model, chain_type="stuff", prompt=prompt)
            
            response = chain.invoke(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )
            
            return f"📚 **Professional Document Analysis:**\n\n{response['output_text']}"
            
        except Exception as e:
            return f"❌ Error analyzing documents: {str(e)}. Please try rephrasing your question or check document processing."

    def analyze_image_professional(self, question=""):
        """Enhanced image analysis with professional computer vision"""
        if st.session_state.current_image is None:
            return "❌ No image uploaded. Please upload an image first for analysis."
        
        if not self.models_ready:
            return "❌ AI vision models not available for image analysis. Please check configuration."
        
        try:
            image = st.session_state.current_image
            
            if question:
                prompt = f"""
                You are an expert computer vision analyst with advanced image understanding capabilities. 
                Analyze this image comprehensively and answer: {question}
                
                Provide a detailed, professional response including:
                1. Direct and comprehensive answer to the specific question
                2. Supporting visual evidence and observations
                3. Technical details and specifications when relevant
                4. Any visible text, numbers, or textual information
                5. Contextual insights and professional recommendations
                6. Confidence levels and potential limitations of analysis
                
                Maintain a professional, detailed, and insightful analytical approach.
                """
            else:
                prompt = """
                You are an expert computer vision analyst. Provide a comprehensive, professional analysis of this image.
                
                Include detailed analysis of:
                
                **Visual Content & Objects:**
                - All visible objects, people, animals, and items
                - Spatial relationships and arrangements
                - Actions, activities, and interactions
                
                **Technical & Quality Aspects:**
                - Image quality, resolution, and technical specifications
                - Lighting conditions, shadows, and illumination
                - Color palette, saturation, and visual characteristics
                - Photographic technique and composition
                
                **Textual Information:**
                - All visible text, signs, labels, and written content
                - Numbers, dates, and numerical information
                - Brands, logos, and identifying marks
                
                **Contextual Analysis:**
                - Setting, location, and environmental context
                - Time period indicators and cultural elements
                - Purpose and intended use of the image
                - Emotional tone and aesthetic qualities
                
                **Professional Insights:**
                - Potential applications and use cases
                - Notable features and unique characteristics
                - Recommendations for optimization or improvement
                
                Provide a thorough, structured, and professional analysis.
                """
            
            response = self.gemini_vision.generate_content([prompt, image])
            return f"🖼️ **Expert Computer Vision Analysis:**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ Error analyzing image: {str(e)}. Please try uploading the image again or check the file format."

    def logout_user(self):
        """Enhanced logout with professional messaging"""
        username = st.session_state.get('username', 'User')
        user_info = self.auth_manager.get_user_info(username)
        
        st.success(f"👋 Thank you for using Elite Document & Image AI Platform, {user_info.get('full_name', username)}!")
        st.info("🔒 Your session has been securely logged out. All data has been cleared.")
        
        # Clear all session state securely
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        time.sleep(3)
        st.rerun()

    def run(self):
        """Run the professional application"""
        try:
            if not st.session_state.authenticated:
                self.render_landing_page()
            else:
                self.render_dashboard()
        except Exception as e:
            st.error(f"❌ Application Error: {str(e)}")
            with st.expander("🔧 Technical Details"):
                st.code(traceback.format_exc())

# ==================== MAIN APPLICATION ENTRY POINT ====================

if __name__ == "__main__":
    try:
        app = EliteDocumentImagePlatform()
        app.run()
    except Exception as e:
        st.error(f"❌ Critical Application Error: {str(e)}")
        st.info("🔄 Please refresh the page and try again.")
        with st.expander("🛠️ Debug Information"):
            st.code(traceback.format_exc())
