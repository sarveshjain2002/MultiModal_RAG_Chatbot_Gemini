import streamlit as st
import os
import time
import json
import hashlib
import pandas as pd
import sqlite3
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
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import nest_asyncio
import re
import csv
import io
import chardet
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Fix for event loop issues
nest_asyncio.apply()
load_dotenv()

# Set page config ONLY ONCE at the very beginning
st.set_page_config(
    page_title="Elite MultiModal AI Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSV Handler Class
class PerfectCSVHandler:
    """Perfect CSV Handler that can handle any CSV format, encoding, and structure"""
    
    def __init__(self):
        self.supported_encodings = [
            'utf-8', 'utf-8-sig', 'latin-1', 'iso-8859-1', 
            'cp1252', 'ascii', 'utf-16', 'utf-32'
        ]
        self.supported_delimiters = [',', ';', '\t', '|', ' ', ':', '-']
        
    def detect_encoding(self, file_content: bytes) -> str:
        """Detect file encoding using chardet with fallbacks"""
        try:
            # Try chardet first
            detection = chardet.detect(file_content)
            if detection['encoding'] and detection['confidence'] > 0.7:
                return detection['encoding']
        except:
            pass
        
        # Fallback to common encodings
        for encoding in self.supported_encodings:
            try:
                file_content.decode(encoding)
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        return 'utf-8'  # Final fallback
    
    def detect_delimiter(self, sample_text: str) -> str:
        """Detect CSV delimiter using multiple methods"""
        try:
            # Method 1: Use CSV Sniffer
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample_text, delimiters=',;\t|').delimiter
            return delimiter
        except:
            pass
        
        # Method 2: Count delimiter frequency
        delimiter_counts = {}
        for delimiter in self.supported_delimiters:
            delimiter_counts[delimiter] = sample_text.count(delimiter)
        
        # Return the most frequent delimiter
        if delimiter_counts:
            return max(delimiter_counts, key=delimiter_counts.get)
        
        return ','  # Default fallback
    
    def smart_csv_reader(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
        """Intelligent CSV reader with multiple fallback strategies"""
        messages = []
        
        try:
            # Step 1: Read file content and detect encoding
            file_content = uploaded_file.read()
            uploaded_file.seek(0)  # Reset file pointer
            
            detected_encoding = self.detect_encoding(file_content)
            messages.append(f"✅ Detected encoding: {detected_encoding}")
            
            # Step 2: Get sample text for delimiter detection
            try:
                sample_text = file_content[:8192].decode(detected_encoding)
                detected_delimiter = self.detect_delimiter(sample_text)
                messages.append(f"✅ Detected delimiter: '{detected_delimiter}'")
            except:
                detected_delimiter = ','
                messages.append(f"⚠️ Using default delimiter: ','")
            
            # Step 3: Multiple reading strategies
            read_strategies = [
                # Strategy 1: Use detected parameters
                {
                    'encoding': detected_encoding,
                    'sep': detected_delimiter,
                    'engine': 'python',
                    'on_bad_lines': 'skip'
                },
                # Strategy 2: Auto-detect with pandas
                {
                    'encoding': detected_encoding,
                    'sep': None,
                    'engine': 'python',
                    'on_bad_lines': 'skip'
                },
                # Strategy 3: UTF-8 with common delimiters
                {
                    'encoding': 'utf-8',
                    'sep': ',',
                    'engine': 'python',
                    'on_bad_lines': 'skip'
                },
                # Strategy 4: Latin-1 fallback
                {
                    'encoding': 'latin-1',
                    'sep': detected_delimiter,
                    'engine': 'python',
                    'on_bad_lines': 'skip'
                },
                # Strategy 5: Robust fallback
                {
                    'encoding': 'utf-8',
                    'sep': ',',
                    'engine': 'python',
                    'on_bad_lines': 'skip',
                    'quoting': csv.QUOTE_MINIMAL,
                    'skipinitialspace': True
                }
            ]
            
            # Try each strategy
            for i, strategy in enumerate(read_strategies, 1):
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, **strategy)
                    
                    if df is not None and not df.empty and len(df.columns) > 0:
                        messages.append(f"✅ Successfully read using strategy {i}")
                        
                        # Post-processing cleanup
                        df = self.cleanup_dataframe(df)
                        
                        # Validate the dataframe
                        if self.validate_dataframe(df):
                            final_message = "\n".join(messages)
                            return df, final_message
                        
                except Exception as e:
                    messages.append(f"❌ Strategy {i} failed: {str(e)[:100]}")
                    continue
            
            # If all strategies fail, try manual parsing
            uploaded_file.seek(0)
            df, manual_message = self.manual_csv_parse(uploaded_file, detected_encoding)
            if df is not None:
                messages.append(manual_message)
                final_message = "\n".join(messages)
                return df, final_message
            
            return None, "❌ All reading strategies failed. Please check your CSV file format."
            
        except Exception as e:
            return None, f"❌ Critical error reading CSV: {str(e)}"
    
    def manual_csv_parse(self, uploaded_file, encoding: str) -> Tuple[Optional[pd.DataFrame], str]:
        """Manual CSV parsing as last resort"""
        try:
            uploaded_file.seek(0)
            content = uploaded_file.read().decode(encoding, errors='replace')
            lines = content.strip().split('\n')
            
            if len(lines) < 2:
                return None, "❌ File has insufficient data"
            
            # Try to parse manually
            delimiter = self.detect_delimiter(lines[0])
            
            data = []
            headers = lines[0].split(delimiter)
            headers = [h.strip().strip('"').strip("'") for h in headers]
            
            for line in lines[1:]:
                if line.strip():
                    row = line.split(delimiter)
                    row = [cell.strip().strip('"').strip("'") for cell in row]
                    # Pad or truncate to match header length
                    while len(row) < len(headers):
                        row.append('')
                    data.append(row[:len(headers)])
            
            df = pd.DataFrame(data, columns=headers)
            df = self.cleanup_dataframe(df)
            
            return df, "✅ Manual parsing successful"
            
        except Exception as e:
            return None, f"❌ Manual parsing failed: {str(e)}"
    
    def cleanup_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and optimize dataframe"""
        try:
            # Remove completely empty rows and columns
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            # Clean column names
            df.columns = df.columns.astype(str)
            df.columns = [col.strip().replace('\n', ' ').replace('\r', '') for col in df.columns]
            
            # Handle unnamed columns
            unnamed_cols = [col for col in df.columns if 'Unnamed:' in str(col)]
            for col in unnamed_cols:
                if df[col].isna().all():
                    df = df.drop(columns=[col])
                else:
                    df = df.rename(columns={col: f'Column_{col.split(":")[-1]}'})
            
            # Convert data types intelligently
            df = self.smart_type_conversion(df)
            
            # Remove completely duplicate rows
            df = df.drop_duplicates()
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            print(f"Cleanup error: {str(e)}")
            return df
    
    def smart_type_conversion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Intelligent data type conversion"""
        for col in df.columns:
            try:
                # Skip if all values are null
                if df[col].isna().all():
                    continue
                
                # Try numeric conversion
                if df[col].dtype == 'object':
                    # Check if it's numeric
                    sample_values = df[col].dropna().astype(str).str.replace(',', '').str.strip()
                    
                    # Try integer conversion
                    try:
                        numeric_sample = pd.to_numeric(sample_values, errors='coerce')
                        if not numeric_sample.isna().all():
                            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
                            continue
                    except:
                        pass
                    
                    # Try datetime conversion
                    try:
                        if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            continue
                    except:
                        pass
                
            except Exception as e:
                continue  # Keep original type if conversion fails
        
        return df
    
    def validate_dataframe(self, df: pd.DataFrame) -> bool:
        """Validate that dataframe is properly loaded"""
        try:
            if df is None or df.empty:
                return False
            
            if len(df.columns) == 0:
                return False
            
            # Check if we have at least some non-null data
            if df.isna().all().all():
                return False
            
            # Check for reasonable column names
            if all(str(col).startswith('Unnamed') for col in df.columns):
                return False
            
            return True
            
        except:
            return False

# Configure Google AI with robust key handling
def configure_google_ai():
    """Configure Google AI with robust key handling"""
    try:
        api_key = None
        
        # Method 1: Try Streamlit secrets
        try:
            if hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
                api_key = st.secrets["GOOGLE_API_KEY"].strip()
                if api_key and len(api_key) > 10:
                    print("✅ API key loaded from Streamlit secrets")
        except Exception as e:
            print(f"Secrets error: {str(e)}")
        
        # Method 2: Try environment variables
        if not api_key:
            try:
                env_key = os.getenv("GOOGLE_API_KEY")
                if env_key and len(env_key.strip()) > 10:
                    api_key = env_key.strip()
                    print("✅ API key loaded from environment")
            except Exception as e:
                print(f"Environment error: {str(e)}")
        
        # Validate and configure
        if not api_key:
            raise Exception("Google API Key not found. Please add GOOGLE_API_KEY to Streamlit secrets.")
        
        # Clean the API key
        api_key = api_key.strip().strip('"').strip("'")
        
        if len(api_key) < 20:
            raise Exception("Invalid Google API Key format. Key seems too short.")
        
        genai.configure(api_key=api_key)
        return api_key, True, "Google AI configured successfully!"
        
    except Exception as e:
        error_msg = f"Google AI configuration error: {str(e)}"
        print(error_msg)
        return None, False, error_msg

# Configure Google AI on startup
API_KEY, AI_CONFIGURED, CONFIG_MESSAGE = configure_google_ai()

# Global AI Models - Initialize once and reuse
@st.cache_resource
def get_ai_models():
    """Initialize and cache AI models"""
    try:
        if not AI_CONFIGURED or not API_KEY:
            raise Exception("Google AI not properly configured")
        
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

class ModernAuthManager:
    def __init__(self):
        self.users_file = "users_db.json"
        self.ensure_users_file()
    
    def ensure_users_file(self):
        """Ensure users database file exists"""
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def hash_password(self, password):
        """Hash password with salt"""
        salt = "modern_ai_platform_2025"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def register_user(self, username, password, email, full_name):
        """Register new user with validation"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                return False, "Username already exists!"
            
            # Email validation
            if '@' not in email or '.' not in email:
                return False, "Please enter a valid email address!"
            
            users[username] = {
                "password": self.hash_password(password),
                "email": email,
                "full_name": full_name,
                "created_at": datetime.now().isoformat(),
                "login_count": 0,
                "csv_queries": 0,
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
        """Login user with enhanced tracking"""
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
        """Get user information"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            return users.get(username, {})
        except:
            return {}
    
    def update_user_stats(self, username, stat_type):
        """Update user statistics"""
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users and stat_type in users[username]:
                users[username][stat_type] += 1
                
                with open(self.users_file, 'w') as f:
                    json.dump(users, f, indent=4)
        except Exception as e:
            print(f"Error updating user stats: {str(e)}")

class ModernMultiModalPlatform:
    def __init__(self):
        self.auth_manager = ModernAuthManager()
        self.csv_handler = PerfectCSVHandler()
        self.initialize_session_state()
        
        # Get AI models from cache
        self.gemini_model, self.gemini_vision, self.embeddings, self.models_ready = get_ai_models()
        
        # Apply CSS immediately
        self.apply_enhanced_css()
    
    def initialize_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            'authenticated': False,
            'username': '',
            'current_page': 'csv',
            # Perfect continuous chat histories for all three modules
            'csv_messages': [],
            'document_messages': [],
            'image_messages': [],
            'current_image': None,
            'document_vector_store': None,
            'processed_files': [],
            'csv_data': None,
            'csv_file_name': '',
            'csv_processing_info': '',
            'db_name': os.path.join(os.getcwd(), 'analytics_db'),  # Fix database path
            'table_name': 'data_table',
            'csv_columns': [],
            'dynamic_examples': []
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def apply_enhanced_css(self):
        """Apply enhanced CSS with modern black theme"""
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-gradient: linear-gradient(135deg, #333333 0%, #1a1a1a 100%);
            --secondary-gradient: linear-gradient(135deg, #444444 0%, #222222 100%);
            --success-gradient: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
            --error-gradient: linear-gradient(135deg, #5c1c1c 0%, #4a1515 100%);
            --bg-gradient: #000000;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-muted: #999999;
            --bg-black: #000000;
            --bg-darker: #0a0a0a;
            --bg-card: #111111;
            --bg-card-hover: #1a1a1a;
            --border-color: #333333;
            --accent-color: #4f46e5;
        }
        
        /* Global Black Theme */
        html, body, [data-testid="stAppViewContainer"] {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        
        .main {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #000000 !important;
            min-height: 100vh;
            color: #ffffff !important;
        }
        
        .stApp {
            background: #000000 !important;
        }
        
        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}
        header[data-testid="stHeader"] {visibility: hidden;}
        
        /* Modern Header */
        .modern-header {
            background: linear-gradient(135deg, #1a1a1a 0%, #000000 100%);
            padding: 4rem 2rem;
            border-radius: 0 0 3rem 3rem;
            text-align: center;
            margin: -1rem -1rem 3rem -1rem;
            box-shadow: 0 10px 30px rgba(255, 255, 255, 0.1);
            border: 2px solid #333333;
            position: relative;
            overflow: hidden;
        }
        
        .modern-header h1 {
            color: #ffffff;
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
            font-family: 'Poppins', sans-serif;
            position: relative;
            z-index: 1;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .modern-header p {
            color: #cccccc;
            font-size: 1.3rem;
            margin: 1rem 0 0 0;
            position: relative;
            z-index: 1;
        }
        
        /* Modern Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #333333 0%, #1a1a1a 100%) !important;
            color: #ffffff !important;
            border: 2px solid #444444 !important;
            border-radius: 1rem !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #444444 0%, #2a2a2a 100%) !important;
            border-color: #666666 !important;
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.2) !important;
        }
        
        /* Chat Messages */
        .stChatMessage {
            background-color: #111111 !important;
            border: 2px solid #333333 !important;
            border-radius: 1rem !important;
            color: #ffffff !important;
            margin: 0.5rem 0 !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }
        
        /* Form Styling */
        .stTextInput > div > div > input {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 2px solid #333333 !important;
            border-radius: 1rem !important;
            padding: 1rem !important;
        }
        
        /* File Uploader */
        .stFileUploader > div {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 2px dashed #444444 !important;
            border-radius: 1rem !important;
            padding: 2rem !important;
        }
        
        /* Data Tables */
        .stDataFrame {
            background: #1a1a1a !important;
            border: 2px solid #333333 !important;
            border-radius: 1rem !important;
        }
        
        .stDataFrame table {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
        }
        
        /* Success/Error Messages */
        .stSuccess {
            background: var(--success-gradient) !important;
            color: #ffffff !important;
            border-radius: 1rem !important;
        }
        
        .stError {
            background: var(--error-gradient) !important;
            color: #ffffff !important;
            border-radius: 1rem !important;
        }
        
        /* Text Elements */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-family: 'Poppins', sans-serif !important;
        }
        
        p, div, span, li {
            color: #ffffff !important;
        }
        
        /* CSV Processing Info Box */
        .csv-info-box {
            background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
            border: 2px solid #3b82f6;
            border-radius: 1rem;
            padding: 1rem;
            margin: 1rem 0;
            font-family: 'Consolas', monospace;
            font-size: 0.9rem;
            white-space: pre-line;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_modern_landing_page(self):
        """Render the modern landing page with authentication"""
        st.markdown("""
        <div class="modern-header">
            <h1>🚀 Elite MultiModal AI Platform</h1>
            <p>Advanced AI-Powered Analytics • Perfect CSV Intelligence • Document Processing • Vision AI</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            tab1, tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])
            
            with tab1:
                self.render_modern_login_form()
            
            with tab2:
                self.render_modern_register_form()

    def render_modern_login_form(self):
        """Render the modern login form"""
        st.markdown("### 🔐 Welcome Back")
        st.markdown("*Sign in to access your AI-powered analytics platform*")
        
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
                            st.success("🤖 AI Models: Ready!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.error("Please fill in all fields!")

    def render_modern_register_form(self):
        """Render the modern registration form"""
        st.markdown("### ✨ Join Elite Platform")
        st.markdown("*Create your account and start exploring AI-powered analytics*")
        
        with st.form("register_form"):
            full_name = st.text_input("👤 Full Name", placeholder="Enter your full name")
            username = st.text_input("🧑‍💼 Username", placeholder="Choose a username")
            email = st.text_input("📧 Email", placeholder="Enter your email address")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("🔐 Confirm Password", type="password", placeholder="Confirm your password")
            submitted = st.form_submit_button("✨ Create Account", use_container_width=True)
            
            if submitted:
                if all([full_name, username, email, password, confirm_password]):
                    if password != confirm_password:
                        st.error("Passwords don't match!")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters!")
                    elif len(username) < 3:
                        st.error("Username must be at least 3 characters!")
                    else:
                        success, message = self.auth_manager.register_user(username, password, email, full_name)
                        if success:
                            st.success(message)
                            st.info("Please sign in with your new account!")
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all fields!")

    def render_modern_dashboard(self):
        """Render the main dashboard"""
        user_info = self.auth_manager.get_user_info(st.session_state.username)
        
        st.markdown(f"""
        <div class="modern-header">
            <h1>👋 Welcome, {user_info.get('full_name', st.session_state.username)}</h1>
            <p>Elite MultiModal AI Dashboard • Perfect CSV, Document & Image Analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced Navigation
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        
        with col1:
            if st.button("📊 CSV Analytics", use_container_width=True):
                st.session_state.current_page = 'csv'
                st.rerun()
        
        with col2:
            if st.button("📄 Documents", use_container_width=True):
                st.session_state.current_page = 'documents'
                st.rerun()
        
        with col3:
            if st.button("🖼️ Images", use_container_width=True):
                st.session_state.current_page = 'images'
                st.rerun()
        
        with col4:
            col_spacer, col_logout = st.columns([3, 1])
            with col_logout:
                if st.button("🚪 Logout", use_container_width=True):
                    self.logout_user()
        
        st.markdown("---")
        
        # Render current page
        if st.session_state.current_page == 'csv':
            self.render_csv_page()
        elif st.session_state.current_page == 'documents':
            self.render_documents_page()
        elif st.session_state.current_page == 'images':
            self.render_images_page()

    def generate_perfect_dynamic_examples(self, df: pd.DataFrame) -> List[str]:
        """Generate comprehensive dynamic examples based on CSV structure and content"""
        if df is None or df.empty:
            return []
        
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        examples = [
            "Show me the first 10 rows of data",
            "What are all the columns in this dataset?",
            "How many total rows and columns are there?",
            "Show me a complete statistical summary",
            "Analyze missing values in detail",
            "Show unique values for each column",
            "Find all duplicate rows if any"
        ]
        
        # Add column-specific examples
        if numeric_columns:
            examples.extend([
                f"What is the average of {numeric_columns[0]}?",
                f"Show me the distribution of {numeric_columns[0]}",
                f"Find outliers in {numeric_columns[0]} column"
            ])
        
        if categorical_columns:
            examples.extend([
                f"Show unique values in {categorical_columns[0]}",
                f"Count records by {categorical_columns[0]}"
            ])
        
        return examples[:15]  # Limit to 15 examples

    def render_csv_page(self):
        """Render enhanced CSV Analytics page with perfect CSV handling"""
        st.markdown("## 📊 Perfect CSV Analytics with Universal Compatibility")
        st.markdown("*Handles ANY CSV format, encoding, delimiter, and structure automatically*")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "📤 Upload ANY CSV file - We'll handle the rest!", 
                type=["csv", "txt", "tsv", "dat"],
                help="Upload any CSV file with any encoding, delimiter, or format."
            )
        
        with col2:
            if st.button("🗑️ Clear Chat History", key="clear_csv", use_container_width=True):
                st.session_state.csv_messages = []
                st.success("✅ Chat history cleared!")
                time.sleep(1)
                st.rerun()
        
        # Perfect CSV Processing
        if uploaded_file:
            with st.spinner("🔧 Analyzing and processing your CSV file..."):
                try:
                    df, processing_info = self.csv_handler.smart_csv_reader(uploaded_file)
                    
                    if df is not None and not df.empty:
                        # Store data and info
                        st.session_state.csv_data = df
                        st.session_state.csv_file_name = uploaded_file.name
                        st.session_state.csv_processing_info = processing_info
                        st.session_state.csv_columns = df.columns.tolist()
                        
                        # Generate perfect dynamic examples
                        st.session_state.dynamic_examples = self.generate_perfect_dynamic_examples(df)
                        
                        # Create enhanced database
                        self.create_enhanced_csv_database(df)
                        
                        # Show success message
                        st.success(f"🎉 Successfully loaded: **{uploaded_file.name}**")
                        
                        # Show processing details
                        with st.expander("🔍 Processing Details", expanded=False):
                            st.markdown('<div class="csv-info-box">' + processing_info + '</div>', unsafe_allow_html=True)
                        
                        # Enhanced metrics dashboard
                        self.display_enhanced_csv_metrics(df)
                        
                    else:
                        st.error("❌ Failed to process the CSV file")
                        
                except Exception as e:
                    st.error(f"❌ Critical error processing CSV: {str(e)}")

        # Perfect Continuous Chat Interface for CSV
        if st.session_state.csv_data is not None:
            st.markdown("### 💬 Intelligent Chat with Your CSV Data")
            st.markdown("*Ask anything about your data in natural language - I understand it all!*")
            
            # Display chat history
            for message in st.session_state.csv_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    if message["role"] == "assistant" and "data_result" in message:
                        if message["data_result"] is not None and not message["data_result"].empty:
                            st.markdown("**📊 Query Results:**")
                            st.dataframe(message["data_result"], use_container_width=True)
                        
                        if "chart" in message and message["chart"]:
                            st.plotly_chart(message["chart"], use_container_width=True)
            
            # Chat input
            if prompt := st.chat_input("🔍 Ask anything about your CSV data!", key="csv_chat"):
                # Add user message
                st.session_state.csv_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 AI is analyzing your data..."):
                        response, data_result, chart = self.process_perfect_csv_chat(prompt)
                        
                        st.markdown(response)
                        
                        if data_result is not None and not data_result.empty:
                            st.markdown("**📊 Query Results:**")
                            st.dataframe(data_result, use_container_width=True)
                        
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        st.session_state.csv_messages.append({
                            "role": "assistant", 
                            "content": response,
                            "data_result": data_result,
                            "chart": chart
                        })
                        
                        self.auth_manager.update_user_stats(st.session_state.username, "csv_queries")
            
            # Dynamic example questions
            if st.session_state.dynamic_examples:
                with st.expander("💡 Smart Example Questions", expanded=False):
                    for example in st.session_state.dynamic_examples:
                        if st.button(f"▶️ {example}", key=f"ex_{hash(example)}", use_container_width=True):
                            st.session_state.csv_messages.append({"role": "user", "content": example})
                            response, data_result, chart = self.process_perfect_csv_chat(example)
                            st.session_state.csv_messages.append({
                                "role": "assistant", 
                                "content": response,
                                "data_result": data_result,
                                "chart": chart
                            })
                            st.rerun()

    def render_documents_page(self):
        """Render Document Intelligence page with PERFECT continuous chat"""
        st.markdown("## 📄 Document Intelligence with Perfect Continuous Chat")
        
        if not self.models_ready:
            st.error("❌ AI models not available. Please check your Google API key configuration.")
            return
        
        # Enhanced file upload section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uploaded_pdfs = st.file_uploader(
                "📄 Upload PDF Documents", 
                type="pdf", 
                accept_multiple_files=True, 
                key="pdf_docs",
                help="Upload PDF files for AI analysis"
            )
        
        with col2:
            uploaded_docs = st.file_uploader(
                "📝 Upload Word Documents", 
                type=["docx", "doc"], 
                accept_multiple_files=True, 
                key="word_docs",
                help="Upload Word documents for AI analysis"
            )
        
        with col3:
            if st.button("🗑️ Clear Chat History", key="clear_docs", use_container_width=True):
                st.session_state.document_messages = []
                st.success("✅ Document chat history cleared!")
                time.sleep(1)
                st.rerun()
        
        # Process documents
        if (uploaded_pdfs or uploaded_docs) and st.button("🔄 Process Documents", use_container_width=True):
            with st.spinner("🧠 Processing documents with AI..."):
                success = self.process_documents(uploaded_pdfs, uploaded_docs)
                if success:
                    st.success("✅ Documents processed successfully! You can now chat with them.")
                else:
                    st.error("❌ Error processing documents. Please try again.")
        
        # Show processed files
        if st.session_state.processed_files:
            st.markdown("### 📚 Processed Documents")
            for i, file_info in enumerate(st.session_state.processed_files, 1):
                st.write(f"{i}. {file_info}")
        
        # PERFECT CONTINUOUS CHAT INTERFACE FOR DOCUMENTS
        if st.session_state.document_vector_store is not None:
            st.markdown("### 💬 Intelligent Chat with Your Documents")
            st.markdown("*Ask anything about your documents - I understand them completely!*")
            
            # Display chat history - EXACTLY LIKE CSV
            for message in st.session_state.document_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input - EXACTLY LIKE CSV
            if prompt := st.chat_input("🔍 Ask anything about your documents...", key="doc_chat"):
                # Add user message to chat history
                st.session_state.document_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 AI is analyzing your documents..."):
                        response = self.process_document_chat(prompt)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Store complete message in chat history
                        st.session_state.document_messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        
                        # Update user stats
                        self.auth_manager.update_user_stats(st.session_state.username, "document_queries")
            
            # Example questions for documents
            with st.expander("💡 Example Document Questions", expanded=False):
                doc_examples = [
                    "What are the main topics discussed?",
                    "Summarize the key points from all documents",
                    "What conclusions are mentioned?",
                    "Who are the key people mentioned?",
                    "Find information about specific topics",
                    "What are the main findings?"
                ]
                
                for example in doc_examples:
                    if st.button(f"📄 {example}", key=f"doc_{hash(example)}", use_container_width=True):
                        st.session_state.document_messages.append({"role": "user", "content": example})
                        response = self.process_document_chat(example)
                        st.session_state.document_messages.append({"role": "assistant", "content": response})
                        st.rerun()

    def render_images_page(self):
        """Render Image Intelligence page with PERFECT continuous chat"""
        st.markdown("## 🖼️ Image Intelligence with Perfect Continuous Chat")
        
        if not self.models_ready:
            st.error("❌ AI models not available. Please check your Google API key configuration.")
            return
        
        # Enhanced image upload section
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image = st.file_uploader(
                "🖼️ Upload Image for Analysis", 
                type=["jpg", "jpeg", "png", "gif", "bmp", "webp"],
                help="Upload an image for AI-powered analysis"
            )
        
        with col2:
            if st.button("🗑️ Clear Chat History", key="clear_images", use_container_width=True):
                st.session_state.image_messages = []
                st.success("✅ Image chat history cleared!")
                time.sleep(1)
                st.rerun()
        
        # Process image
        if uploaded_image and st.button("🔄 Process Image", use_container_width=True):
            try:
                image = Image.open(uploaded_image)
                
                # Validate and resize large images
                if image.size[0] > 10000 or image.size[1] > 10000:
                    st.warning("⚠️ Large image detected. Resizing for optimal processing...")
                    image.thumbnail((4096, 4096), Image.Resampling.LANCZOS)
                
                st.session_state.current_image = image
                st.success(f"✅ Image processed successfully: {uploaded_image.name}")
                
                # Show image info
                st.info(f"📏 Image dimensions: {image.size[0]} x {image.size[1]} pixels")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")

        # Display current image and analysis tools
        if st.session_state.current_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🖼️ Current Image")
                st.image(st.session_state.current_image, use_container_width=True)
            
            with col2:
                st.markdown("### 🔧 Quick Actions")
                
                if st.button("📝 Extract Text (OCR)", use_container_width=True):
                    with st.spinner("🔍 Extracting text..."):
                        response = self.analyze_image_perfect("Extract all text visible in this image with high accuracy.")
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()
                
                if st.button("🔍 Comprehensive Analysis", use_container_width=True):
                    with st.spinner("🧠 Analyzing image..."):
                        response = self.analyze_image_perfect()
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()
            
            # PERFECT CONTINUOUS CHAT INTERFACE FOR IMAGES
            st.markdown("### 💬 Intelligent Chat with Your Image")
            st.markdown("*Ask anything about your image - I can see and understand everything!*")
            
            # Display chat history - EXACTLY LIKE CSV
            for message in st.session_state.image_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input - EXACTLY LIKE CSV
            if prompt := st.chat_input("🔍 Ask anything about your image...", key="image_chat"):
                # Add user message to chat history
                st.session_state.image_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 AI is analyzing your image..."):
                        response = self.analyze_image_perfect(prompt)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Store complete message in chat history
                        st.session_state.image_messages.append({
                            "role": "assistant", 
                            "content": response
                        })
                        
                        # Update user stats
                        self.auth_manager.update_user_stats(st.session_state.username, "image_queries")
            
            # Example questions for images
            with st.expander("💡 Example Image Questions", expanded=False):
                img_examples = [
                    "What objects can you see in this image?",
                    "Describe the setting and environment",
                    "What are the dominant colors?",
                    "Extract all visible text",
                    "Read any signs or labels",
                    "Describe interesting details"
                ]
                
                for example in img_examples:
                    if st.button(f"🖼️ {example}", key=f"img_{hash(example)}", use_container_width=True):
                        st.session_state.image_messages.append({"role": "user", "content": example})
                        response = self.analyze_image_perfect(example)
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()

    # ==================== CSV Processing Methods ====================
    
    def display_enhanced_csv_metrics(self, df: pd.DataFrame):
        """Display enhanced CSV metrics dashboard"""
        st.markdown("### 📊 Dataset Overview")
        
        # Primary metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📋 Total Rows", f"{len(df):,}")
        
        with col2:
            st.metric("📊 Columns", len(df.columns))
        
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("💾 Memory", f"{memory_mb:.1f} MB")
        
        with col4:
            missing_count = df.isnull().sum().sum()
            missing_pct = (missing_count / df.size) * 100 if df.size > 0 else 0
            st.metric("❓ Missing", f"{missing_count:,} ({missing_pct:.1f}%)")
        
        with col5:
            duplicate_count = df.duplicated().sum()
            st.metric("🔄 Duplicates", f"{duplicate_count:,}")
        
        # Data preview
        st.markdown("### 📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

    def process_perfect_csv_chat(self, question: str) -> Tuple[str, Optional[pd.DataFrame], Any]:
        """Perfect CSV chat processing with comprehensive analysis capabilities"""
        try:
            df = st.session_state.csv_data
            question_lower = question.lower()
            
            # First attempt: Try to understand the question and provide appropriate response
            if any(keyword in question_lower for keyword in ["show", "display", "head", "first", "preview", "sample"]):
                return self.handle_display_query(question, df)
            elif any(keyword in question_lower for keyword in ["columns", "fields", "structure", "info"]):
                return self.handle_column_query(question, df)
            elif any(keyword in question_lower for keyword in ["summary", "describe", "statistics", "stats", "overview"]):
                return self.handle_summary_query(question, df)
            elif any(keyword in question_lower for keyword in ["missing", "null", "empty", "na"]):
                return self.handle_missing_query(question, df)
            elif any(keyword in question_lower for keyword in ["unique", "distinct", "different"]):
                return self.handle_unique_query(question, df)
            elif any(keyword in question_lower for keyword in ["duplicate", "duplicated", "repeated"]):
                return self.handle_duplicate_query(question, df)
            else:
                # Default response for other questions
                response = f"🤔 **I understand you're asking:** '{question}'\n\nLet me show you a sample of your data to help:"
                return response, df.head(10), None
                    
        except Exception as e:
            error_response = f"❌ **Error processing question:** {str(e)}\n\nHere's a sample of your data:"
            return error_response, df.head(5) if df is not None else None, None

    # ALL MISSING HANDLER METHODS - COMPLETE IMPLEMENTATION
    
    def handle_display_query(self, question: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Any]:
        """Handle data display queries"""
        numbers = re.findall(r'\d+', question)
        num_rows = min(int(numbers[0]), 100) if numbers else 10
        
        result_df = df.head(num_rows)
        response = f"📋 **Displaying first {len(result_df)} rows:**\n• Dataset shape: **{df.shape[0]:,} rows × {df.shape[1]} columns**"
        return response, result_df, None

    def handle_column_query(self, question: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Any]:
        """Handle column information queries"""
        columns_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Sample Value': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'N/A' for col in df.columns]
        })
        
        response = f"📊 **Column Information:**\n• Total columns: **{len(df.columns)}**"
        return response, columns_info, None

    def handle_summary_query(self, question: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Any]:
        """Handle summary statistics queries"""
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            summary_df = numeric_df.describe()
            response = f"📈 **Statistical Summary:**\n• Numeric columns: **{len(numeric_df.columns)}**"
            return response, summary_df, None
        else:
            response = f"📊 **Dataset Overview:**\n• No numeric columns for statistical summary"
            basic_info = pd.DataFrame({
                'Metric': ['Total Rows', 'Total Columns', 'Data Types'],
                'Value': [len(df), len(df.columns), ', '.join(df.dtypes.astype(str).unique())]
            })
            return response, basic_info, None

    def handle_missing_query(self, question: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Any]:
        """Handle missing values analysis"""
        missing_data = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': round((missing_data.values / len(df)) * 100, 2)
        }).sort_values('Missing Count', ascending=False)
        
        total_missing = missing_data.sum()
        response = f"🔍 **Missing Values Analysis:**\n• Total missing: **{total_missing:,}** values"
        return response, missing_df, None

    def handle_unique_query(self, question: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Any]:
        """Handle unique values analysis"""
        unique_info = pd.DataFrame({
            'Column': df.columns,
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Total Values': len(df),
            'Most Common': [str(df[col].mode().iloc[0]) if not df[col].mode().empty else 'N/A' for col in df.columns]
        })
        
        response = f"🔢 **Unique Values Analysis:**\n• Columns analyzed: **{len(df.columns)}**"
        return response, unique_info, None

    def handle_duplicate_query(self, question: str, df: pd.DataFrame) -> Tuple[str, pd.DataFrame, Any]:
        """Handle duplicate analysis"""
        duplicates = df.duplicated()
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            duplicate_rows = df[duplicates].head(20)
            response = f"🔍 **Duplicate Analysis:**\n• Found **{duplicate_count}** duplicate rows"
            return response, duplicate_rows, None
        else:
            info_df = pd.DataFrame({
                'Analysis': ['Total Rows', 'Duplicate Rows', 'Unique Rows'],
                'Result': [len(df), duplicate_count, len(df) - duplicate_count]
            })
            response = f"✅ **No duplicate rows found!**"
            return response, info_df, None

    def create_enhanced_csv_database(self, df):
        """Create enhanced SQLite database with proper path handling"""
        try:
            # Create a simple database name without path issues
            db_path = "analytics_db.db"
            
            connection = sqlite3.connect(db_path)
            df.to_sql(st.session_state.table_name, connection, if_exists='replace', index=False)
            connection.close()
            return True
        except Exception as e:
            print(f"Database creation error: {str(e)}")
            return False

    # ==================== Document Processing Methods ====================
    
    def process_documents(self, uploaded_pdfs, uploaded_docs):
        """Process uploaded documents with enhanced progress tracking"""
        try:
            all_text = ""
            processed_files = []
            
            # Process PDFs
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
                            processed_files.append(f"📄 {pdf_file.name} ({len(pdf_reader.pages)} pages)")
                        
                    except Exception as e:
                        st.error(f"❌ Error reading {pdf_file.name}: {str(e)}")
            
            # Process Word documents
            if uploaded_docs:
                for doc_file in uploaded_docs:
                    try:
                        doc = docx.Document(doc_file)
                        doc_text = ""
                        para_count = 0
                        
                        for paragraph in doc.paragraphs:
                            if paragraph.text.strip():
                                doc_text += paragraph.text + "\n"
                                para_count += 1
                        
                        if doc_text.strip():
                            all_text += f"\n--- {doc_file.name} ---\n{doc_text}\n"
                            processed_files.append(f"📝 {doc_file.name} ({para_count} paragraphs)")
                        
                    except Exception as e:
                        st.error(f"❌ Error reading {doc_file.name}: {str(e)}")
            
            # Create vector store if we have text
            if all_text.strip():
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=10000,
                    chunk_overlap=1000,
                    length_function=len
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
        """Process document chat with enhanced analysis"""
        if not self.models_ready:
            return "❌ AI models not available for document analysis."
        
        try:
            # Get relevant documents
            docs = st.session_state.document_vector_store.similarity_search(question, k=4)
            
            if not docs:
                return "❌ No relevant information found in the documents for your question."
            
            # Enhanced prompt template
            prompt_template = """
            You are an expert document analyst. Analyze the provided document context and answer the question comprehensively.
            
            Document Context:
            {context}
            
            Question: {question}
            
            Professional Analysis:
            """
            
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create and run chain
            chain = load_qa_chain(self.gemini_model, chain_type="stuff", prompt=prompt)
            
            response = chain.invoke(
                {"input_documents": docs, "question": question},
                return_only_outputs=True
            )
            
            return f"📚 **Document Analysis:**\n\n{response['output_text']}"
            
        except Exception as e:
            return f"❌ Error analyzing documents: {str(e)}. Please try rephrasing your question."

    # ==================== Image Processing Methods ====================
    
    def analyze_image_perfect(self, question=""):
        """Enhanced image analysis with comprehensive prompts"""
        if st.session_state.current_image is None:
            return "❌ No image uploaded. Please upload an image first."
        
        if not self.models_ready:
            return "❌ AI vision models not available for image analysis."
        
        try:
            image = st.session_state.current_image
            
            if question:
                prompt = f"""
                You are an expert computer vision analyst. Analyze this image and answer: {question}
                
                Provide a comprehensive response with:
                1. Direct answer to the question
                2. Visual evidence supporting your answer
                3. Any visible text or labels
                4. Additional relevant details
                
                Be thorough and accurate in your analysis.
                """
            else:
                prompt = """
                You are an expert computer vision analyst. Provide a comprehensive analysis of this image.
                
                Include:
                - Main subjects and objects
                - Setting and environment
                - Colors and visual elements
                - Any visible text or writing
                - Notable details and observations
                - Technical aspects (lighting, composition, quality)
                
                Provide a detailed, professional analysis.
                """
            
            response = self.gemini_vision.generate_content([prompt, image])
            return f"🖼️ **Expert Image Analysis:**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ Error analyzing image: {str(e)}. Please try uploading the image again."

    def logout_user(self):
        """Enhanced logout with cleanup"""
        username = st.session_state.get('username', 'User')
        user_info = self.auth_manager.get_user_info(username)
        
        st.success(f"👋 Thank you for using Elite AI Platform, {user_info.get('full_name', username)}!")
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        time.sleep(2)
        st.rerun()

    def run(self):
        """Main application runner"""
        try:
            if not st.session_state.authenticated:
                self.render_modern_landing_page()
            else:
                self.render_modern_dashboard()
        except Exception as e:
            st.error(f"❌ Application Error: {str(e)}")
            st.info("Please refresh the page and try again.")

# Application Entry Point
if __name__ == "__main__":
    try:
        app = ModernMultiModalPlatform()
        app.run()
    except Exception as e:
        st.error(f"❌ Critical Application Error: {str(e)}")
        st.info("Please check your environment setup and API keys.")
