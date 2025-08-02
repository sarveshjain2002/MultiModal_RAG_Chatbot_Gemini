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
#from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chat_models import ChatGoogleGenerativeAI
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

# Configure Google AI
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except Exception as e:
    st.error(f"Google AI configuration error: {str(e)}")

# Global AI Models - Initialize once and reuse
@st.cache_resource
def get_ai_models():
    """Initialize and cache AI models"""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise Exception("Google API Key not found")
        
        gemini_model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=api_key
        )
        
        gemini_vision = genai.GenerativeModel('gemini-2.0-flash')
        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
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
        if not os.path.exists(self.users_file):
            with open(self.users_file, 'w') as f:
                json.dump({}, f)
    
    def hash_password(self, password):
        salt = "modern_ai_platform_2025"
        return hashlib.sha256((password + salt).encode()).hexdigest()
    
    def register_user(self, username, password, email, full_name):
        try:
            with open(self.users_file, 'r') as f:
                users = json.load(f)
            
            if username in users:
                return False, "Username already exists!"
            
            users[username] = {
                "password": self.hash_password(password),
                "email": email,
                "full_name": full_name,
                "created_at": datetime.now().isoformat(),
                "login_count": 0,
                "csv_queries": 0,
                "document_queries": 0,
                "image_queries": 0
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
        except:
            pass

class ModernMultiModalPlatform:
    def __init__(self):
        self.auth_manager = ModernAuthManager()
        self.initialize_session_state()
        
        # Get AI models from cache
        self.gemini_model, self.gemini_vision, self.embeddings, self.models_ready = get_ai_models()
        
        # Apply CSS immediately
        self.apply_full_black_css()
    
    def initialize_session_state(self):
        defaults = {
            'authenticated': False,
            'username': '',
            'current_page': 'csv',
            # Separate chat histories for each module - PERFECT CONTINUOUS CHAT
            'csv_messages': [],
            'document_messages': [],
            'image_messages': [],
            'current_image': None,
            'document_vector_store': None,
            'processed_files': [],
            'csv_data': None,
            'csv_file_name': '',
            'db_name': 'analytics_db',
            'table_name': 'data_table',
            'csv_columns': [],
            'dynamic_examples': []
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def apply_full_black_css(self):
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-gradient: linear-gradient(135deg, #333333 0%, #1a1a1a 100%);
            --secondary-gradient: linear-gradient(135deg, #444444 0%, #222222 100%);
            --success-gradient: linear-gradient(135deg, #333333 0%, #1a1a1a 100%);
            --bg-gradient: #000000;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-muted: #999999;
            --bg-black: #000000;
            --bg-darker: #0a0a0a;
            --bg-card: #111111;
            --bg-card-hover: #1a1a1a;
            --border-color: #333333;
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
        
        [data-testid="stAppViewContainer"] > .main {
            background: #000000 !important;
        }
        
        [data-testid="stHeader"] {
            background: #000000 !important;
        }
        
        /* Hide Streamlit Elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display:none;}
        
        /* Modern Header */
        .modern-header {
            background: #000000;
            padding: 4rem 2rem;
            border-radius: 0 0 3rem 3rem;
            text-align: center;
            margin: -1rem -1rem 3rem -1rem;
            box-shadow: 0 4px 20px rgba(255, 255, 255, 0.1);
            border: 2px solid #333333;
        }
        
        .modern-header h1 {
            color: #ffffff;
            font-size: 3.5rem;
            font-weight: 800;
            margin: 0;
            font-family: 'Poppins', sans-serif;
        }
        
        .modern-header p {
            color: #cccccc;
            font-size: 1.3rem;
            margin: 1rem 0 0 0;
        }
        
        /* Modern Buttons */
        .stButton > button {
            background: #222222 !important;
            color: #ffffff !important;
            border: 2px solid #444444 !important;
            border-radius: 1rem !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background: #333333 !important;
            border-color: #666666 !important;
        }
        
        /* Chat Messages */
        .stChatMessage {
            background-color: #111111 !important;
            border: 2px solid #333333 !important;
            border-radius: 1rem !important;
            color: #ffffff !important;
        }
        
        /* Form Styling */
        .stTextInput > div > div > input {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 2px solid #333333 !important;
            border-radius: 1rem !important;
        }
        
        /* File Uploader */
        .stFileUploader > div {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 2px dashed #333333 !important;
            border-radius: 1rem !important;
        }
        
        /* Data Tables */
        .stDataFrame {
            background: #000000 !important;
            border: 2px solid #333333 !important;
            border-radius: 1rem !important;
        }
        
        .stDataFrame table {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        
        .stDataFrame th {
            background-color: #111111 !important;
            color: #ffffff !important;
            border-color: #333333 !important;
        }
        
        .stDataFrame td {
            background-color: #000000 !important;
            color: #ffffff !important;
            border-color: #333333 !important;
        }
        
        /* Success/Error Messages */
        .stSuccess {
            background: #222222 !important;
            color: #ffffff !important;
            border: 2px solid #444444 !important;
        }
        
        .stError {
            background: #330000 !important;
            color: #ffffff !important;
            border: 2px solid #660000 !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab"] {
            background: #222222 !important;
            color: #ffffff !important;
            border: 1px solid #444444 !important;
        }
        
        h1, h2, h3, h4, h5, h6, p {
            color: #ffffff !important;
        }
        
        /* Plotly Charts */
        .js-plotly-plot {
            background: #000000 !important;
        }
        </style>
        """, unsafe_allow_html=True)

    def render_modern_landing_page(self):
        st.markdown("""
        <div class="modern-header">
            <h1>🚀 Elite MultiModal AI Platform</h1>
            <p>Advanced AI-Powered Analytics • CSV Intelligence • Document Processing • Vision AI</p>
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
        st.markdown("### 🔐 Welcome Back")
        
        with st.form("login_form"):
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("🚀 Sign In")
            
            if submitted and username and password:
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

    def render_modern_register_form(self):
        st.markdown("### ✨ Join Elite Platform")
        
        with st.form("register_form"):
            full_name = st.text_input("👤 Full Name", placeholder="Enter your full name")
            username = st.text_input("🧑‍💼 Username", placeholder="Choose a username")
            email = st.text_input("📧 Email", placeholder="Enter your email address")
            password = st.text_input("🔒 Password", type="password", placeholder="Create a password")
            confirm_password = st.text_input("🔐 Confirm Password", type="password", placeholder="Confirm your password")
            submitted = st.form_submit_button("✨ Create Account")
            
            if submitted:
                if all([full_name, username, email, password, confirm_password]):
                    if password != confirm_password:
                        st.error("Passwords don't match!")
                    elif len(password) < 6:
                        st.error("Password must be at least 6 characters!")
                    else:
                        success, message = self.auth_manager.register_user(username, password, email, full_name)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all fields!")

    def render_modern_dashboard(self):
        user_info = self.auth_manager.get_user_info(st.session_state.username)
        
        st.markdown(f"""
        <div class="modern-header">
            <h1>👋 Welcome, {user_info.get('full_name', st.session_state.username)}</h1>
            <p>Elite MultiModal AI Dashboard • Advanced Analytics & Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
        
        with col1:
            if st.button("📊 CSV Analytics"):
                st.session_state.current_page = 'csv'
                st.rerun()
        
        with col2:
            if st.button("📄 Documents"):
                st.session_state.current_page = 'documents'
                st.rerun()
        
        with col3:
            if st.button("🖼️ Images"):
                st.session_state.current_page = 'images'
                st.rerun()
        
        with col4:
            col_spacer, col_logout = st.columns([3, 1])
            with col_logout:
                if st.button("🚪 Logout"):
                    self.logout_user()
        
        st.markdown("---")
        
        # Render pages
        if st.session_state.current_page == 'csv':
            self.render_csv_page()
        elif st.session_state.current_page == 'documents':
            self.render_documents_page()
        elif st.session_state.current_page == 'images':
            self.render_images_page()

    def generate_dynamic_examples(self, df):
        """Generate smart example questions based on CSV columns and data types"""
        if df is None or df.empty:
            return []
        
        columns = df.columns.tolist()
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        examples = [
            "Show me the first 10 rows",
            "What columns are in this dataset?",
            "How many rows are there?",
            "Show me the data summary",
            "Show missing values analysis",
            "Show unique values for each column"
        ]
        
        # Add column-specific examples
        if numeric_columns:
            examples.extend([
                f"What is the average {numeric_columns[0]}?",
                f"Show me the maximum {numeric_columns[0]}",
                f"Calculate the sum of {numeric_columns[0]}"
            ])
            
            if len(numeric_columns) > 1:
                examples.append(f"Compare {numeric_columns[0]} and {numeric_columns[1]}")
        
        if categorical_columns:
            examples.extend([
                f"Show unique values in {categorical_columns[0]}",
                f"Count records by {categorical_columns[0]}"
            ])
            
            if numeric_columns and categorical_columns:
                examples.extend([
                    f"Show average {numeric_columns[0]} by {categorical_columns[0]}",
                    f"Create a chart of {numeric_columns[0]} by {categorical_columns[0]}"
                ])
        
        # Add filtering examples
        if len(columns) > 0:
            examples.extend([
                f"Filter rows where {columns[0]} contains 'value'",
                f"Show top 5 records sorted by {columns[0]}",
                "Find duplicate rows in the dataset"
            ])
        
        return examples

    def render_csv_page(self):
        st.markdown("## 📊 CSV Analytics with Perfect Continuous Chat")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader("📤 Choose CSV file for analysis", type="csv", help="Upload your dataset for AI-powered analytics")
        
        with col2:
            if st.button("🗑️ Clear Chat History", key="clear_csv"):
                st.session_state.csv_messages = []
                st.success("✅ Chat history cleared!")
                time.sleep(1)
                st.rerun()
        
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.csv_data = df
                st.session_state.csv_file_name = uploaded_file.name
                st.session_state.csv_columns = df.columns.tolist()
                
                # Generate dynamic examples based on the uploaded CSV
                st.session_state.dynamic_examples = self.generate_dynamic_examples(df)
                
                # Create database
                self.create_csv_database(df)
                
                st.success(f"✅ Dataset loaded successfully: {uploaded_file.name}")
                
                # Display enhanced metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Total Rows", f"{len(df):,}")
                with col2:
                    st.metric("📋 Columns", len(df.columns))
                with col3:
                    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                    st.metric("💾 Memory Usage", f"{memory_mb:.1f} MB")
                with col4:
                    missing = df.isnull().sum().sum()
                    st.metric("❓ Missing Values", f"{missing:,}")
                
                # Enhanced data preview
                st.markdown("### 📋 Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading CSV: {str(e)}")

        # Perfect Continuous Chat Interface - WORKING PERFECTLY
        if st.session_state.csv_data is not None:
            st.markdown("### 💬 Intelligent Chat with Your CSV Data")
            
            # Display chat history with enhanced formatting
            for message in st.session_state.csv_messages:
                with st.chat_message(message["role"]):
                    # Display text content
                    st.markdown(message["content"])
                    
                    # Display dataframe results if present
                    if message["role"] == "assistant" and "data_result" in message:
                        if message["data_result"] is not None and not message["data_result"].empty:
                            st.markdown("**📊 Query Results:**")
                            st.dataframe(message["data_result"], use_container_width=True)
                        
                        # Display charts if present
                        if "chart" in message and message["chart"]:
                            st.plotly_chart(message["chart"], use_container_width=True)
            
            # Chat input with perfect processing
            if prompt := st.chat_input("🔍 Ask anything about your CSV data...", key="csv_chat"):
                # Add user message
                st.session_state.csv_messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate assistant response
                with st.chat_message("assistant"):
                    with st.spinner("🧠 AI is analyzing your data..."):
                        response, data_result, chart = self.process_csv_chat(prompt)
                        
                        # Display response
                        st.markdown(response)
                        
                        # Display dataframe if available
                        if data_result is not None and not data_result.empty:
                            st.markdown("**📊 Query Results:**")
                            st.dataframe(data_result, use_container_width=True)
                        
                        # Display chart if available
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        
                        # Store complete message
                        st.session_state.csv_messages.append({
                            "role": "assistant", 
                            "content": response,
                            "data_result": data_result,
                            "chart": chart
                        })
                        
                        # Update user stats
                        self.auth_manager.update_user_stats(st.session_state.username, "csv_queries")
            
            # Dynamic example questions based on uploaded data
            if st.session_state.dynamic_examples:
                with st.expander("💡 Smart Example Questions (Based on Your Data)"):
                    st.markdown("**🔍 Data Exploration:**")
                    for example in st.session_state.dynamic_examples[:6]:
                        st.markdown(f"- \"{example}\"")
                    
                    if len(st.session_state.dynamic_examples) > 6:
                        st.markdown("**📈 Advanced Analysis:**")
                        for example in st.session_state.dynamic_examples[6:]:
                            st.markdown(f"- \"{example}\"")

    def render_documents_page(self):
        st.markdown("## 📄 Document Intelligence with Perfect Continuous Chat")
        
        if not self.models_ready:
            st.error("❌ AI models not available. Please check your Google API key configuration.")
            return
        
        # File upload section
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uploaded_pdfs = st.file_uploader("📄 Upload PDF Documents", type="pdf", accept_multiple_files=True, key="pdf_docs")
        
        with col2:
            uploaded_docs = st.file_uploader("📝 Upload Word Documents", type=["docx", "doc"], accept_multiple_files=True, key="word_docs")
        
        with col3:
            if st.button("🗑️ Clear Chat History", key="clear_docs"):
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
            for file_info in st.session_state.processed_files:
                st.write(f"- {file_info}")
        
        # PERFECT CONTINUOUS CHAT INTERFACE FOR DOCUMENTS - LIKE CSV
        if st.session_state.document_vector_store is not None:
            st.markdown("### 💬 Intelligent Chat with Your Documents")
            
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
            
            # Example questions
            with st.expander("💡 Example Document Questions"):
                st.markdown("""
                **📋 Content Analysis:**
                - "What are the main topics discussed in the documents?"
                - "Summarize the key points from all documents"
                - "What conclusions or recommendations are mentioned?"
                - "List all important dates mentioned"
                
                **🔍 Specific Information:**
                - "What is mentioned about [specific topic]?"
                - "Find information about [keyword or phrase]"
                - "Who are the key people mentioned?"
                - "What are the main findings?"
                
                **📊 Comparative Analysis:**
                - "Compare the main arguments across documents"
                - "What are the differences between documents?"
                - "What common themes appear in all documents?"
                """)

    def render_images_page(self):
        st.markdown("## 🖼️ Image Intelligence with Perfect Continuous Chat")
        
        if not self.models_ready:
            st.error("❌ AI models not available. Please check your Google API key configuration.")
            return
        
        # Image upload section
        col1, col2 = st.columns(2)
        
        with col1:
            uploaded_image = st.file_uploader("🖼️ Upload Image for Analysis", type=["jpg", "jpeg", "png", "gif", "bmp", "webp"])
        
        with col2:
            if st.button("🗑️ Clear Chat History", key="clear_images"):
                st.session_state.image_messages = []
                st.success("✅ Image chat history cleared!")
                time.sleep(1)
                st.rerun()
        
        # Process image
        if uploaded_image and st.button("🔄 Process Image", use_container_width=True):
            try:
                image = Image.open(uploaded_image)
                st.session_state.current_image = image
                st.success(f"✅ Image processed: {uploaded_image.name}")
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
        
        # Display current image and analysis
        if st.session_state.current_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🖼️ Current Image")
                st.image(st.session_state.current_image, use_container_width=True)
            
            with col2:
                st.markdown("### 🔧 Quick Actions")
                
                if st.button("📝 Extract Text (OCR)", use_container_width=True):
                    with st.spinner("🔍 Extracting text..."):
                        response = self.analyze_image_perfect("Extract all text visible in this image with high accuracy. Provide the text in a structured format.")
                        # Add to chat history automatically
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()
                
                if st.button("🔍 Comprehensive Analysis", use_container_width=True):
                    with st.spinner("🧠 Analyzing image..."):
                        response = self.analyze_image_perfect()
                        # Add to chat history automatically
                        st.session_state.image_messages.append({"role": "assistant", "content": response})
                        st.rerun()
            
            # PERFECT CONTINUOUS CHAT INTERFACE FOR IMAGES - LIKE CSV
            st.markdown("### 💬 Intelligent Chat with Your Image")
            
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
            with st.expander("💡 Example Image Questions"):
                st.markdown("""
                **🖼️ Visual Analysis:**
                - "What objects and people can you see in this image?"
                - "Describe the setting and environment"
                - "What are the dominant colors and visual style?"
                - "What emotions or mood does this image convey?"
                
                **📝 Text & Information:**
                - "Extract all text visible in this image"
                - "Read any signs, labels, or documents shown"
                - "What brand names or logos can you identify?"
                - "Are there any numbers, codes, or dates visible?"
                
                **🔍 Detailed Analysis:**
                - "What safety issues or hazards can you identify?"
                - "Describe the lighting and time of day"
                - "What can you tell about the photo quality and composition?"
                - "Are there any interesting or unusual details?"
                """)

    # CSV Processing Methods - WORKING PERFECTLY
    def process_csv_chat(self, question):
        """Enhanced CSV chat processing with your working logic"""
        try:
            df = st.session_state.csv_data
            
            # First, try to handle with AI if available
            if self.models_ready:
                # Try to generate and execute SQL query
                result = self.execute_csv_query_safe(question)
                if result.get('success', False):
                    result_df = result['result']
                    response = f"✅ **Query executed successfully!**\n📊 Found **{len(result_df):,}** records"
                    
                    # Create visualization if appropriate
                    chart = self.create_chart_if_applicable(result_df)
                    return response, result_df, chart
            
            # Fallback to keyword-based processing
            question_lower = question.lower()
            
            if any(keyword in question_lower for keyword in ["show data", "first", "head", "rows"]):
                # Extract number if mentioned
                num_rows = 10
                numbers = [int(word) for word in question.split() if word.isdigit()]
                if numbers:
                    num_rows = min(numbers[0], 100)  # Cap at 100 rows
                
                result_df = df.head(num_rows)
                response = f"📋 **Showing first {len(result_df)} rows of your dataset:**"
                return response, result_df, None
                
            elif any(keyword in question_lower for keyword in ["columns", "column names", "fields"]):
                columns_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Sample Value': [str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else 'N/A' for col in df.columns]
                })
                response = f"📊 **Dataset Column Information:**\n- Total Columns: **{len(df.columns)}**"
                return response, columns_info, None
                
            elif any(keyword in question_lower for keyword in ["summary", "describe", "statistics", "stats"]):
                try:
                    summary_df = df.describe(include='all').fillna('')
                    response = f"📈 **Complete Data Summary:**"
                    return response, summary_df, None
                except:
                    info_df = pd.DataFrame({
                        'Column': df.columns,
                        'Type': df.dtypes.astype(str),
                        'Count': df.count(),
                        'Unique': [df[col].nunique() for col in df.columns]
                    })
                    response = f"📊 **Dataset Overview:**"
                    return response, info_df, None
                    
            elif any(keyword in question_lower for keyword in ["missing", "null", "empty"]):
                missing_data = df.isnull().sum()
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing Percentage': round((missing_data.values / len(df)) * 100, 2),
                    'Data Type': df.dtypes.astype(str)
                }).sort_values('Missing Count', ascending=False)
                
                response = f"🔍 **Missing Values Analysis:**\n- Total missing values: **{missing_data.sum()}**"
                return response, missing_df, None
                
            elif any(keyword in question_lower for keyword in ["unique", "distinct"]):
                unique_info = pd.DataFrame({
                    'Column': df.columns,
                    'Unique Values': [df[col].nunique() for col in df.columns],
                    'Total Values': len(df),
                    'Uniqueness %': [round((df[col].nunique() / len(df)) * 100, 2) for col in df.columns],
                    'Data Type': df.dtypes.astype(str)
                })
                response = f"🔢 **Unique Values Analysis:**"
                return response, unique_info, None
                
            else:
                # For any other question, try natural language processing
                if self.models_ready:
                    try:
                        # Use AI to understand the question and generate response
                        ai_response = self.generate_ai_response(question, df)
                        return ai_response, df.head(10), None
                    except:
                        pass
                
                # Final fallback
                response = f"🤔 **I understand you're asking:** '{question}'\n\nLet me show you a sample of your data to help:"
                return response, df.head(10), None
                    
        except Exception as e:
            return f"❌ Error processing question: {str(e)}\n\nHere's a sample of your data:", df.head(5) if df is not None else None, None

    def generate_ai_response(self, question, df):
        """Generate AI response for complex questions"""
        if not self.models_ready:
            return "❌ AI models not available for complex analysis."
        
        # Create a summary of the dataset for AI context
        data_summary = f"""
        Dataset Info:
        - Rows: {len(df)}
        - Columns: {', '.join(df.columns.tolist())}
        - Data types: {dict(df.dtypes.astype(str))}
        - Sample data: 
        {df.head(3).to_string()}
        """
        
        prompt = f"""
        You are a data analyst. Answer this question about the dataset: {question}
        
        {data_summary}
        
        Provide a helpful response about the data. If you need to show specific data, suggest what to look for.
        Keep the response concise and informative.
        """
        
        try:
            response = self.gemini_model.invoke(prompt)
            return f"🤖 **AI Analysis:**\n\n{response.content}"
        except:
            return f"🤔 I understand you're asking about: '{question}'. Here's what I can show you:"

    def create_chart_if_applicable(self, result_df):
        """Create chart if the result data is suitable for visualization"""
        if result_df is None or result_df.empty or len(result_df.columns) != 2:
            return None
        
        try:
            col1, col2 = result_df.columns
            
            # Only create chart for reasonable number of rows
            if len(result_df) > 50:
                return None
            
            # Check if second column is numeric
            if result_df[col2].dtype.kind in 'biufc':  # numeric types
                chart = px.bar(
                    result_df, 
                    x=col1, 
                    y=col2,
                    title=f"{col2} by {col1}",
                    template="plotly_dark"
                )
                chart.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    title=dict(font=dict(color='white'))
                )
                return chart
        except Exception:
            pass
        
        return None

    def execute_csv_query_safe(self, question):
        """Execute CSV query safely with improved error handling"""
        if st.session_state.csv_data is None:
            return {"success": False, "error": "No CSV data loaded"}
        
        try:
            if not self.models_ready:
                return {"success": False, "error": "AI models not available"}
            
            df = st.session_state.csv_data
            sql_query = self.generate_sql_query_improved(question, df)
            cleaned_query = self.clean_sql_query(sql_query)
            
            # Execute query
            connection = sqlite3.connect("analytics_db.db")
            result_df = pd.read_sql_query(cleaned_query, connection)
            connection.close()
            
            self.auth_manager.update_user_stats(st.session_state.username, "csv_queries")
            
            return {
                'sql_query': cleaned_query,
                'result': result_df,
                'success': True
            }
            
        except Exception as e:
            return {'error': str(e), 'success': False}

    def generate_sql_query_improved(self, question, df):
        """Generate improved SQL query with better prompting"""
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_data = df.head(3).to_string()
        
        prompt = f"""
        You are a SQL expert. Generate a clean, executable SQL query to answer this question: "{question}"
        
        Database Information:
        - Table name: data_table
        - Columns and types: {dtypes}
        - Sample data:
        {sample_data}
        
        Rules:
        1. Return ONLY the SQL query, no explanations or formatting
        2. Use proper SQLite syntax
        3. Use double quotes for column names with spaces
        4. Limit results to reasonable numbers (use LIMIT clause)
        5. Handle case-insensitive searches with LOWER() function
        
        Generate the SQL query:
        """
        
        try:
            response = self.gemini_model.invoke(prompt)
            query = response.content.strip()
            
            # Ensure query doesn't contain dangerous operations
            dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
            query_upper = query.upper()
            
            for keyword in dangerous_keywords:
                if keyword in query_upper:
                    return f"SELECT * FROM data_table LIMIT 10"  # Safe fallback
            
            return query
            
        except Exception:
            return f"SELECT * FROM data_table LIMIT 10"

    def clean_sql_query(self, raw_query):
        """Clean SQL query with robust handling"""
        if not raw_query:
            return "SELECT * FROM data_table LIMIT 10"
        
        cleaned = raw_query.strip()
        
        # Remove markdown code blocks
        if "```" in cleaned:
            # Split by triple backticks
            parts = cleaned.split("```")
            if len(parts) >= 3:
                cleaned = parts[1]  # extract the part between triple backticks
            else:
                cleaned = cleaned.replace("```", "")

        # Remove 'sql' language identifier if it's the first word
        if cleaned.lower().startswith("sql"):
            cleaned = cleaned[3:].strip()
        
        # Clean up line by line
        lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
        sql_lines = []
        
        for line in lines:
            if line and not line.startswith('#') and not line.startswith('--'):
                sql_lines.append(line)
        
        final_query = ' '.join(sql_lines).rstrip(';')
        
        # Basic validation
        if not final_query or len(final_query.strip()) < 5:
            return "SELECT * FROM data_table LIMIT 10"
        
        return final_query

    def create_csv_database(self, df):
        """Create SQLite database from CSV with better error handling"""
        try:
            # Clean column names for SQL compatibility
            df_clean = df.copy()
            df_clean.columns = [f'"{col}"' if ' ' in col or '-' in col else col for col in df.columns]
            
            connection = sqlite3.connect("analytics_db.db")
            df_clean.to_sql("data_table", connection, if_exists='replace', index=False)
            connection.close()
            return True
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            return False

    # Document Processing Methods - NOW WITH PERFECT CONTINUOUS CHAT
    def process_documents(self, uploaded_pdfs, uploaded_docs):
        """Process uploaded documents and create vector store"""
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
                        
                        # Process tables
                        for table in doc.tables:
                            for row in table.rows:
                                row_text = " | ".join([cell.text.strip() for cell in row.cells])
                                if row_text.strip():
                                    doc_text += row_text + "\n"
                        
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
        """Process document chat with enhanced responses - PERFECT CONTINUOUS CHAT"""
        if not self.models_ready:
            return "❌ AI models not available for document analysis."
        
        try:
            # Get relevant documents
            docs = st.session_state.document_vector_store.similarity_search(question, k=4)
            
            if not docs:
                return "❌ No relevant information found in the documents for your question."
            
            # Enhanced prompt template
            prompt_template = """
            You are an expert document analyst. Provide a comprehensive answer based on the document context.
            
            Instructions:
            - Give detailed, accurate answers based solely on the provided context
            - Include specific quotes when relevant and cite the source
            - If the answer is not in the context, clearly state that
            - Structure your response with clear sections and bullet points
            - Be thorough and provide actionable insights
            
            Document Context:
            {context}
            
            Question: {question}
            
            Comprehensive Answer:
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
            return f"❌ Error analyzing documents: {str(e)}"

    # Image Processing Methods - NOW WITH PERFECT CONTINUOUS CHAT
    def analyze_image_perfect(self, question=""):
        """Perfect image analysis with enhanced prompts - PERFECT CONTINUOUS CHAT"""
        if st.session_state.current_image is None:
            return "❌ No image uploaded. Please upload an image first."
        
        if not self.models_ready:
            return "❌ AI vision models not available for image analysis."
        
        try:
            image = st.session_state.current_image
            
            if question:
                prompt = f"""
                You are an expert computer vision analyst. Analyze this image carefully and answer: {question}
                
                Provide a comprehensive response including:
                - Direct and specific answer to the question
                - Detailed visual observations supporting your answer
                - Any text content visible in the image (OCR)
                - Contextual information and background details
                - Professional insights and technical observations
                
                Be thorough, accurate, and provide actionable information.
                """
            else:
                prompt = """
                You are an expert computer vision analyst. Provide a comprehensive analysis of this image.
                
                Structure your analysis as follows:
                
                **🎯 Main Subject & Focus:**
                - Primary subject or focal point
                - Overall scene composition
                
                **👥 People & Objects:**
                - Detailed description of people, objects, and their relationships
                - Actions or activities taking place
                
                **📝 Text Content (OCR):**
                - All visible text, signs, labels, or writing
                - Document content if applicable
                
                **🎨 Visual Elements:**
                - Colors, lighting, and photographic style
                - Composition and artistic elements
                
                **📍 Setting & Context:**
                - Location, environment, and time context
                - Background and surroundings
                
                **🔍 Notable Details:**
                - Interesting, important, or unusual observations
                - Technical or safety considerations if relevant
                
                Be comprehensive, accurate, and professional in your analysis.
                """
            
            response = self.gemini_vision.generate_content([prompt, image])
            return f"🖼️ **Expert Image Analysis:**\n\n{response.text}"
            
        except Exception as e:
            return f"❌ Error analyzing image: {str(e)}"

    def logout_user(self):
        """Enhanced logout with confirmation"""
        # Store current username for goodbye message
        username = st.session_state.get('username', 'User')
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.success(f"👋 Thank you for using Elite AI Platform, {username}! Successfully logged out.")
        time.sleep(2)
        st.rerun()

    def run(self):
        """Main application runner with error handling"""
        try:
            if not st.session_state.authenticated:
                self.render_modern_landing_page()
            else:
                self.render_modern_dashboard()
        except Exception as e:
            st.error(f"❌ Application Error: {str(e)}")
            st.info("Please refresh the page and try again. If the issue persists, check your API keys.")

# Run Application
if __name__ == "__main__":
    try:
        app = ModernMultiModalPlatform()
        app.run()
    except Exception as e:
        st.error(f"❌ Critical Application Error: {str(e)}")
        st.info("Please check your environment setup and API keys.")
        
        # Debug information
        with st.expander("🔧 Debug Information"):
            st.code("""
            Required Environment Variables:
            GOOGLE_API_KEY=your_google_api_key_here
            
            Required Dependencies:
            pip install streamlit pandas pillow PyPDF2 google-generativeai python-docx 
            pip install langchain langchain-google-genai langchain-community faiss-cpu 
            pip install python-dotenv plotly nest-asyncio
            """)

