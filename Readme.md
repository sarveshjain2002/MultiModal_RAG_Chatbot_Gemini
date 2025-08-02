
### 📦 Project Environment Setup Guide

#### 🛠️ 1. Create a Virtual Environment

Open your **Command Prompt** and run the following command:

python -m venv myenv


#### 🚀 2. Activate the Environment

* On **Windows**:

myenv\Scripts\activate

#### 📄 3. Install Required Dependencies

Ensure you are inside the activated virtual environment, then run:

pip install -r requirements.txt

#### 🔑 4. Google API Key Setup

To use Google API services, get your API key from the following link:
🌐 [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

Then create a file named `.env` in your project root directory and add your API key like this:

GOOGLE_API_KEY=your_api_key_here



#### ▶️ 5. Run the Application

To start your project using **Streamlit**, execute:

streamlit run app.py

