🌱 MindfulYouth – AI Mental Wellness Assistant

<img width="119" height="28" alt="image" src="https://github.com/user-attachments/assets/4b25f96c-2e59-4ed1-8e67-583d0b8c2eec" />


MindfulYouth is an AI-powered mental wellness assistant that offers real-time emotion detection, empathetic responses, voice input, and both offline (Ollama) & online (Gemini) AI support. Built with Streamlit, it helps users explore AI, emotional awareness, and interactive chat experiences.

✨ Features

🤖 Dual AI Mode: Supports both online (Gemini) and offline (Ollama) AI models.

😊 Emotion Detection: Real-time analysis of user emotions using Hugging Face transformers.

🎙️ Voice Input: Allows users to interact via speech-to-text functionality.

🎨 Beautiful UI: Modern, responsive interface built with Streamlit.

🔒 Privacy Focused: Optional offline mode ensures user data privacy.

⚡ Real-time Interaction: Instant responses with emotion visualization.

🚀 Quick Start
Prerequisites

Python 3.8+

Ollama (for offline mode)

Google Gemini API key (for online mode)

Installation

Clone the repository
git clone https://github.com/techanand8/mindmap_ai.git
cd mindmap_ai


Create a virtual environment
python -m venv python-env
source python-env/bin/activate  # On Windows, use `python-env\Scripts\activate`

Install dependencies
pip install -r requirements.txt

Set up environment variables

For Gemini API key:export GEMINI_API_KEY="your_api_key_here"  # On Windows, use `set GEMINI_API_KEY=your_api_key_here`

for offline model ollama download it from website and run it using ollama run any model name...

Run the application
streamlit run app.py

