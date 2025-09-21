🌱 MindfulYouth – AI Mental Wellness Assistant

<img width="119" height="28" alt="image" src="https://github.com/user-attachments/assets/215152a3-ec31-4b82-93a4-8fcb0866d742" />


<img width="145" height="28" alt="image" src="https://github.com/user-attachments/assets/6b6f356d-5636-4864-911d-fa8799d150b9" />




<img width="249" height="28" alt="image" src="https://github.com/user-attachments/assets/4e913b04-b997-4514-98c9-ff9185f150d3" />


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

