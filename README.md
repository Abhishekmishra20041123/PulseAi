# PulsePoint AI

PulsePoint AI is a platform where users upload long-form videos and the app uses GenAI and Multimodal models to automatically generate viral short-form reels.

## Demo
![[Demo Video](https://drive.google.com/file/d/1bylBG49cQqduVboykUhKUeQs5OP3P8Wk/view?usp=sharing)](assets/demo.mp4)

## Features
- **Identify "Emotional Peaks"**: Detect high-energy or profound moments using audio spikes and sentiment analysis.
- **Smart-Crop to Vertical**: Uses **SubjectStabilizer** (rolling average) to track the speaker's face smoothly without jitter.
- **Cinematic Effects**: Adds professional fade-ins, subtle zooms, and viral-style transitions.
- **Dynamic Captions**: Word-by-word highlighted captions with premium animations and drop shadows.
- **Robust AI Analysis**: Multi-model fallback (Gemini 2.0 → 1.5 → Groq Llama 3) ensures 100% reliability.
- **1080x1920 Standard**: Enforces strict vertical HD output for compatibility with all social platforms.

## Tech Stack
- **Backend**: FastAPI
- **AI**: Google Gemini (2.0/1.5), Groq (Llama 3, Whisper)
- **Video Processing**: MoviePy, OpenCV (Face Tracking), Librosa
- **Frontend**: HTML5, Vanilla CSS (Glassmorphism), JavaScript

## Setup
1. Clone the repository.
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `.\venv\Scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Create a `.env` file and add your `GEMINI_API_KEY`.
6. Run the server: `python main.py`

## Usage
1. Open your browser and go to `http://localhost:8000`.
2. Upload a long-form video.
3. Wait for the AI to process and generate reels.
4. Download and share!
