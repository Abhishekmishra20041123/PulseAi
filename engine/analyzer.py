import os
import time
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv
import librosa
import numpy as np

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Groq for fallback
from groq import Groq
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Multi-model rotation fallback
# Using specific versions to avoid 404s
MODELS = ["gemini-2.0-flash", "gemini-2.0-flash-exp", "gemini-1.5-flash-002", "gemini-1.5-flash", "gemini-1.5-pro-002"]

def get_audio_spikes(video_path):
    """
    Identifies audio energy spikes to help the AI find high-impact moments.
    """
    try:
        y, sr = librosa.load(video_path, sr=None)
        # Calculate short-term energy
        hop_length = 512
        frame_length = 1024
        energy = np.array([
            np.sum(np.abs(y[i:i+frame_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        
        # Find spikes (simple thresholding)
        threshold = np.mean(energy) * 2.0
        spikes = np.where(energy > threshold)[0]
        
        # Convert frame indices to seconds
        spike_times = spikes * hop_length / sr
        
        # Cluster spikes that are close together (within 2s)
        clusters = []
        if len(spike_times) > 0:
            current_cluster = [spike_times[0]]
            for i in range(1, len(spike_times)):
                if spike_times[i] - current_cluster[-1] < 2.0:
                    current_cluster.append(spike_times[i])
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [spike_times[i]]
            clusters.append(np.mean(current_cluster))
            
        print(f"Detected {len(clusters)} audio spikes.")
        return clusters
    except Exception as e:
        print(f"Librosa failed, using MoviePy fallback: {e}")
        # Fallback to extremely basic detection or empty if everything fails
        return []

def analyze_video(video_path, status_callback=None):
    """
    Analyzes video using Gemini.
    status_callback: function(str) -> None, to update UI status.
    """
    print(f"Analyzing video: {video_path}")
    
    # Upload the file
    video_file = client.files.upload(file=video_path)
    print(f"Completed upload: {video_file.uri}")

    # Wait for the file to be processed by Gemini (crucial for larger videos)
    while video_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)
        
    if video_file.state.name == "FAILED":
        raise Exception("Video processing failed on Gemini server")

    audio_spikes = get_audio_spikes(video_path)
    spike_info = f"Audio spikes detected at these seconds: {audio_spikes[:10]}" if audio_spikes else "No distinct audio spikes detected."

    prompt = f"""
    Analyze this video and identify EXACTLY 5 high-impact, emotional, or viral moments that would make great short-form reels (TikTok/Reels/Shorts).
    
    {spike_info}
    
    IMPORTANT: 
    - Each moment MUST be unique and cover a different part of the video.
    - DO NOT repeat timestamps or generate identical hooks.
    - Focus on the most engaging part of each segment.
    - Provide exactly 5 distinct results.
    
    For each moment, provide:
    1. Start and end timestamps (formatted as MM:SS or seconds).
    2. A catchy, high-engagement 'hook' headline for the caption.
    3. A brief reason why this moment is viral.
    
    Respond STRICTLY in JSON format like this:
    [
      {{"start": "00:15", "end": "00:45", "hook": "The Secret to Peak Performance", "reason": "High energy insight"}},
      ... (exactly 5 items)
    ]
    """

    last_error = None
    for model_name in MODELS:
        print(f"Attempting analysis with model: {model_name}...")
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[video_file, prompt],
                    config=types.GenerateContentConfig(
                        max_output_tokens=2048,
                        temperature=0.8,  # Increased for more variety
                    )
                )
                # Clean up the uploaded file
                client.files.delete(name=video_file.name)
                return response.text
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "429" in error_str or "quota" in error_str:
                    wait_time = 20 * (attempt + 1)
                    msg = f"API high traffic ({model_name}), retrying in {wait_time}s..."
                    print(msg)
                    if status_callback: status_callback(msg)
                    time.sleep(wait_time)
                else:
                    print(f"Error with {model_name}: {e}. Trying next model...")
                    break # Break to next model
    
    # Final cleanup if all models fail
    try:
        client.files.delete(name=video_file.name)
    except:
        pass
        
    print("Gemini exhausted. Trying Groq Fallback...")
    return analyze_with_groq_fallback(video_path, status_callback)

def analyze_with_groq_fallback(video_path, status_callback=None):
    """
    Fallback method: Extract audio -> Transcribe (Groq Whisper) -> Analyze (Groq Llama 3)
    """
    print("Falling back to Groq Llama 3 analysis...")
    if status_callback: status_callback("Gemini busy. Falling back to Groq (Llama 3)...")
    
    # 1. Extract Audio to MP3
    audio_path = video_path + ".mp3"
    # 1. Extract Audio to MP3
    audio_path = video_path + ".mp3"
    from moviepy import VideoFileClip
    try:
        if status_callback: status_callback("Extracting audio for analysis...")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(audio_path, logger=None)
        clip.close()
        
        # 2. Transcribe with Groq Whisper
        if status_callback: status_callback("Transcribing audio (Groq Whisper)...")
        with open(audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio_path, file.read()),
                model="distil-whisper-large-v3-en",
                response_format="verbose_json",
            )
        
        # 3. Analyze Transcript with Llama 3
        if status_callback: status_callback("Analyzing transcript for viral moments...")
        
        transcript_text = ""
        # Create a text with timestamps for the LLM
        for segment in transcription.segments:
            start = int(segment['start'])
            text = segment['text']
            transcript_text += f"[{start}s] {text}\n"

        prompt = f"""
        Analyze this video transcript and identify EXACTLY 5 viral short-form video moments.
        
        TRANSCRIPT:
        {transcript_text[:15000]} # Limit context
        
        Return JSON format:
        [
          {{"start": 10, "end": 30, "hook": "Catchy Hook", "reason": "Why needed"}}
        ]
        """
        
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a viral content expert. Return strictly JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            response_format={"type": "json_object"}
        )
        
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Groq Fallback Failed: {e}")
        raise e
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    # Test
    # print(analyze_video("path/to/test.mp4"))
    pass
