from moviepy import TextClip
import os

try:
    clip = TextClip(text="Test", font="Arial", font_size=70, color="white")
    print("TextClip created successfully")
except Exception as e:
    print(f"TextClip failed: {e}")
    print("This usually means ImageMagick is not installed or not configured for MoviePy.")
