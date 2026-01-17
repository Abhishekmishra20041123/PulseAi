import os
from moviepy import VideoFileClip, CompositeVideoClip
import cv2
import numpy as np
import whisper
from dotenv import load_dotenv

load_dotenv()

# Explicitly set FFmpeg path for MoviePy 2.0
ffmpeg_path = os.getenv("FFMPEG_PATH")
if ffmpeg_path:
    # MoviePy 2.0 reads FFMPEG_BINARY from env
    os.environ["FFMPEG_BINARY"] = ffmpeg_path

# Initialize OpenCV Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class SubjectStabilizer:
    def __init__(self, window_size=25):
        self.window_size = window_size
        self.positions = []

    def get_stable_pos(self, current_pos):
        if current_pos is not None:
            self.positions.append(current_pos)
            if len(self.positions) > self.window_size:
                self.positions.pop(0)
        return sum(self.positions) / len(self.positions) if self.positions else 0.5

def get_face_center(frame):
    """ Detects face center in fractional coordinates. Expects RGB. """
    try:
        # MoviePy RGB -> OpenCV BGR
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x, y, w, h = faces[0]
            return (x + w/2) / frame.shape[1]
    except: pass
    return None

def detect_talking_head(clip, samples=8):
    duration = clip.duration
    times = np.linspace(0.1, duration - 0.1, samples)
    face_count = 0
    for t in times:
        if get_face_center(clip.get_frame(t)) is not None:
            face_count += 1
    return (face_count / samples) >= 0.3

def process_frame_standard(frame, text, t, duration, stabilizer, w, h, target_w, target_h, is_talking_head):
    """
    SINGLE-PASS frame processing. 
    Handles: Cropping/Reframing, Zoom, and Captions.
    This prevents memory layout issues between multiple transform calls.
    """
    try:
        # 1. Base Reframe/Crop
        if is_talking_head:
            # Face tracking crop
            current_face = get_face_center(frame)
            stable_x_pct = stabilizer.get_stable_pos(current_face)
            
            center_x = int(stable_x_pct * w)
            x1 = max(0, min(w - target_w, center_x - target_w // 2))
            # Crop to 9:16 area
            processed = frame[:, x1 : x1 + target_w].copy()
            # Resize to final 1080x1920
            processed = cv2.resize(processed, (1080, 1920))
        else:
            # Blurred Background Reframing
            # Canvas to hold the result
            canvas = np.zeros((1920, 1080, 3), dtype=np.uint8)
            
            # 1.1 Create Background
            bg = cv2.resize(frame, (1080, 1920))
            bg = cv2.GaussianBlur(bg, (95, 95), 0)
            
            # 1.2 Create Centerpiece (maintain aspect ratio)
            # scale h to 1920 * 0.7
            scaled_h = int(1920 * 0.75)
            scaled_w = int(w * (scaled_h / h))
            fg = cv2.resize(frame, (scaled_w, scaled_h))
            
            # 1.3 Composite
            y_off = (1920 - scaled_h) // 2
            x_off = (1080 - scaled_w) // 2
            canvas = bg
            canvas[y_off:y_off+scaled_h, x_off:x_off+scaled_w] = fg
            processed = canvas

        # 2. Add Cinematic Hook Zoom (Subtle 1.1x -> 1.0x over 1.5s)
        if t < 1.5:
            zoom = 1.1 - 0.1 * (t / 1.5)
            zh, zw = processed.shape[:2]
            new_h, new_w = int(zh * zoom), int(zw * zoom)
            zoomed = cv2.resize(processed, (new_w, new_h))
            # Center crop back to 1080x1920
            y1, x1 = (new_h - 1920) // 2, (new_w - 1080) // 2
            processed = zoomed[y1:y1+1920, x1:x1+1080]

        # 3. Fade In (0.5s)
        if t < 0.5:
            processed = (processed.astype(np.float32) * (t / 0.5)).astype(np.uint8)

        # 4. Draw Captions
        # Convert to BGR for OpenCV Drawing
        img_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.7
        thick = 3
        words = text.split()
        curr_idx = min(len(words)-1, int((t / duration) * len(words) * 1.4))
        
        # Wrapping
        lines, curr_line = [], []
        for word in words:
            if cv2.getTextSize(" ".join(curr_line + [word]), font, font_scale, thick)[0][0] < 900:
                curr_line.append(word)
            else:
                lines.append(curr_line); curr_line = [word]
        lines.append(curr_line)
        
        y_text = int(1920 * 0.15 * min(1.0, t/0.6)) # Slide in
        line_h = 100
        
        # BG Box for readability
        box_h = len(lines) * line_h + 40
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0, y_text - 80), (1080, y_text + box_h - 80), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0, img_bgr)

        word_count = 0
        for line_words in lines:
            line_str = " ".join(line_words)
            tw = cv2.getTextSize(line_str, font, font_scale, thick)[0][0]
            cx = (1080 - tw) // 2
            for word in line_words:
                w_str = word + " "
                ws = cv2.getTextSize(w_str, font, font_scale, thick)[0][0]
                color = (0, 255, 255) if word_count == curr_idx else (255, 255, 255)
                # Text with shadow
                cv2.putText(img_bgr, w_str, (cx+2, y_text+2), font, font_scale, (0,0,0), thick+2, cv2.LINE_AA)
                cv2.putText(img_bgr, w_str, (cx, y_text), font, font_scale, color, thick, cv2.LINE_AA)
                cx += ws
                word_count += 1
            y_text += line_h

        # Back to RGB and force memory alignment
        final_frame = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(final_frame, dtype=np.uint8)

    except Exception as e:
        # Fallback to raw frame if processing fails
        return np.ascontiguousarray(cv2.resize(frame, (1080, 1920)), dtype=np.uint8)

def generate_reel(video_path, start_time, end_time, hook, output_path):
    def to_seconds(t_str):
        if isinstance(t_str, (int, float)): return float(t_str)
        try:
            p = str(t_str).split(':')
            if len(p) == 3: return float(p[0])*3600 + float(p[1])*60 + float(p[2])
            if len(p) == 2: return float(p[0])*60 + float(p[1])
            return float(t_str)
        except: return 0.0

    start_s, end_s = to_seconds(start_time), to_seconds(end_time)
    full_clip = VideoFileClip(video_path)
    clip = full_clip.subclipped(start_s, min(end_s, full_clip.duration))
    
    # Pre-calculate settings
    is_talking_head = detect_talking_head(clip)
    w, h = clip.size
    target_w = int(h * 9/16)
    if target_w % 2 != 0: target_w -= 1
    
    stabilizer = SubjectStabilizer()
    duration = clip.duration

    # THE CORE FIX: Consolidate everything into ONE transform that produces 1080x1920 frames
    def master_processor(get_frame, t):
        frame = get_frame(t)
        return process_frame_standard(frame, hook, t, duration, stabilizer, w, h, target_w, h, is_talking_head)

    # Force the clip to have exactly the right size metadata
    final_clip = clip.transform(master_processor)
    # Ensure MoviePy knows the final size is 1080x1920
    final_clip.size = (1080, 1920)

    print(f"Exporting Professional Reel: {output_path}")
    final_clip.write_videofile(
        output_path, 
        codec="libx264", 
        audio_codec="aac", 
        fps=24,
        bitrate="6000k",
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    )
    
    final_clip.close()
    clip.close()
    full_clip.close()
    return output_path

if __name__ == "__main__": pass
