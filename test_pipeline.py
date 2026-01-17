import os
import json
from engine.analyzer import analyze_video
from engine.processor import generate_reel
from dotenv import load_dotenv

load_dotenv()

video_path = r"C:\Users\abhis\Downloads\Input video for ByteSize Hackathon.mp4"

def test_pipeline():
    print(f"Starting test with video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    try:
        # 1. Analyze
        print("Step 1: Analyzing video...")
        try:
            analysis_result_raw = analyze_video(video_path)
            print(f"Raw Analysis Result: {analysis_result_raw}")
            
            # Extract JSON
            if "```json" in analysis_result_raw:
                analysis_result_raw = analysis_result_raw.split("```json")[1].split("```")[0]
            
            analysis_result = json.loads(analysis_result_raw)
        except Exception as e:
            print(f"Gemini API failed or rate limited: {e}")
            print("Proceeding with mock data for verification of the rest of the pipeline...")
            analysis_result = [
                {"start": 5, "end": 15, "hook": "The Power of Focus", "reason": "High energy moment"},
                {"start": 20, "end": 30, "hook": "Why Consistency Matters", "reason": "Profound insight"},
                {"start": 40, "end": 50, "hook": "The consistency", "reason": "High energy moment"},
                {"start": 25, "end": 35, "hook": "Why Goal Matters", "reason": "Profound insight"},
                {"start": 60, "end": 90, "hook": "The Power of hardwork", "reason": "High energy moment"},
                {"start": 100, "end": 130, "hook": "The hardwork and Consistency Matters", "reason": "Profound insight"}
            ]



        # 2. Generate Reels
        print(f"Step 2: Generating {len(analysis_result)} reels...")
        if not os.path.exists("output"):
            os.makedirs("output")
            
        for i, moment in enumerate(analysis_result):
            output_path = f"output/test_reel_{i}.mp4"
            print(f"\n--- Generating reel {i+1} ---")
            print(f"Hook: {moment['hook']}")
            print(f"Time: {moment['start']} to {moment['end']}")
            
            try:
                generate_reel(
                    video_path, 
                    moment['start'], 
                    moment['end'], 
                    moment['hook'], 
                    output_path
                )
                print(f"Successfully generated: {output_path}")
            except Exception as e:
                print(f"Failed to generate reel {i+1}: {e}")
                import traceback
                traceback.print_exc()

    except Exception as e:
        import traceback
        print(f"Pipeline failed: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()


