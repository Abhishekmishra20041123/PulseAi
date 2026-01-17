import os
import uuid
import json
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from engine.analyzer import analyze_video
from engine.processor import generate_reel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/output", StaticFiles(directory="output"), name="output")

# In-memory job status (for a real app, use a DB or Redis)
jobs = {}

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

from pydantic import BaseModel

class LinkRequest(BaseModel):
    link: str

@app.post("/process-link")
async def process_link(request: LinkRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    link = request.link
    
    # Basic Google Drive link extraction (simplified)
    # In a real app, use a library like gdown or google-api-python-client
    if "drive.google.com" not in link:
        raise HTTPException(status_code=400, detail="Only Google Drive links are supported currently")
    
    file_path = f"uploads/{job_id}_video.mp4"
    jobs[job_id] = {"status": "downloading", "reels": []}
    
    background_tasks.add_task(download_and_process_task, job_id, link, file_path)
    
    return {"job_id": job_id}

async def download_and_process_task(job_id: str, link: str, file_path: str):
    try:
        import gdown
        print(f"Downloading from link: {link}")
        
        # Handle Google Drive direct download
        if "drive.google.com" in link:
            # gdown handles drive links automatically
            output = gdown.download(link, file_path, quiet=False, fuzzy=True)
            if not output:
                # Fallback to local test path if gdown fails (useful for local development)
                local_test_path = r"C:\Users\abhis\Downloads\Input video for ByteSize Hackathon.mp4"
                if os.path.exists(local_test_path):
                    import shutil
                    shutil.copy(local_test_path, file_path)
                    print("Used local fallback path")
                else:
                    jobs[job_id] = {"status": "failed", "error": f"Failed to download from link and no local fallback found"}
                    return
        else:
             # Try direct download anyway if it's not drive
             import requests
             response = requests.get(link, stream=True)
             if response.status_code == 200:
                 with open(file_path, 'wb') as f:
                     for chunk in response.iter_content(chunk_size=8192):
                         f.write(chunk)
             else:
                jobs[job_id] = {"status": "failed", "error": "Only Google Drive links or direct video URLs are supported"}
                return

        await process_video_task(job_id, file_path)
        
    except Exception as e:
        print(f"Error in download_and_process_task: {e}")
        jobs[job_id] = {"status": "failed", "error": str(e)}

@app.post("/upload")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):

    job_id = str(uuid.uuid4())
    file_path = f"uploads/{job_id}_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    jobs[job_id] = {"status": "processing", "reels": []}
    
    background_tasks.add_task(process_video_task, job_id, file_path)
    
    return {"job_id": job_id}

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

async def process_video_task(job_id: str, file_path: str):
    try:
        # 1. Analyze Video
        jobs[job_id]["status"] = "Analyzing video..."
        
        def update_status(msg):
            jobs[job_id]["status"] = msg
            
        analysis_result_raw = analyze_video(file_path, status_callback=update_status)
        
        # Extract JSON from response
        if "```json" in analysis_result_raw:
            analysis_result_raw = analysis_result_raw.split("```json")[1].split("```")[0]
        
        analysis_result = json.loads(analysis_result_raw)
        
        # 2. Generate Reels
        reels = []
        total = len(analysis_result)
        for i, moment in enumerate(analysis_result):
            jobs[job_id]["status"] = f"Generating Reel {i+1} of {total}..."
            output_filename = f"reel_{job_id}_{i}.mp4"
            output_path = f"output/{output_filename}"
            
            generate_reel(
                file_path, 
                moment['start'], 
                moment['end'], 
                moment['hook'], 
                output_path
            )
            reels.append({
                "url": f"/output/{output_filename}",
                "hook": moment['hook'],
                "reason": moment['reason']
            })
            # Partial completion to show user
            jobs[job_id]["reels"] = reels
        
        jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        print(f"Error processing job {job_id}: {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
