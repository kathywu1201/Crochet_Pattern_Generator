import os
from fastapi import APIRouter, Query, Body, HTTPException
from fastapi.responses import FileResponse
from typing import Dict, Any, Optional
import glob
import json
import traceback

# Define Router
router = APIRouter()

# Podcasts should be downloaded from a GCS bucket (ML Task generates and saves into a bucket)
# We will assume the Podcasts have already been downloaded into "podcasts" folder locally
data_folder = "podcasts"

@router.get("/")
async def get_podcasts(limit: Optional[int] = None):
    """Get all podcasts, optionally limited to a specific number"""
    podcasts = []
    data_files = glob.glob(os.path.join(data_folder,"*.json"))
    for filepath in data_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)        
                podcasts.append(chat_data)
        except Exception as e:
            print(f"Error loading podcasts from {filepath}: {str(e)}")
            traceback.print_exc()

    # Sort by dts
    podcasts.sort(key=lambda x: x.get('dts', 0), reverse=True)
    if limit:
        return podcasts[:limit]

    return podcasts

@router.get("/{podcast_id}")
async def get_podcast(podcast_id: str):
    """Get a specific ID"""
    filepath = os.path.join(data_folder,f"{podcast_id}.json")
    with open(filepath, 'r', encoding='utf-8') as f:
        podcast = json.load(f) 
    if not podcast:
        raise HTTPException(status_code=404, detail="Podcast not found")
    return podcast

@router.get("/audio/{audio_name}")
async def get_podcast_audio(audio_name: str):
    """
    Serve the MP3 file for a specific podcast episode
    """
    try:
        # Construct the file path - adjust the file naming convention as needed
        audio_path = os.path.join(data_folder,"assets",audio_name)
        
        if not os.path.exists(audio_path):
            raise HTTPException(status_code=404, detail="Podcast audio not found")
            
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Disposition": f"attachment; filename={audio_name}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))