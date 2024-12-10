import os
from fastapi import APIRouter, Header, Query, Body, HTTPException
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
import uuid
import time
from datetime import datetime
import mimetypes
from pathlib import Path
from api.utils.llm_utils import chat_sessions, create_chat_session, generate_chat_response, rebuild_chat_session
from api.utils.chat_utils import ChatHistoryManager

# Define Router
router = APIRouter()

# Initialize chat history manager and sessions
chat_manager = ChatHistoryManager(model="llm")

@router.get("/chats")
async def get_chats(x_session_id: str = Header(None, alias="X-Session-ID"), limit: Optional[int] = None):
    """Get all chats, optionally limited to a specific number"""
    print("x_session_id:", x_session_id)
    return chat_manager.get_recent_chats(x_session_id, limit)

@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str, x_session_id: str = Header(None, alias="X-Session-ID")):
    """Get a specific chat by ID"""
    print("x_session_id:", x_session_id)
    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat

@router.post("/chats")
async def start_chat_with_llm(message: Dict, x_session_id: str = Header(None, alias="X-Session-ID")):
    print("content:", message["content"])
    print("x_session_id:", x_session_id)
    """Start a new chat with an initial message"""
    chat_id = str(uuid.uuid4())
    current_time = int(time.time())
    
    # Create a new chat session
    chat_session = create_chat_session()
    chat_sessions[chat_id] = chat_session
    
    # Add ID and role to the user message
    message["message_id"] = str(uuid.uuid4())
    message["role"] = "user"
    
    # Generate response
    assistant_response = generate_chat_response(chat_session, message)
    
    # Create chat response
    title = message.get("content")
    if title == "":
        title =  "Image chat"
    title = title[:50] + "..."
    chat_response = {
        "chat_id": chat_id,
        "title": title,
        "dts": current_time,
        "messages": [
            message,
            {
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": assistant_response
            }
        ]
    }
    
    # Save chat
    chat_manager.save_chat(chat_response, x_session_id)
    return chat_response

@router.post("/chats/{chat_id}")
async def continue_chat_with_llm(chat_id: str, message: Dict, x_session_id: str = Header(None, alias="X-Session-ID")):
    print("content:", message["content"])
    print("x_session_id:", x_session_id)
    """Add a message to an existing chat"""
    chat = chat_manager.get_chat(chat_id, x_session_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    # Get or rebuild chat session
    chat_session = chat_sessions.get(chat_id)
    if not chat_session:
        chat_session = rebuild_chat_session(chat["messages"])
        chat_sessions[chat_id] = chat_session
    
    # Update timestamp
    current_time = int(time.time())
    chat["dts"] = current_time
    
    # Add message ID and role
    message["message_id"] = str(uuid.uuid4())
    message["role"] = "user"
    
    # Generate response
    assistant_response = generate_chat_response(chat_session, message)
    
    # Add messages
    chat["messages"].append(message)
    chat["messages"].append({
        "message_id": str(uuid.uuid4()),
        "role": "assistant",
        "content": assistant_response
    })
    
    # Save updated chat
    chat_manager.save_chat(chat, x_session_id)
    return chat

@router.get("/images/{chat_id}/{message_id}.png")
async def get_chat_image(chat_id: str, message_id: str):
    """
    Serve an image from the chat history.
    
    Args:
        chat_id: The chat ID
        message_id: The message ID
    
    Returns:
        FileResponse: The image file with appropriate content type
    """
    try:
        # Construct the image path
        image_path = os.path.join(
            chat_manager.images_dir,
            chat_id,
            f"{message_id}.png"
        )
        
        # Verify the path exists and is within the images directory
        image_path = Path(image_path).resolve()
        images_dir = Path(chat_manager.images_dir).resolve()
        
        # Security check: ensure the requested file is within the images directory
        if not str(image_path).startswith(str(images_dir)):
            raise HTTPException(
                status_code=403,
                detail="Access denied"
            )
        
        if not image_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Image not found"
            )
        
        # Determine content type
        content_type, _ = mimetypes.guess_type(str(image_path))
        if not content_type:
            content_type = "application/octet-stream"
        
        return FileResponse(
            path=image_path,
            media_type=content_type
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error serving image: {str(e)}"
        )