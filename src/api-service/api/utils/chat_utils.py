import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import shutil
import glob
import base64
import traceback
import io
        
class ChatHistoryManager:
    def __init__(self, model, history_dir: str = "chat-history"):
        """Initialize the chat history manager with the specified directory"""
        self.model = model
        self.history_dir = os.path.join(history_dir, model)
        self.images_dir = os.path.join(self.history_dir, "images")
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure the chat history directory exists"""
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
    
    def _get_chat_filepath(self, chat_id: str, session_id: str) -> str:
        """Get the full file path for a chat JSON file"""
        return os.path.join(self.history_dir, session_id, f"{chat_id}.json")
    
    def _save_image(self, chat_id: str, message_id: str, image_data: str) -> str:
        """
        Save image data to a file and return the relative path.
        
        Args:
            chat_id: The chat ID
            message_id: The message ID
            image_data: Base64 encoded image data
        
        Returns:
            str: Relative path to the saved image
        """
        # Create chat-specific image directory
        chat_images_dir = os.path.join(self.images_dir, chat_id)
        os.makedirs(chat_images_dir, exist_ok=True)
        
        # Save image to file
        image_path = os.path.join(chat_images_dir, f"{message_id}.png")
        try:
            # Extract the actual base64 data and mime type
            base64_string = image_data
            if ',' in base64_string:
                header, base64_data = base64_string.split(',', 1)
                mime_type = header.split(':')[1].split(';')[0]
            else:
                base64_data = base64_string
                mime_type = 'image/jpeg'  # default to JPEG if no header
            
            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_data)
            
            with open(image_path, 'wb') as f:
                f.write(image_bytes)
            
            # Return relative path from chat history root
            return os.path.relpath(image_path, self.history_dir)
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            traceback.print_exc()
            return ""

    def _load_image(self, relative_path: str) -> Optional[str]:
        """
        Load image data from file and return as base64.
        
        Args:
            relative_path: Relative path to the image from chat history root
        
        Returns:
            Optional[str]: Base64 encoded image data or None if loading fails
        """
        full_path = os.path.join(self.history_dir, relative_path)
        try:
            if os.path.exists(full_path):
                with open(full_path, 'rb') as f:
                    image_bytes = f.read()
                return base64.b64encode(image_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            traceback.print_exc()
        return None
    
    def save_chat(self, chat_to_save: Dict, session_id: str) -> None:
        """Save a chat to both memory and file, handling images separately"""
        chat_dir = os.path.join(self.history_dir,session_id)
        os.makedirs(chat_dir, exist_ok=True)
        
        # Process messages to save images separately
        for message in chat_to_save["messages"]:
            if "image" in message and message["image"] is not None:
                #print("image:",message["image"])
                # Save image and replace with path
                image_path = self._save_image(
                    chat_to_save["chat_id"],
                    message["message_id"],
                    message["image"]
                )
                if image_path:
                    message["image_path"] = image_path
                del message["image"]
        
        # Save chat data
        filepath = self._get_chat_filepath(chat_to_save["chat_id"], session_id)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(chat_to_save, f, indent=2, ensure_ascii=False)    
        except Exception as e:
            print(f"Error saving chat {chat_to_save['chat_id']}: {str(e)}")
            traceback.print_exc()
            raise e

    def get_chat(self, chat_id: str, session_id: str) -> Optional[Dict]:
        """Get a specific chat by ID"""
        filepath = os.path.join(self.history_dir,session_id,f"{chat_id}.json")
        chat_data = {}
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)        
        except Exception as e:
            print(f"Error loading chat history from {filepath}: {str(e)}")
            traceback.print_exc()
        return chat_data
    
    def get_recent_chats(self, session_id: str, limit: Optional[int] = None) -> List[Dict]:
        """Get recent chats, optionally limited to a specific number"""        
        chat_dir = os.path.join(self.history_dir,session_id)
        os.makedirs(chat_dir, exist_ok=True)
        recent_chats = []
        chat_files = glob.glob(os.path.join(chat_dir,"*.json"))
        for filepath in chat_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    chat_data = json.load(f)        
                    recent_chats.append(chat_data)
            except Exception as e:
                print(f"Error loading chat history from {filepath}: {str(e)}")
                traceback.print_exc()

        # Sort by dts
        recent_chats.sort(key=lambda x: x.get('dts', 0), reverse=True)
        if limit:
            return recent_chats[:limit]

        return recent_chats
