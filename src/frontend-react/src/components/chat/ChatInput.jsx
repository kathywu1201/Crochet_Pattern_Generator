'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, CameraAltOutlined } from '@mui/icons-material';
import IconButton from '@mui/material/IconButton';

// Styles
import styles from './ChatInput.module.css';

export default function ChatInput({
    onSendMessage,
    selectedModel,
    onModelChange,
    disableModelSelect = false
}) {
    // Component States
    const [message, setMessage] = useState('');
    const [selectedImage, setSelectedImage] = useState(null);
    const textAreaRef = useRef(null);
    const fileInputRef = useRef(null);

    const adjustTextAreaHeight = () => {
        const textarea = textAreaRef.current;
        if (textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = `${textarea.scrollHeight}px`;
        }
    };

    // Setup Component
    useEffect(() => {
        adjustTextAreaHeight();
    }, [message]);

    // Handlers
    const handleMessageChange = (e) => {
        setMessage(e.target.value);
    };
    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            if (e.shiftKey) {
                // Shift + Enter: add new line
                return;
            } else {
                // Enter only: submit
                e.preventDefault();
                handleSubmit();
            }
        }
    };
    const handleSubmit = () => {

        if (message.trim() || selectedImage) {
            console.log('Submitting message:', message);
            const newMessage = {
                content: message.trim(),
                image: selectedImage?.preview || null
            };

            // Send the message
            onSendMessage(newMessage);

            // Reset
            setMessage('');
            setSelectedImage(null);
            if (textAreaRef.current) {
                textAreaRef.current.style.height = 'auto';
            }
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
        }
    };
    const handleImageClick = () => {
        fileInputRef.current?.click();
    };
    const handleImageChange = (e) => {
        const file = e.target.files?.[0];
        if (file) {
            if (file.size > 5000000) { // 5MB limit
                alert('File size should be less than 5MB');
                return;
            }

            const reader = new FileReader();
            reader.onloadend = () => {
                setSelectedImage({
                    file: file,
                    preview: reader.result
                });
            };
            reader.readAsDataURL(file);
        }
    };
    const handleModelChange = (event) => {
        onModelChange(event.target.value);
    };

    const removeImage = () => {
        setSelectedImage(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <section>
            <div className={styles.inputTip}>Step 1: Upload an Image</div>
            <div className={styles.chatInputContainer}>
                {selectedImage && (
                    <div className={styles.imagePreview}>
                        <img
                            src={selectedImage.preview}
                            alt="Preview"
                        />
                        <button
                            className={styles.removeImageBtn}
                            onClick={removeImage}
                        >
                            ×
                        </button>
                    </div>
                )}
                <div className={styles.inputControls}>
                    <div className={styles.leftControls}>
                        <input
                            type="file"
                            ref={fileInputRef}
                            className={styles.hiddenFileInput}
                            accept="image/*"
                            onChange={handleImageChange}
                        />
                        <IconButton aria-label="camera" className={styles.iconButton} onClick={handleImageClick}>
                            <CameraAltOutlined />
                        </IconButton>
                    </div>
                </div>
            </div>
            <div className={styles.inputTip}>Step 2: Enter a Prompt</div>
            <div className={styles.chatInputContainer}>
                <div className={styles.textareaWrapper}>
                    <textarea
                        ref={textAreaRef}
                        className={styles.chatInput}
                        placeholder="Ex: Show me how to make this blue, heart-shape cup mat"
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        onKeyDown={(e) => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handleSubmit();
                            }
                        }}
                        rows={1}
                    />
                </div>
                <div className={styles.rightControls}>
                        <select
                            className={styles.modelSelect}
                            value={selectedModel}
                            onChange={handleModelChange}
                            disabled={disableModelSelect}
                        >
                            <option value="llm">Yarn Master Assistant (LLM)</option>
                            {/* <option value="llm-cnn">Yarn Master Assistant Assistant (LLM + CNN)</option> */}
                            <option value="llm-rag">Crochet Expert (RAG)</option>
                            {/* <option value="llm-agent">Crochet Expert (Agent)</option> */}
                        </select>
                </div>
            </div>
            <button
                className={`${styles.submitButton} ${message.trim() ? styles.active : ''}`}
                onClick={handleSubmit}
                disabled={!message.trim() && !selectedImage}
            >Submit
            </button>
        </section>
    )
}