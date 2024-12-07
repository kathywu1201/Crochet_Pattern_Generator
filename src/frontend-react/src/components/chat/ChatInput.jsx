'use client';

import { useState, useRef, useEffect } from 'react';
import { CameraAltOutlined } from '@mui/icons-material';
import IconButton from '@mui/material/IconButton';
import imageCompression from 'browser-image-compression';

// Styles
import styles from './ChatInput.module.css';

export default function ChatInput({
    onSendMessage,
    selectedModel,
    onModelChange,
    disableModelSelect = false, // Determines behavior for input vs. chat page
}) {
    const [message, setMessage] = useState('');
    const [selectedImage, setSelectedImage] = useState(null);
    const textAreaRef = useRef(null);
    const fileInputRef = useRef(null);

    // Redirect to input page when "New Pattern" button is clicked
    const handleNewPattern = () => {
        window.location.href = '/chat'; // Redirects to the input page
    };

    const handleModelChange = (event) => {
        onModelChange(event.target.value);
    };

    const handleImageChange = async (e) => {
        const file = e.target.files?.[0];
        if (file) {
            try {
                const options = {
                    maxSizeMB: 0.25,
                    maxWidthOrHeight: 512,
                    useWebWorker: true
                };

                const compressedFile = await imageCompression(file, options);
                const reader = new FileReader();

                reader.onloadend = () => {
                    setSelectedImage({
                        file: compressedFile,
                        preview: reader.result
                    });
                };

                reader.readAsDataURL(compressedFile);
            } catch (error) {
                console.error('Compression error:', error);
                setError(error);
            }
        }
    };

    return (
        <section>
            {!disableModelSelect ? (
                <>
                    {/* Input Page: Image Upload and Text Prompt */}
                    <div className={styles.inputTip}>Step 1: Upload an Image</div>
                    <div className={styles.chatInputContainer}>
                        {selectedImage && (
                            <div className={styles.imagePreview}>
                                <img src={selectedImage.preview} alt="Preview" />
                                <button
                                    className={styles.removeImageBtn}
                                    onClick={() => setSelectedImage(null)}
                                >
                                    ×
                                </button>
                            </div>
                        )}
                        <div className={styles.inputControls}>
                            <input
                                type="file"
                                ref={fileInputRef}
                                className={styles.hiddenFileInput}
                                accept="image/*"
                                onChange={handleImageChange}
                            />
                            <IconButton
                                aria-label="Upload an image"
                                className={styles.iconButton}
                                onClick={() => fileInputRef.current?.click()}
                            >
                                <CameraAltOutlined />
                            </IconButton>
                        </div>
                    </div>
                    <div className={styles.inputTip}>Step 2: Enter a Prompt</div>
                    <div className={styles.chatInputContainer}>
                        <textarea
                            ref={textAreaRef}
                            className={styles.chatInput}
                            placeholder="Describe your pattern in detail... (e.g. Can you show me how to crochet this blue heart coaster which has around 6 rows.)"
                            value={message}
                            onChange={(e) => setMessage(e.target.value)}
                        />
                        <select
                            className={styles.modelSelect}
                            value={selectedModel}
                            onChange={handleModelChange}
                            disabled={disableModelSelect}
                        >
                            <option value="llm-llama">Yarn Bachelor (Llama)</option>
                            <option value="llm">Yarn Master (LLM)</option>
                            <option value="llm-rag">Yarn Phd (RAG)</option>
                        </select>
                    </div>
                    <button
                        className={styles.submitButton}
                        onClick={() => onSendMessage({ content: message.trim(), image: selectedImage?.preview || null })}
                        disabled={!message.trim() && !selectedImage}
                    >
                        Submit
                    </button>
                </>
            ) : (
                /* Chat Page: Only "New Pattern" Button */
                <button
                    className={styles.submitButton}
                    onClick={handleNewPattern}
                >
                    New Pattern
                </button>
            )}
        </section>
    );
}
