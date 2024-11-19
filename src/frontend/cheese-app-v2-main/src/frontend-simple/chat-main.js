// chat-main.js
const BASE_API_URL = 'http://localhost:9000';
function uuid() {
    const newUuid = ([1e7] + -1e3 + -4e3 + -8e3 + -1e11).replace(/[018]/g, (c) =>
        (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16),
    )
    return newUuid;
}

// Create an axios instance with base configuration
const api = axios.create({
    baseURL: BASE_API_URL
});
const sessionId = uuid();
// Add request interceptor to include session ID in headers
api.interceptors.request.use((config) => {
    if (sessionId) {
        config.headers['X-Session-ID'] = sessionId;
    }
    return config;
}, (error) => {
    return Promise.reject(error);
});

const DataService = {
    GetChat: async function (model, chat_id) {
        const response = await api.get(BASE_API_URL + "/" + model + "/chats/" + chat_id);
        return response.data;
    },
    StartChatWithLLM: async function (model, message) {
        const response = await api.post(BASE_API_URL + "/" + model + "/chats/", message);
        return response.data;
    },
    ContinueChatWithLLM: async function (model, chat_id, message) {
        const response = await api.post(BASE_API_URL + "/" + model + "/chats/" + chat_id, message);
        return response.data;
    },
    GetChatMessageImage: function (model, image_path) {
        return BASE_API_URL + "/" + model + "/" + image_path;
    },
};

class ChatApp {
    constructor() {
        this.chatHistory = document.getElementById('chatHistory');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.fileInput = document.getElementById('fileInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.modelSelect = document.getElementById('modelSelect');

        this.currentChatId = null;
        this.selectedImage = null;
        this.isTyping = false;

        this.setupEventListeners();
        this.adjustTextAreaHeight();
    }

    setupEventListeners() {
        this.messageInput.addEventListener('input', () => {
            this.adjustTextAreaHeight();
            this.updateSendButton();
        });

        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.handleSendMessage();
            }
        });

        this.sendButton.addEventListener('click', () => this.handleSendMessage());
        this.fileInput.addEventListener('change', (e) => this.handleImageChange(e));
    }

    adjustTextAreaHeight() {
        this.messageInput.style.height = 'auto';
        this.messageInput.style.height = `${this.messageInput.scrollHeight}px`;
    }

    updateSendButton() {
        this.sendButton.disabled = !this.messageInput.value.trim() && !this.selectedImage;
    }

    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message assistant';
        typingDiv.innerHTML = `
            <div class="message-icon">
                <i class="fas fa-robot" style="color: #FFD700;"></i>
            </div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
        typingDiv.id = 'typingIndicator';
        this.chatHistory.appendChild(typingDiv);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
        this.isTyping = true;
    }

    hideTypingIndicator() {
        const typingIndicator = document.getElementById('typingIndicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
        this.isTyping = false;
    }

    async handleSendMessage() {
        const message = this.messageInput.value.trim();
        if (!message && !this.selectedImage) return;

        const messageData = {
            content: message,
            image: this.selectedImage?.preview || null,
            timestamp: new Date().toISOString()
        };

        // Add user message to chat
        this.appendMessage('user', messageData);

        // Clear input
        this.messageInput.value = '';
        this.adjustTextAreaHeight();
        this.selectedImage = null;
        this.imagePreview.style.display = 'none';
        this.updateSendButton();

        // Show typing indicator
        this.showTypingIndicator();

        try {
            let response;
            if (this.currentChatId) {
                // Continue existing chat
                response = await DataService.ContinueChatWithLLM(
                    this.modelSelect.value,
                    this.currentChatId,
                    messageData
                );
            } else {
                // Start new chat
                response = await DataService.StartChatWithLLM(
                    this.modelSelect.value,
                    messageData
                );
                this.currentChatId = response.chat_id;
            }

            // Hide typing indicator
            this.hideTypingIndicator();

            // Add assistant response to chat
            this.appendMessage('assistant', {
                content: response.messages[response.messages.length - 1].content,
                timestamp: new Date(response.dts * 1000).toISOString()
            });
        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            this.appendMessage('assistant', {
                content: 'Sorry, there was an error processing your message.',
                timestamp: new Date().toISOString()
            });
        }
    }

    handleImageChange(event) {
        const file = event.target.files?.[0];
        if (file) {
            if (file.size > 5000000) { // 5MB limit
                alert('File size should be less than 5MB');
                return;
            }

            const reader = new FileReader();
            reader.onloadend = () => {
                this.selectedImage = {
                    file: file,
                    preview: reader.result
                };
                this.previewImg.src = reader.result;
                this.imagePreview.style.display = 'inline-block';
                this.updateSendButton();
            };
            reader.readAsDataURL(file);
        }
    }

    removeImage() {
        this.selectedImage = null;
        this.imagePreview.style.display = 'none';
        this.fileInput.value = '';
        this.updateSendButton();
    }

    formatTime(timestamp) {
        return new Date(timestamp).toLocaleTimeString([], {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    appendMessage(role, messageData) {
        console.log(role, messageData);

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        // Icon
        const iconDiv = document.createElement('div');
        iconDiv.className = 'message-icon';
        const icon = document.createElement('i');
        icon.className = role === 'user' ? 'fas fa-user' : 'fas fa-robot';
        icon.style.color = role === 'user' ? '#FFFFFF' : '#FFD700';
        iconDiv.appendChild(icon);

        // Content
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        // Add image if present
        if (messageData.image) {
            const img = document.createElement('img');
            img.src = messageData.image;
            img.style.maxWidth = '200px';
            img.style.maxHeight = '200px';
            img.style.borderRadius = '8px';
            img.style.marginBottom = '8px';
            contentDiv.appendChild(img);
        }

        // Add text content
        if (messageData.content) {
            const textContent = document.createElement('div');
            textContent.innerHTML = marked.parse(messageData.content);
            contentDiv.appendChild(textContent);
        }

        // Add timestamp if present
        if (messageData.timestamp) {
            const timeSpan = document.createElement('span');
            timeSpan.className = 'message-time';
            timeSpan.textContent = this.formatTime(messageData.timestamp);
            messageDiv.appendChild(timeSpan);
        }

        // Assemble message
        messageDiv.appendChild(iconDiv);
        messageDiv.appendChild(contentDiv);

        // Add to chat history
        this.chatHistory.appendChild(messageDiv);
        this.chatHistory.scrollTop = this.chatHistory.scrollHeight;
    }
}

// Initialize the chat app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.chatApp = new ChatApp();
});