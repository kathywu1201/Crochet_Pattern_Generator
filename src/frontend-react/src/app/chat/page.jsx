'use client';

import { useState, use, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import ChatInput from '@/components/chat/ChatInput';
import ChatHistory from '@/components/chat/ChatHistory'; // Import ChatHistory
import ChatHistorySidebar from '@/components/chat/ChatHistorySidebar';
import ChatMessage from '@/components/chat/ChatMessage';
import DataService from "../../services/DataService";
import { uuid } from "../../services/Common";

// Styles
import styles from './styles.module.css';

export default function ChatPage({ searchParams }) {
    const params = use(searchParams);
    const chat_id = params.id;
    const model = params.model || 'llm';
    console.log(chat_id, model);

    // Component States
    const [chatId, setChatId] = useState(params.id);
    const [hasActiveChat, setHasActiveChat] = useState(false);
    const [chat, setChat] = useState(null);
    const [refreshKey, setRefreshKey] = useState(0);
    const [isTyping, setIsTyping] = useState(false);
    const [selectedModel, setSelectedModel] = useState(model);
    const router = useRouter();

    // Fetch chat details when there's a chat ID
    const fetchChat = async (id) => {
        try {
            setChat(null);
            const response = await DataService.GetChat(model, id);
            setChat(response.data);
            console.log(chat);
        } catch (error) {
            console.error('Error fetching chat:', error);
            setChat(null);
        }
    };

    // Setup Component
    useEffect(() => {
        if (chat_id) {
            fetchChat(chat_id);
            setHasActiveChat(true);
        } else {
            setChat(null);
            setHasActiveChat(false);
        }
    }, [chat_id]);
    useEffect(() => {
        setSelectedModel(model);
    }, [model]);

    function tempChatMessage(message) {
        // Set temp values
        message["message_id"] = uuid();
        message["role"] = 'user';
        if (chat) {
            // Append message
            var temp_chat = { ...chat };
            temp_chat["messages"].push(message);
        } else {
            var temp_chat = {
                "messages": [message]
            }
            return temp_chat;
        }
    }

    // Handlers
    const startNewChat = (message) => {
        console.log(message);
        // Start a new chat and submit to LLM
        const startChat = async (message) => {
            try {
                // Show typing indicator
                setIsTyping(true);
                setHasActiveChat(true);
                setChat(tempChatMessage(message)); // Show the user input message while LLM is invoked

                // Submit chat
                const response = await DataService.StartChatWithLLM(model, message);
                console.log(response.data);

                // Hide typing indicator and add response
                setIsTyping(false);

                setChat(response.data);
                setChatId(response.data["chat_id"]);
                router.push('/chat?model=' + selectedModel + '&id=' + response.data["chat_id"]);
            } catch (error) {
                console.error('Error fetching chat:', error);
                setError(error);
                setIsTyping(false);
                setChat(null);
                setChatId(null);
                setHasActiveChat(false);
                router.push('/chat?model=' + selectedModel);
            }
        };
        startChat(message);

    };

    const handleModelChange = (newValue) => {

        setSelectedModel(newValue);
        var path = '/chat?model=' + newValue;
        if (chat_id) {
            path = path + '&id=' + chat_id;
        }
        router.push(path)
    };

    return (
        <div className={styles.container}>
            {!hasActiveChat ? (
                /* Input Page */
                <section className={styles.hero}>
                    <div className={styles.heroContent}>
                        <h1>Pattern Assistant 🌟</h1>
                        <ChatInput
                            onSendMessage={startNewChat} // Use startNewChat to handle submissions
                            selectedModel={model}
                            onModelChange={handleModelChange}
                            disableModelSelect={false}
                        />
                    </div>
                </section>
            ) : (
                /* Chat Page */
                <div className={styles.chatInterface}>
                    {/* Chat History Sidebar: Only in the active chat interface */}
                    <ChatHistorySidebar chat_id={chat_id} model={model}></ChatHistorySidebar>
                    <div className={styles.mainContent}>
                        <ChatMessage chat={chat} model={model} />
                        <ChatInput disableModelSelect={true} /> {/* Only show "New Pattern" button */}
                    </div>
                </div>
            )}
        </div>
    );
    
}