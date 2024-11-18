'use client';

import { useState, use, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import IconButton from '@mui/material/IconButton';
import ChatInput from '@/components/chat/ChatInput';
import ChatHistory from '@/components/chat/ChatHistory';
import ChatHistorySidebar from '@/components/chat/ChatHistorySidebar';
import ChatMessage from '@/components/chat/ChatMessage';
//import DataService from "../../services/MockDataService"; // Mock
import DataService from "../../services/DataService";
import { uuid } from "../../services/Common";

// Import the styles
import styles from "./styles.module.css";

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
    const newChat = (message) => {
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
                setIsTyping(false);
                setChat(null);
                setChatId(null);
                setHasActiveChat(false);
                router.push('/chat?model=' + selectedModel)
            }
        };
        startChat(message);

    };
    const appendChat = (message) => {
        console.log(message);
        // Append message and submit to LLM

        const continueChat = async (id, message) => {
            try {
                // Show typing indicator
                setIsTyping(true);
                setHasActiveChat(true);
                tempChatMessage(message);

                // Submit chat
                const response = await DataService.ContinueChatWithLLM(model, id, message);
                console.log(response.data);

                // Hide typing indicator and add response
                setIsTyping(false);

                setChat(response.data);
                forceRefresh();
            } catch (error) {
                console.error('Error fetching chat:', error);
                setIsTyping(false);
                setChat(null);
                setHasActiveChat(false);
            }
        };
        continueChat(chat_id, message);
    };
    // Force re-render by updating the key
    const forceRefresh = () => {
        setRefreshKey(prevKey => prevKey + 1);
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

            {/* Hero Section */}
            {!hasActiveChat && (
                <section className={styles.hero}>
                    <div className={styles.heroContent}>
                        <h1>Cheese Assistant 🌟</h1>
                        {/* Main Chat Input: ChatInput */}
                        <ChatInput onSendMessage={newChat} className={styles.heroChatInputContainer} selectedModel={selectedModel} onModelChange={handleModelChange}></ChatInput>
                    </div>
                </section>
            )}

            {/* Chat History Section: ChatHistory */}
            {!hasActiveChat && (
                <ChatHistory model={model}></ChatHistory>
            )}

            {/* Chat Block Header Section */}
            {hasActiveChat && (
                <div className={styles.chatHeader}></div>
            )}
            {/* Active Chat Interface */}
            {hasActiveChat && (
                <div className={styles.chatInterface}>
                    {/* Chat History Sidebar: ChatHistorySidebar */}
                    <ChatHistorySidebar chat_id={chat_id} model={model}></ChatHistorySidebar>

                    {/* Main chat area */}
                    <div className={styles.mainContent}>
                        {/* Chat message: ChatMessage */}
                        <ChatMessage chat={chat} key={refreshKey} isTyping={isTyping} model={model}></ChatMessage>
                        {/* Sticky chat input area: ChatInput */}
                        <ChatInput
                            onSendMessage={appendChat}
                            chat={chat}
                            selectedModel={selectedModel}
                            onModelChange={setSelectedModel}
                            disableModelSelect={true}
                        ></ChatInput>
                    </div>
                </div>
            )}
        </div>
    );
}