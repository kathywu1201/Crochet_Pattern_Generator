'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
//import DataService from "../../services/MockDataService"; // Mock
import DataService from "../../services/DataService";
import { formatRelativeTime } from "../../services/Common";

// Styles
import styles from './ChatHistorySidebar.module.css';

export default function ChatHistorySidebar({
    chat_id,
    model
}) {
    // Component States
    const [chatHistory, setChatHistory] = useState([]);
    const router = useRouter();

    // Setup Component
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await DataService.GetChats(model, 20);
                setChatHistory(response.data);
            } catch (error) {
                console.error('Error fetching podcasts:', error);
                setChatHistory([]); // Set empty array in case of error
            }
        };

        fetchData();
    }, []);

    return (
        <div className={styles.sidebar}>
            <div className={styles.sidebarHeader}>
                <h2>Chat History</h2>
                <button
                    className={styles.newChatButton}
                    onClick={() => router.push('/chat?model=' + model)}
                >
                    New Chat
                </button>
            </div>
            <div className={styles.chatList}>
                {chatHistory.map((item) => (
                    <div
                        key={item.chat_id}
                        className={`${styles.chatItem} ${chat_id === item.id ? styles.active : ''}`}
                        onClick={() => router.push('/chat?model=' + model + '&id=' + item.chat_id)}
                    >
                        <div className={styles.chatItemContent}>
                            <span className={styles.chatTitle}>
                                {item.title}
                            </span>
                            <span className={styles.chatDate}>
                                {formatRelativeTime(item.dts)}
                            </span>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    )
}