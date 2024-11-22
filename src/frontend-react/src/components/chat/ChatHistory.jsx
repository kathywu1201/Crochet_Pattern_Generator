'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowForward } from '@mui/icons-material';
import HistoryIcon from '@mui/icons-material/History';
//import DataService from "../../services/MockDataService"; // Mock
import DataService from "../../services/DataService";
import { formatRelativeTime, MOCK_SERVICE } from "../../services/Common";


// Styles
import styles from './ChatHistory.module.css';

export default function ChatHistory({
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
    }, [model]);

    return (
        <div className={styles.recentChats}>
            <div className={styles.recentHeader}>
                <h2>
                    <span className={styles.chatIcon}><HistoryIcon></HistoryIcon></span>
                    Your recent chats
                </h2>
                <button className={styles.viewAllButton}>
                    View all <ArrowForward />
                </button>
            </div>

            <div className={styles.chatGrid}>
                {chatHistory.map((item) => (
                    <div key={item.chat_id} className={styles.chatCard} onClick={() => router.push('/chat?model=' + model + '&id=' + item.chat_id)}>
                        <h3 className={styles.chatTitle}>{item.title}</h3>
                        <span className={styles.chatTime}>
                            {formatRelativeTime(item.dts)}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    )
}