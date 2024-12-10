'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { ArrowForward } from '@mui/icons-material';
import HistoryIcon from '@mui/icons-material/History';
//import DataService from "../../services/MockDataService"; // Mock
import DataService from "../../services/DataService";
import { formatRelativeTime, MOCK_SERVICE } from "../../services/Common";


// Styles
import styles from './Community.module.css';

export default function Card(){
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
                <div  className={styles.chatCard}>
                    <h3 className={styles.chatTitle}>aaa</h3>
                    <span className={styles.chatTime}>
                        aafew
                    </span>
                </div>
            </div>
        </div>
    )
}