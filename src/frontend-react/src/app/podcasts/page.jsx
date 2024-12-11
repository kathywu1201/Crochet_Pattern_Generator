'use client';

import React from 'react';
import styles from './styles.module.css';

const HomePage = () => {
    return (
        <div className={styles.wrapper}>
            <div className={styles.container}>
                <div className={styles.card}>
                    <img src="/assets/guess_you_like.jpg" alt="Guess You Like" className={styles.cardImage} />
                    <div className={styles.cardTitle}>Guess You Like</div>
                </div>
                <div className={styles.card}>
                    <img src="/assets/community.jpg" alt="Community Spotlight" className={styles.cardImage} />
                    <div className={styles.cardTitle}>Community Spotlight</div>
                </div>
                <div className={styles.card}>
                    <img src="/assets/library.jpg" alt="Your Library" className={styles.cardImage} />
                    <div className={styles.cardTitle}>Your Library</div>
                </div>
            </div>
        </div>
    );
};

export default HomePage;