'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import styles from './Newsletters.module.css';
//import DataService from "../../services/MockDataService"; // Mock
import DataService from "../../services/DataService";


export default function Newsletter() {
    // Component States
    const [newsletters, setNewsletters] = useState([]);

    // Setup Component
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await DataService.GetNewsletters(4); // Limiting to 4 episodes for the main view
                setNewsletters(response.data);
            } catch (error) {
                console.error('Error fetching podcasts:', error);
                setNewsletters([]); // Set empty array in case of error
            }
        };

        fetchData();
    }, []);

    return (
        <section className={styles.section} id="newsletters">
            <h2 className={styles.title}>Newsletters</h2>
            <div className={styles.underline}></div>

            <div className={styles.content}>
                <div className={styles.newsletterGrid}>
                    {newsletters.map((newsletter) => (
                        <article key={newsletter.id} className={styles.newsletterCard}>
                            <div className={styles.cardHeader}>
                                <span className={styles.date}>{newsletter.date}</span>
                                <span className={styles.readTime}>{newsletter.readTime}</span>
                            </div>

                            <h3 className={styles.newsletterTitle}>{newsletter.title}</h3>

                            <p className={styles.excerpt}>{newsletter.excerpt}</p>

                            <Link href={`/newsletters?id=${newsletter.id}`} className={styles.readMore}>
                                Read More →
                            </Link>
                        </article>
                    ))}
                </div>
                <div className={styles.aboutNewsletter}>
                    <Image
                        src="/assets/newsletter.png"
                        alt="Newsletter Icon"
                        width={240}
                        height={240}
                        style={{
                            width: 'auto',
                            height: 'auto',
                        }}
                    />
                    <h3>About Newsletters</h3>
                    <p>
                        Welcome to Formaggio.me's Cheese Chronicles, your weekly digest of all things cheese! Our newsletters dive deep into the fascinating world of artisanal cheese-making, featuring expert insights, tasting notes, and the latest innovations in cheese technology. From traditional techniques to AI-powered cheese analysis, we explore the intersection of time-honored craftsmanship and modern innovation. Whether you're a cheese professional, enthusiast, or just beginning your cheese journey, our newsletters provide valuable insights, pairing suggestions, and behind-the-scenes looks at the world's finest cheeses. Stay informed, inspired, and connected to the global cheese community with our weekly updates!
                    </p>
                </div>
            </div>
            <div className={styles.viewAllContainer}>
                <Link href="/newsletters" className={styles.viewAllButton}>
                    View All Newsletters
                </Link>
            </div>
        </section>
    );
}