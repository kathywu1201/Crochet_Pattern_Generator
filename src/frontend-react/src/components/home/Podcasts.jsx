
'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import styles from './Podcasts.module.css';
import PodcastCard from '../shared/PodcastCard';
//import DataService from "../../services/MockDataService"; // Mock
import DataService from "../../services/DataService";


export default function Podcasts() {
    // Component States
    const [episodes, setEpisodes] = useState([]);

    // Setup Component
    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await DataService.GetPodcasts(4); // Limiting to 4 episodes for the main view
                setEpisodes(response.data);
            } catch (error) {
                console.error('Error fetching podcasts:', error);
                setEpisodes([]); // Set empty array in case of error
            }
        };

        fetchData();
    }, []);

    return (
        <section className={styles.section} id="podcasts">
            <h2 className={styles.title}>Podcast</h2>
            <div className={styles.underline}></div>

            <div className={styles.content}>
                <div className={styles.aboutPodcast}>
                    <Image
                        src="/assets/podcast.png"
                        alt="Podcast Icon"
                        width={240}
                        height={240}
                        style={{
                            width: 'auto',
                            height: 'auto',
                        }}
                    />
                    <h3>About Podcasts</h3>
                    <p>
                        Welcome to The Cheese Podcast, where we celebrate cheeses from around
                        the world in multiple languages! Each episode dives into the flavors,
                        textures, and stories behind different cheeses, bringing together
                        cultures and cuisines. Whether you're a cheese connoisseur or just curious, join us as we explore the world of cheese, one language at a time!
                    </p>
                </div>

                <div className={styles.episodesList}>
                    {episodes.map((episode) => (
                        <PodcastCard key={episode.id} podcast={episode}></PodcastCard>
                    ))}
                </div>
            </div>
            <div className={styles.viewAllContainer}>
                <Link href="/podcasts" className={styles.viewAllButton}>
                    View All Podcasts
                </Link>
            </div>
        </section>
    )
}