'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { PlayCircle, PauseCircle } from '@mui/icons-material';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import DataService from "../../services/DataService";
import styles from './PodcastCard.module.css';

export default function PodcastCard({ podcast }) {
    // Component States
    const [currentlyPlaying, setCurrentlyPlaying] = useState(null);
    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const audioRef = useRef(new Audio());

    // Setup Component
    useEffect(() => {
        // Cleanup function
        return () => {
            audioRef.current.pause();
            audioRef.current.src = '';
        };
    }, []);

    // Handle audio time updates
    useEffect(() => {
        const audio = audioRef.current;

        const handleTimeUpdate = () => {
            setCurrentTime(audio.currentTime);
        };

        const handleEnded = () => {
            setIsPlaying(false);
            setCurrentTime(0);
        };

        audio.addEventListener('timeupdate', handleTimeUpdate);
        audio.addEventListener('ended', handleEnded);

        return () => {
            audio.removeEventListener('timeupdate', handleTimeUpdate);
            audio.removeEventListener('ended', handleEnded);
        };
    }, []);

    const formatTime = (timeInSeconds) => {
        const hours = Math.floor(timeInSeconds / 3600);
        const minutes = Math.floor((timeInSeconds % 3600) / 60);
        const seconds = Math.floor(timeInSeconds % 60);
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
    };

    const parseDuration = (durationString) => {
        const [hours, minutes, seconds] = durationString.split(':').map(Number);
        return hours * 3600 + minutes * 60 + seconds;
    };

    const togglePlayPause = async (episodeId) => {
        console.log(episodeId);
        const audio = audioRef.current;

        if (currentlyPlaying === episodeId) {
            if (isPlaying) {
                audio.pause();
                setIsPlaying(false);
            } else {
                await audio.play();
                setIsPlaying(true);
            }
        } else {
            if (currentlyPlaying) {
                audio.pause();
            }
            audio.src = DataService.GetPodcastAudio(episodeId + "-EN.mp3");
            setCurrentlyPlaying(episodeId);
            setCurrentTime(0);
            try {
                await audio.play();
                setIsPlaying(true);
            } catch (error) {
                console.error('Error playing audio:', error);
            }
        }
    };

    return (
        <div className={styles.episodeCard}>
            <div className={styles.episodeHeader}>
                <div className={styles.episodeInfo}>
                    <span className={styles.podcast}>FORMAGGIO</span>
                    <h4 className={styles.episodeTitle}>{podcast.title}</h4>
                    <span className={styles.date}>{podcast.date}</span>
                </div>
                <div className={styles.controls}>
                    <button
                        className={styles.playButton}
                        onClick={() => togglePlayPause(podcast.id)}
                    >
                        {isPlaying && currentlyPlaying === podcast.id ?
                            <PauseCircle /> :
                            <PlayCircle />
                        }
                    </button>
                </div>
            </div>

            <div className={styles.progressContainer}>
                <div className={styles.timeStamp}>
                    {currentlyPlaying === podcast.id ?
                        formatTime(currentTime) :
                        "00:00:00"
                    }
                </div>
                <div className={styles.progressBar}>
                    <div
                        className={styles.progress}
                        style={{
                            width: currentlyPlaying === podcast.id ?
                                `${(currentTime / parseDuration(podcast.duration)) * 100}%` :
                                '0%'
                        }}
                    ></div>
                </div>
                <div className={styles.timeStamp}>{podcast.duration}</div>
            </div>

            <div className={styles.episodeFooter}>
                <button className={styles.descriptionToggle}>
                    Show description ▼
                </button>
                <VolumeUpIcon className={styles.volumeIcon} />
            </div>
        </div>
    )
}