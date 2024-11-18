// main.js

// DOM Elements
const header = document.querySelector('.header');
const hamburger = document.querySelector('.hamburger');
const mobileMenu = document.querySelector('.mobile-menu');
const podcastGrid = document.querySelector('.podcast-grid');

// Mobile Menu Toggle
hamburger.addEventListener('click', () => {
    mobileMenu.classList.toggle('active');
    hamburger.classList.toggle('active');
});

// Sticky Header
let lastScroll = 0;
window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll <= 0) {
        header.classList.remove('scroll-up');
        return;
    }

    if (currentScroll > lastScroll && !header.classList.contains('scroll-down')) {
        // Scroll Down
        header.classList.remove('scroll-up');
        header.classList.add('scroll-down');
    } else if (currentScroll < lastScroll && header.classList.contains('scroll-down')) {
        // Scroll Up
        header.classList.remove('scroll-down');
        header.classList.add('scroll-up');
    }
    lastScroll = currentScroll;
});

// Sample Podcast Data (This would normally come from an API)
const podcastData = [
    {
        title: "Episode 1 (Halloumi) [FR]",
        description: "Discover the wonderful world of Halloumi cheese",
        date: "September 12, 2024",
        duration: "5:36"
    },
    {
        title: "Episode 1 (Halloumi) [ES]",
        description: "Discover the wonderful world of Halloumi cheese",
        date: "September 12, 2024",
        duration: "5:59"
    },
    // Add more episodes as needed
];

// Create Podcast Cards
function createPodcastCard(podcast) {
    return `
        <div class="podcast-card">
            <div class="podcast-content">
                <h3>${podcast.title}</h3>
                <p>${podcast.description}</p>
                <div class="podcast-meta">
                    <span>${podcast.date}</span>
                    <span>${podcast.duration}</span>
                </div>
            </div>
            <div class="podcast-controls">
                <button class="play-button" aria-label="Play">▶</button>
            </div>
        </div>
    `;
}

// Render Podcast Cards
function renderPodcasts() {
    podcastGrid.innerHTML = podcastData.map(podcast =>
        createPodcastCard(podcast)
    ).join('');
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Header scroll effect
    const header = document.querySelector('.header');

    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    });

    // Mobile menu functionality
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');

    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        hamburger.classList.toggle('active');
    });

    renderPodcasts();
});

// Smooth Scroll for Navigation Links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth'
            });
            // Close mobile menu if open
            mobileMenu.classList.remove('active');
            hamburger.classList.remove('active');
        }
    });
});