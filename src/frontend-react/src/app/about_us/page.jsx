'use client';

import { useState, useEffect, use } from 'react';
import Image from 'next/image';
import Link from 'next/link';
//import DataService from "../../services/MockDataService"; // Mock
import DataService from "../../services/DataService";

// Import the styles
import styles from "./styles.module.css";


// export default function About() {
//     return (
//         <section className={styles.about} id="about">
//             <h1 className={styles.title}>About Us</h1>
//             <div className={styles.underline}></div>
//             <div className={styles.content}>
//                 <p>
//                 Yarn Master is a web application born from a shared passion for the timeless art of handmade crochet and knit creations, 
//                 blended with the possibilities of cutting-edge technology. Created by participants of a prestigious Harvard course and avid crochet enthusiasts,
//                 this platform bridges tradition and innovation to empower crochet and knit enthusiasts around the world.

//                 Our mission is to make crafting easier, faster, and more enjoyable. 
//                 With the power of AI, we aim to help you access patterns and related information effortlessly while fostering a vibrant community where creativity thrives.
//                 </p>

//                 <p>
//                 Yarn Master offers a range of features tailored to meet the needs of every crochet and knit lover:

//                 Custom Pattern Generation: Upload a photo of your crochet or knit creation and receive a personalized, step-by-step tutorial. This includes a detailed guide to creating your masterpiece, a list of required materials, and links to recommended purchase options.

//                 Join the Community: Connect with like-minded crafters in our vibrant community space. Share your ideas, showcase your latest projects, ask questions, and learn from others' experiences.

//                 Whether you are a beginner taking your first steps, a hobbyist exploring your passion, or a seasoned professional, Yarn Master is designed to support and inspire everyone on their crafting journey.


//                 </p>

//                 <p>
//                 AI-Powered Innovation: Save time and effort with our state-of-the-art AI technology that transforms images into detailed pattern instructions in moments.
//                 Inclusive Community: Be part of a supportive and inspiring group of creators who share your love for crochet and knitting.
//                 Comprehensive Resources: Access curated material recommendations and purchasing links to get everything you need to bring your projects to life.
//                 At Yarn Master, we are committed to revolutionizing how you approach crochet and knitting, making it more accessible, enjoyable, and inspiring for everyone.
//                 </p>

//                 <p>
//                 We’d love to hear from you! If you have questions, suggestions, or simply want to connect, please reach out to us via [contact information here]. Together, let’s create something beautiful with Yarn Master!
//                 </p>


//                 <Link href="mailto:pavlos@seas.harvard.edu?subject=Feedback%20from%20Formaggio.me" className={styles.contactButton}>
//                     CONTACT US
//                 </Link>
//             </div>
//         </section>
//     );
// }

export default function About() {
    return (
        <section
            className={`${styles.about} pt-16 pb-8`}
            id="about"
            style={{
                background: 'linear-gradient(45deg, rgba(251, 219, 20, 0.4), rgba(1, 249, 198, 0.4))'
            }}
        >
            {/* Inner Container for Centered Content */}
            <div className="max-w-screen-lg mx-auto px-8 lg:px-24">
                {/* Centered Title with Top Margin */}
                <h1 className={`${styles.title} text-center mt-8 mb-8`}>About Us</h1>
                <div className={`${styles.underline} mx-auto mb-8`}></div>
                <div className={styles.content}>
                    {/* Left-Aligned Subtitles */}
                    <h2 className={`${styles.subtitle} mb-4`}>Our Mission</h2>
                    <p className="mb-8">
                        Yarn Master is a web application born from a shared passion for the timeless art of handmade crochet and knit creations, 
                        blended with the possibilities of cutting-edge technology. Created by participants of a prestigious Harvard course and avid crochet enthusiasts,
                        this platform bridges tradition and innovation to empower crochet and knit enthusiasts around the world.
                        Our mission is to make crafting easier, faster, and more enjoyable. 
                        With the power of AI, we aim to help you access patterns and related information effortlessly while fostering a vibrant community where creativity thrives.
                    </p>

                    <h2 className={`${styles.subtitle} mb-4`}>What We Offer</h2>
                    <p className="mb-8">
                        Yarn Master offers a range of features tailored to meet the needs of every crochet and knit lover:
                    </p>
                    <p className="mb-4">
                        <strong>Pattern Generation:</strong> Upload a photo of your crochet or knit creation and receive a personalized, step-by-step tutorial.
                    </p>
                    <p className="mb-8">
                        <strong>Community:</strong> Connect with like-minded crafters in our vibrant community space. [Comming Soon]
                    </p>
                    <p className="mb-8">
                        <strong>Guess You Like:</strong> Whether you are a beginner taking your first steps, a hobbyist exploring your passion, or a seasoned professional, Yarn Master is designed to support and inspire everyone on their crafting journey. [Comming Soon]
                    </p>
                    <p className="mb-8">
                        <strong>Resources:</strong> Access curated material recommendations and purchasing links to get everything you need to bring your projects to life. [Comming Soon]
                    </p>
                    <p className="mb-8">
                        Any other new feature you want to add here, we'd love to hear your suggestions!
                    </p>

                    <h2 className={`${styles.subtitle} mb-4`}>Features</h2>
                    <p className="mb-4">
                        <strong>Custom Pattern Generation:</strong> Upload a photo of your crochet or knit creation and receive a personalized, step-by-step tutorial. 
                        This includes a detailed guide to creating your masterpiece, a list of required materials, and links to recommended purchase options.
                    </p>
                    <p className="mb-8">
                        <strong>Join the Community:</strong> Connect with like-minded crafters in our vibrant community space. 
                        Share your ideas, showcase your latest projects, ask questions, and learn from others' experiences.
                    </p>

                    <h2 className={`${styles.subtitle} mb-4`}>Why Choose Us?</h2>
                    <p className="mb-4">
                        <strong>AI-Powered Innovation:</strong> Save time and effort with our state-of-the-art AI technology that transforms images into detailed pattern instructions in moments.
                    </p>
                    <p className="mb-4">
                        <strong>Inclusive Community:</strong> Be part of a supportive and inspiring group of creators who share your love for crochet and knitting.
                    </p>
                    <p className="mb-8">
                        <strong>Comprehensive Resources:</strong> Access curated material recommendations and purchasing links to get everything you need to bring your projects to life.
                    </p>

                    <h2 className={`${styles.subtitle} mb-4`}>Contact Us</h2>
                    <p className="mb-8">
                        We’d love to hear from you! If you have questions, suggestions, or simply want to connect, please reach out to us via 
                        <a href="mailto:yarnmaster2024@gmail.com" style={{ color: 'blue' }}> yarnmaster2024@gmail.com</a>. 
                        Together, let’s create something beautiful with Yarn Master!
                    </p>
                </div>
                {/* Centered Image */}
                <div className="image-container text-center mt-8">
                    <img
                        src="/assets/birds.jpg"
                        alt="Crochet Artwork"
                        className="about-image"
                    />
                </div>
            </div>
        </section>
    );
}


