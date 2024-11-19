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
        <section className={styles.about} id="about">
            <h1 className={styles.title}>About Us</h1>
            <div className={styles.underline}></div>
            <div className={styles.content}>
                <h2 className={styles.subtitle}>Our Mission</h2>
                <p>
                    Yarn Master is a web application born from a shared passion for the timeless art of handmade crochet and knit creations, 
                    blended with the possibilities of cutting-edge technology. Created by participants of a prestigious Harvard course and avid crochet enthusiasts,
                    this platform bridges tradition and innovation to empower crochet and knit enthusiasts around the world.
                    Our mission is to make crafting easier, faster, and more enjoyable. 
                    With the power of AI, we aim to help you access patterns and related information effortlessly while fostering a vibrant community where creativity thrives.
                </p>

                <h2 className={styles.subtitle}>What We Offer</h2>
                <p>
                    Yarn Master offers a range of features tailored to meet the needs of every crochet and knit lover:
                </p>
                <ul className={styles.featuresList}>
                    <li>
                        <strong>Custom Pattern Generation:</strong> Upload a photo of your crochet or knit creation and receive a personalized, step-by-step tutorial. 
                        This includes a detailed guide to creating your masterpiece, a list of required materials, and links to recommended purchase options.
                    </li>
                    <li>
                        <strong>Join the Community:</strong> Connect with like-minded crafters in our vibrant community space. 
                        Share your ideas, showcase your latest projects, ask questions, and learn from others' experiences.
                    </li>
                </ul>
                <p>
                    Whether you are a beginner taking your first steps, a hobbyist exploring your passion, or a seasoned professional, 
                    Yarn Master is designed to support and inspire everyone on their crafting journey.
                </p>

                <h2 className={styles.subtitle}>Why Choose Us?</h2>
                <ul className={styles.benefitsList}>
                    <li>
                        <strong>AI-Powered Innovation:</strong> Save time and effort with our state-of-the-art AI technology that transforms images into detailed pattern instructions in moments.
                    </li>
                    <li>
                        <strong>Inclusive Community:</strong> Be part of a supportive and inspiring group of creators who share your love for crochet and knitting.
                    </li>
                    <li>
                        <strong>Comprehensive Resources:</strong> Access curated material recommendations and purchasing links to get everything you need to bring your projects to life.
                    </li>
                </ul>

                <h2 className={styles.subtitle}>Contact Us</h2>
                <p>
                    We’d love to hear from you! If you have questions, suggestions, or simply want to connect, please reach out to us via [contact information here]. 
                    Together, let’s create something beautiful with Yarn Master!
                </p>
            </div>
        </section>
    );
}
