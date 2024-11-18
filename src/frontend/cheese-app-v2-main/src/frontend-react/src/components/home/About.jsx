import Link from 'next/link';
import styles from './About.module.css';

export default function About() {
    return (
        <section className={styles.about} id="about">
            <h1 className={styles.title}>About Us</h1>
            <div className={styles.underline}></div>
            <div className={styles.content}>
                <p>
                    Welcome to <strong>Formaggio.me</strong>, a web application born out of a passion for both cheese and cutting-edge
                    technology. This site was created as part of a demonstration project for developing applications using large
                    language models (AI). My name is Pavlos Protopapas, and I am the instructor of <strong>AC215</strong>, a course offered at
                    <strong> Harvard University</strong>. You can find more details about the course <Link href="http://harvard-iacs.github.io/2024-AC215/">here</Link> and learn more about me and my
                    research <Link href="https://www.stellardnn.org/">here</Link>.
                </p>

                <p>
                    If you're interested in taking the course as a Harvard student you can find it in the my.harvard
                    catalog, it is also available through <strong>Harvard's Division of Continuing Education (DCE)</strong> for everyone else,
                    with the next offering scheduled for <strong>Spring 2025</strong>.
                </p>

                <p>
                    The course is designed to provide structured experiential learning, where I, the instructor, build the <strong>AI
                        Cheese Web App</strong> step by step during the semester. This hands-on approach helps students understand
                    both the technical and creative aspects of AI application development.
                </p>

                <p>
                    Please note that this is a demonstration project, so some features may be incomplete or still under
                    development. However, we hope you enjoy exploring it and would love to hear your thoughts! Feel free to
                    send us an email with comments.
                </p>

                <p>
                    Thank you for visiting <strong>Formaggio.me</strong>, and we hope you have fun exploring the intersection of cheese and
                    AI!
                </p>

                <Link href="mailto:pavlos@seas.harvard.edu?subject=Feedback%20from%20Formaggio.me" className={styles.contactButton}>
                    CONTACT US
                </Link>
            </div>
        </section>
    );
}