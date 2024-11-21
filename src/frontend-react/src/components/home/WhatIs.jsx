import Image from 'next/image';
import styles from './WhatIs.module.css';

export default function WhatIs() {
    return (
        <section className={styles.section}>
            <h2 className={styles.title}>Ready for formaggio.me!</h2>
            <div className={styles.underline}></div>

            <div className={styles.content}>
                <div className={styles.textContent}>
                    <h3 className={styles.subtitle}>Discover the world of cheese with formaggio.me!</h3>

                    <p>
                        Imagine being able to identify a cheese by simply taking a photo of it. Our app uses <strong>AI-powered</strong> visual
                        recognition technology to help you identify the cheese you're looking at, and then provides you with a
                        wealth of information about it.
                    </p>

                    <p>
                        Take a photo of the cheese, and our app will identify it for you. Then, dive deeper into the world of cheese
                        with our interactive chatbot. Ask questions about the cheese's origin, production process, nutritional
                        information, and history.
                    </p>

                    <p>
                        Want to host a cheese-tasting party? Formaggio.me makes it easy. Use our app to select the perfect cheeses
                        for your gathering, and then get expert advice on pairing them with wines, crackers, and other
                        accompaniments. Our chatbot is always available to help you plan the perfect cheese platter.
                    </p>

                    <p>
                        Formaggio.me is your one-stop-shop for all things cheese. With our app, you'll never be stuck wondering
                        what that delicious cheese is or how to pair it with other foods. Whether you're a cheese aficionado or just
                        starting to explore the world of cheese, Formaggio.me is the perfect companion for your culinary journey.
                    </p>

                    <div className={styles.features}>
                        <h4>Key Features:</h4>
                        <ul>
                            <li>Visual cheese identification using AI-powered technology</li>
                            <li>Interactive chatbot for asking questions about cheese</li>
                            <li>In-depth information on cheese origin, production process, nutritional information, and history</li>
                            <li>Expert advice on pairing cheese with wines, crackers, and other accompaniments</li>
                            <li>Perfect for cheese enthusiasts, party planners, and anyone looking to explore the world of cheese</li>
                        </ul>
                    </div>
                </div>

                <div className={styles.imageContainer}>
                    <Image
                        src="/assets/cheese-platter.png"
                        alt="Cheese platter with various types of cheese"
                        fill
                        sizes="(max-width: 768px) 100vw, 400px"
                        style={{
                            objectFit: 'cover',
                        }}
                        priority
                    />
                </div>
            </div>
        </section>
    );
}