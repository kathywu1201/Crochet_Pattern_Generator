import Link from 'next/link';
import styles from './Hero.module.css';

export default function Hero() {
    return (
        <section
            className="relative h-screen flex items-center justify-center text-center"
            style={{
                // backgroundImage: "linear-gradient(135deg, #FFFCF0 0%, #EBFFF0 100%), url('/assets/hero_background.png')", // Adjust gradient colors based on your image example
                // backgroundImage: "linear-gradient(135deg, rgba(255, 252, 240, 0.5), rgba(235, 255, 240, 0.5)), url('/assets/yarn2.jpeg')",
                backgroundImage: "url('/assets/yarn2.jpeg')",
                backgroundSize: 'cover',
                backgroundPosition: 'center'
            }}
        >
            <div className="container mx-auto px-4">
                <h1 className="text-5xl md:text-7xl font-playfair text-pink-600 mb-6">
                    🧀 Formaggio.me is here!
                </h1>
                <p className="text-xl md:text-2xl text-gray-700">
                    Discover the world of cheese through AI
                </p>
            </div>
            <div className={styles.viewAllContainer}>
                <Link href="/chat" className={styles.viewAllButton}>
                    Quick Start
                </Link>
            </div>
        </section>
    )
}
