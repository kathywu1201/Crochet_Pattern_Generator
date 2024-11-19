import Link from 'next/link';
import styles from './Hero.module.css';

export default function Hero() {
    return (
        <section
            className="relative h-screen flex items-center justify-center text-center"
            style={{
                // backgroundImage: "linear-gradient(135deg, #FFFCF0 0%, #EBFFF0 100%), url('/assets/hero_background.png')", // Adjust gradient colors based on your image example
                // backgroundImage: "linear-gradient(135deg, rgba(255, 252, 240, 0.5), rgba(235, 255, 240, 0.5)), url('/assets/yarn2.jpeg')",
                backgroundImage: "linear-gradient(45deg, rgba(251, 219, 20, 0.4), rgba(1, 249, 198, 0.4)),url('/assets/yarn3.jpeg')",
                backgroundSize: 'cover',
                backgroundPosition: 'center'
            }}
        >
            <div className="container mx-auto px-4">
                <h1 className="text-5xl md:text-7xl text-#FFFDF2 mb-6 " style={{ color: '#FFFFFF',textShadow: '2px 2px 4px rgba(0, 0, 0, 0.3)' }}>
                    FIRST-EVER Crochet/Knit AI
                </h1>
                <h6 className="text-xl md:text-2xl" style={{ color: '#FFFDF2',textShadow: '2px 2px 4px rgba(0, 0, 0, 0.3)' }}>
                    Learn how to crochet and knit with your personal AI instructor
                </h6>
            </div>
            <div className={`${styles.viewAllContainer} absolute bottom-12 left-1/2 transform -translate-x-1/2`}>
                <Link href="/chat" className={styles.viewAllButton}>
                    Quick Start
                </Link>
            </div>
        </section>
        
    )
}
