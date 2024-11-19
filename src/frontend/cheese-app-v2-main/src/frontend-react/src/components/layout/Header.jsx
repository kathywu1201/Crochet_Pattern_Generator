'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Info, Podcasts, Email, SmartToy } from '@mui/icons-material'; // Removed unused icons
import styles from './Header.module.css';

const navItems = [
    { name: 'About', path: '/', sectionId: 'about', icon: <Info fontSize="small" /> },
    { name: 'Community', path: '/podcasts', sectionId: 'podcasts', icon: <Podcasts fontSize="small" /> },
    { name: 'Log in / Sign up', path: '/newsletters', sectionId: 'newsletters', icon: <Email fontSize="small" /> },
    { name: 'Cheese Assistant', path: '/chat', sectionId: '', icon: <SmartToy fontSize="small" /> }
];

export default function Header() {
    const pathname = usePathname();
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    // Handle scrolling effect on header
    useEffect(() => {
        if (typeof window !== 'undefined') {
            const handleScroll = () => {
                setIsScrolled(window.scrollY > 50);
            };

            window.addEventListener('scroll', handleScroll);
            return () => window.removeEventListener('scroll', handleScroll);
        }
    }, []);

    // Modified buildHref function to return direct links
    function buildHref(item) {
        // If the item has a sectionId and it's on the home page, use the hash link.
        if (item.path === '/' && item.sectionId) {
            return `/about`; // Direct hash link
        }
        return item.path; // Otherwise, return the path
    }

    return (
        <header
            className={`fixed w-full top-0 z-50 transition-all duration-300 ${isScrolled ? 'bg-[rgba(255,253,242,0.9)]' : 'bg-transparent'}`}
        >
            <div className="container mx-auto px-4 h-20 flex items-center justify-between">
                <Link href="/" className="text-white hover:text-white/90 transition-colors">
                    {/* Logo link */}
                    <img
                        src="/assets/home_logo.png" // Path to your logo file
                        alt="Yarn Master Logo"
                        className="h-10" // Adjust height as needed
                    />
                </Link>

                <div className={styles.navLinks}>
                    {navItems.map((item) => (
                        <Link
                            key={item.name}
                            href={buildHref(item)}
                            className={`${styles.navLink} ${pathname === item.path ? styles.active : ''}`}
                        >
                            <span className={styles.icon}>{item.icon}</span>
                            <span className={styles.linkText}>{item.name}</span>
                        </Link>
                    ))}
                </div>

                {/* Mobile Menu Button */}
                <button
                    className="md:hidden p-2"
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    aria-label="Toggle mobile menu"
                >
                    <div className={`w-6 h-0.5 bg-white mb-1.5 transition-all ${isMobileMenuOpen ? 'rotate-45 translate-y-2' : ''}`} />
                    <div className={`w-6 h-0.5 bg-white mb-1.5 ${isMobileMenuOpen ? 'opacity-0' : ''}`} />
                    <div className={`w-6 h-0.5 bg-white transition-all ${isMobileMenuOpen ? '-rotate-45 -translate-y-2' : ''}`} />
                </button>

                {/* Mobile Menu */}
                <div
                    className={`
                        fixed md:hidden top-20 left-0 w-full bg-white shadow-lg transform transition-transform duration-300
                        ${isMobileMenuOpen ? 'translate-y-0' : '-translate-y-full'}
                    `}
                >
                    <nav className="flex flex-col p-4">
                        <Link
                            href="/"
                            className="py-3 text-gray-800 border-b border-gray-200"
                            onClick={() => setIsMobileMenuOpen(false)}
                        >
                            Home
                        </Link>
                        <Link
                            href="#about"
                            className="py-3 text-gray-800 border-b border-gray-200"
                            onClick={() => setIsMobileMenuOpen(false)}
                        >
                            About
                        </Link>
                        <Link
                            href="#podcasts"
                            className="py-3 text-gray-800"
                            onClick={() => setIsMobileMenuOpen(false)}
                        >
                            Podcasts
                        </Link>
                    </nav>
                </div>
            </div>
        </header>
    );
}
