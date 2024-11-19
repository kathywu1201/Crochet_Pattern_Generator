'use client'

import { useState, useEffect } from 'react'
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Info, Podcasts, Email, SmartToy } from '@mui/icons-material'; // Removed unused icons
import styles from './Header.module.css';


const navItems = [
    { name: 'About', path: '/about_us', sectionId: '', icon: <Info fontSize="small" /> },
    { name: 'Community', path: '/podcasts', sectionId: '', icon: <Podcasts fontSize="small" /> },
    { name: 'Patterns', path: '/chat', sectionId: '', icon: <Email fontSize="small" /> }
];

export default function Header() {
    const pathname = usePathname();

    return (
        <header className="fixed w-full-* top-4 z-50 left-14 right-14 rounded-full rounded-full shadow-lg"  style={{ backgroundColor: 'rgba(255, 253, 242, 0.85)' }}>
            <div className="container mx-auto px-2 py-2 flex items-center justify-between">
                {/* Logo */}
                <Link href="/" className="flex items-center">
                    <img src="/assets/home_logo.png" alt="Yarn Master Logo" className="h-8 w-auto mr-2" />
                </Link>

                {/* Navigation Links */}
                <nav className="hidden md:flex space-x-8">
                    {navItems.map((item) => (
                        <Link
                            key={item.name}
                            href={item.path}
                            className={`${styles.navLink} ${pathname === item.path ? styles.active : ''}`}
                        >
                            {item.name}
                        </Link>
                    ))}
                </nav>

                {/* Right-Side Special Link (Cheese Assistant) */}
                <Link href="/newsletters" className={`${styles.navLink} flex items-center`}>
                    <span>Log in</span>
                    <SmartToy fontSize="small" />
                </Link>
            </div>

            {/* Mobile Menu (Optional) */}
            <div className="md:hidden flex items-center justify-between">
                {/* Add a mobile menu toggle logic here if necessary */}
            </div>
        </header>
    );
}