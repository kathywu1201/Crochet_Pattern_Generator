'use client'

import Link from 'next/link'

export default function MobileMenu({ isOpen, onClose }) {
    const handleLinkClick = () => {
        onClose()
    }

    return (
        <div className={`
      fixed top-20 left-0 w-full bg-white shadow-lg transform transition-transform duration-300
      ${isOpen ? 'translate-y-0' : '-translate-y-full'}
      md:hidden
    `}>
            <nav className="flex flex-col p-4">
                <Link href="/" className="mobile-link" onClick={handleLinkClick}>
                    Home
                </Link>
                <Link href="#about" className="mobile-link" onClick={handleLinkClick}>
                    About
                </Link>
                <Link href="#podcasts" className="mobile-link" onClick={handleLinkClick}>
                    Podcasts
                </Link>
            </nav>
        </div>
    )
}