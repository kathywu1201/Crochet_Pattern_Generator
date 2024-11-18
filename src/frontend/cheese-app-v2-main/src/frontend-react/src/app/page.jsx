import Hero from '@/components/home/Hero';
import About from '@/components/home/About';
import Podcasts from '@/components/home/Podcasts';
import Newsletters from '@/components/home/Newsletters';
import WhatIs from '@/components/home/WhatIs';

export default function Home() {
    return (
        <>
            <Hero />
            <WhatIs></WhatIs>
            <Podcasts />
            <Newsletters />
            <About />
        </>
    )
}