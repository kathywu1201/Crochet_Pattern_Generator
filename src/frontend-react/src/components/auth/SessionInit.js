'use client'
import { useEffect } from 'react'
import { v4 as uuidv4 } from 'uuid'

export default function SessionInit() {
    useEffect(() => {
        if (typeof window !== 'undefined' && !localStorage.getItem('userSessionId')) {
            localStorage.setItem('userSessionId', uuidv4())
        }
    }, [])

    return null
}