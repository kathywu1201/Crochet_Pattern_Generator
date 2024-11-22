import NextAuth from 'next-auth'
import CredentialsProvider from 'next-auth/providers/credentials'

export const authOptions = {
    providers: [
        CredentialsProvider({
            name: 'Credentials',
            credentials: {
                username: { label: "Username", type: "text" },
                password: { label: "Password", type: "password" }
            },
            async authorize(credentials) {
                if (credentials.username === "demo" && credentials.password === "password") {
                    return {
                        id: "1",
                        name: "Demo User",
                        email: "demo@example.com"
                    }
                }
                return null
            }
        })
    ],
    callbacks: {
        async session({ session, token }) {
            if (token) {
                session.user.sessionId = token.sessionId
            }
            return session
        },
        async jwt({ token, user }) {
            if (user && typeof window !== 'undefined') {
                token.sessionId = localStorage.getItem('userSessionId')
            }
            return token
        }
    },
    pages: {
        signIn: '/signin',
    }
}