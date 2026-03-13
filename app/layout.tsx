import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'UND Map - Geospatial AI Dashboard',
  description: 'Advanced satellite imagery analysis with AI-powered pathfinding and optimization',
  viewport: {
    width: 'device-width',
    initialScale: 1,
    maximumScale: 1,
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className="bg-gradient-to-br from-background via-[#1a1a3e] to-background text-foreground antialiased">
        {children}
      </body>
    </html>
  )
}
