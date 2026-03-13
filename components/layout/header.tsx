'use client'

import { Bell, User } from 'lucide-react'
import { Button } from '@/components/ui/button'

export function Header() {
  return (
    <header className="fixed top-0 right-0 left-64 md:left-64 z-20 glass glass-border border-b">
      <div className="flex items-center justify-between h-20 px-6">
        <h1 className="text-2xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
          Geospatial Analysis Dashboard
        </h1>
        
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon">
            <Bell size={20} className="text-primary" />
          </Button>
          <Button variant="ghost" size="icon">
            <User size={20} className="text-primary" />
          </Button>
        </div>
      </div>
    </header>
  )
}
