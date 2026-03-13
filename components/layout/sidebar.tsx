'use client'

import { useState } from 'react'
import { Menu, X, Map, Settings, Info, BarChart3 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface NavItem {
  label: string
  icon: React.ReactNode
  active?: boolean
}

const navItems: NavItem[] = [
  { label: 'Dashboard', icon: <Map size={20} />, active: true },
  { label: 'Analytics', icon: <BarChart3 size={20} /> },
  { label: 'Settings', icon: <Settings size={20} /> },
  { label: 'About', icon: <Info size={20} /> },
]

export function Sidebar() {
  const [isOpen, setIsOpen] = useState(true)

  return (
    <>
      {/* Mobile toggle */}
      <Button
        variant="ghost"
        size="icon"
        className="fixed top-4 left-4 z-50 md:hidden"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? <X size={24} /> : <Menu size={24} />}
      </Button>

      {/* Sidebar */}
      <aside
        className={cn(
          'fixed left-0 top-0 h-full w-64 glass glass-border transition-all duration-300 ease-in-out z-40',
          isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'
        )}
      >
        {/* Logo */}
        <div className="flex items-center justify-center h-20 border-b glass-border">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-gradient-to-br from-primary to-accent flex items-center justify-center">
              <Map size={18} className="text-background" />
            </div>
            <span className="text-lg font-bold text-primary">UND Map</span>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item, idx) => (
            <Button
              key={idx}
              variant={item.active ? 'default' : 'ghost'}
              className={cn(
                'w-full justify-start gap-3',
                item.active && 'shadow-glow'
              )}
            >
              <span className="text-lg">{item.icon}</span>
              <span>{item.label}</span>
            </Button>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t glass-border p-4 space-y-2">
          <p className="text-xs text-foreground/60 text-center">
            v1.0.0
          </p>
          <p className="text-xs text-foreground/50 text-center">
            Geospatial AI Dashboard
          </p>
        </div>
      </aside>

      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 md:hidden z-30"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  )
}
