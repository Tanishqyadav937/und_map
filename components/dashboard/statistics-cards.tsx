'use client'

import { Card, CardContent } from '@/components/ui/card'
import { TrendingUp, MapPin, Zap, Clock } from 'lucide-react'

interface StatCard {
  label: string
  value: string | number
  change?: string
  icon: React.ReactNode
  color: 'primary' | 'accent' | 'success' | 'warning'
}

const stats: StatCard[] = [
  {
    label: 'Areas Analyzed',
    value: '2,847',
    change: '+12% from last week',
    icon: <MapPin size={24} />,
    color: 'primary',
  },
  {
    label: 'Processing Speed',
    value: '4.2s',
    change: '-15% improvement',
    icon: <Zap size={24} />,
    color: 'accent',
  },
  {
    label: 'Success Rate',
    value: '98.5%',
    change: 'All systems optimal',
    icon: <TrendingUp size={24} />,
    color: 'success',
  },
  {
    label: 'Avg Processing Time',
    value: '3.8s',
    change: 'Per satellite image',
    icon: <Clock size={24} />,
    color: 'warning',
  },
]

const colorMap = {
  primary: 'text-primary',
  accent: 'text-accent',
  success: 'text-success',
  warning: 'text-warning',
}

export function StatisticsCards() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, idx) => (
        <Card key={idx}>
          <CardContent className="pt-6">
            <div className="flex items-start justify-between mb-3">
              <div className={`${colorMap[stat.color]}`}>
                {stat.icon}
              </div>
            </div>
            <div>
              <p className="text-sm text-foreground/70 mb-1">{stat.label}</p>
              <p className="text-2xl font-bold mb-2">{stat.value}</p>
              {stat.change && (
                <p className="text-xs text-foreground/50">{stat.change}</p>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
