'use client'

import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'

const data = [
  { time: '00:00', processing: 120, accuracy: 88, coverage: 95 },
  { time: '04:00', processing: 132, accuracy: 91, coverage: 97 },
  { time: '08:00', processing: 101, accuracy: 89, coverage: 93 },
  { time: '12:00', processing: 145, accuracy: 94, coverage: 98 },
  { time: '16:00', processing: 167, accuracy: 92, coverage: 96 },
  { time: '20:00', processing: 198, accuracy: 95, coverage: 99 },
  { time: '23:59', processing: 142, accuracy: 93, coverage: 97 },
]

export function AnalyticsChart() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
      {/* Processing Time Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Processing Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={data}>
              <defs>
                <linearGradient id="colorProcessing" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100, 200, 255, 0.1)" />
              <XAxis dataKey="time" stroke="rgba(255, 255, 255, 0.5)" />
              <YAxis stroke="rgba(255, 255, 255, 0.5)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'rgba(20, 20, 35, 0.9)',
                  border: '1px solid rgba(0, 212, 255, 0.3)',
                  borderRadius: '8px'
                }}
              />
              <Area 
                type="monotone" 
                dataKey="processing" 
                stroke="#00d4ff" 
                fillOpacity={1} 
                fill="url(#colorProcessing)" 
                name="Processing Time (ms)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Accuracy & Coverage Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(100, 200, 255, 0.1)" />
              <XAxis dataKey="time" stroke="rgba(255, 255, 255, 0.5)" />
              <YAxis stroke="rgba(255, 255, 255, 0.5)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'rgba(20, 20, 35, 0.9)',
                  border: '1px solid rgba(0, 212, 255, 0.3)',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="accuracy" 
                stroke="#10b981" 
                dot={{ fill: '#10b981', r: 4 }}
                activeDot={{ r: 6 }}
                name="Accuracy (%)"
              />
              <Line 
                type="monotone" 
                dataKey="coverage" 
                stroke="#f59e0b" 
                dot={{ fill: '#f59e0b', r: 4 }}
                activeDot={{ r: 6 }}
                name="Coverage (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
