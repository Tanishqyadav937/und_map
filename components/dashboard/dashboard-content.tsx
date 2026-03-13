'use client'

import { SatelliteMap } from './satellite-map'
import { ControlPanel } from './control-panel'
import { StatisticsCards } from './statistics-cards'
import { AnalyticsChart } from './analytics-chart'
import { Network3D } from './network-3d'

export function DashboardContent() {
  return (
    <div className="space-y-6">
      {/* Statistics Section */}
      <div>
        <h2 className="text-xl font-semibold mb-4 text-foreground">System Overview</h2>
        <StatisticsCards />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Map */}
        <div className="lg:col-span-2">
          <h2 className="text-lg font-semibold mb-4 text-foreground">Satellite Imagery View</h2>
          <SatelliteMap />
        </div>

        {/* Right: Control Panel */}
        <div>
          <h2 className="text-lg font-semibold mb-4 text-foreground">Processing</h2>
          <ControlPanel />
        </div>
      </div>

      {/* 3D Network Visualization */}
      <div>
        <h2 className="text-lg font-semibold mb-4 text-foreground">Processing Network</h2>
        <Network3D />
      </div>

      {/* Analytics Charts */}
      <div>
        <h2 className="text-lg font-semibold mb-4 text-foreground">Performance Analytics</h2>
        <AnalyticsChart />
      </div>
    </div>
  )
}
