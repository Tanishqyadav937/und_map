import { Sidebar } from '@/components/layout/sidebar'
import { Header } from '@/components/layout/header'
import { DashboardContent } from '@/components/dashboard/dashboard-content'

export default function Home() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-[#1a1a3e] to-background">
      <Sidebar />
      <Header />
      
      {/* Main content */}
      <main className="ml-64 mt-20 p-6">
        <DashboardContent />
      </main>
    </div>
  )
}
