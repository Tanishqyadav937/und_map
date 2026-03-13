'use client'

import { useRef, useEffect, useState } from 'react'
import { Card } from '@/components/ui/card'
import { ZoomIn, ZoomOut, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface MapControls {
  zoom: number
  panX: number
  panY: number
}

export function SatelliteMap() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [controls, setControls] = useState<MapControls>({
    zoom: 1,
    panX: 0,
    panY: 0,
  })
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    // Set canvas size
    canvas.width = canvas.offsetWidth
    canvas.height = canvas.offsetHeight

    // Clear canvas
    ctx.fillStyle = 'rgba(15, 15, 30, 0.5)'
    ctx.fillRect(0, 0, canvas.width, canvas.height)

    // Draw grid background
    drawGridBackground(ctx, canvas.width, canvas.height, controls)

    // Draw placeholder satellite image representation
    drawSatelliteOverlay(ctx, canvas, controls)

    // Draw UI elements
    drawMapUI(ctx, canvas, controls)
  }, [controls])

  const drawGridBackground = (
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
    controls: MapControls
  ) => {
    const gridSize = 50 * controls.zoom
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.1)'
    ctx.lineWidth = 1

    for (let x = controls.panX % gridSize; x < width; x += gridSize) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, height)
      ctx.stroke()
    }

    for (let y = controls.panY % gridSize; y < height; y += gridSize) {
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }
  }

  const drawSatelliteOverlay = (
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    controls: MapControls
  ) => {
    // Draw center circle (representing satellite data)
    const centerX = canvas.width / 2
    const centerY = canvas.height / 2
    const radius = 150 * controls.zoom

    // Gradient for satellite footprint
    const gradient = ctx.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius)
    gradient.addColorStop(0, 'rgba(0, 212, 255, 0.3)')
    gradient.addColorStop(0.7, 'rgba(168, 85, 247, 0.1)')
    gradient.addColorStop(1, 'rgba(0, 212, 255, 0)')

    ctx.fillStyle = gradient
    ctx.beginPath()
    ctx.arc(centerX, centerY, radius, 0, Math.PI * 2)
    ctx.fill()

    // Draw border
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.5)'
    ctx.lineWidth = 2
    ctx.stroke()

    // Draw some grid points inside
    for (let i = 0; i < 5; i++) {
      for (let j = 0; j < 5; j++) {
        const x = centerX - 100 + (i * 50) * controls.zoom
        const y = centerY - 100 + (j * 50) * controls.zoom
        
        ctx.fillStyle = 'rgba(0, 212, 255, 0.4)'
        ctx.beginPath()
        ctx.arc(x, y, 3 * controls.zoom, 0, Math.PI * 2)
        ctx.fill()
      }
    }
  }

  const drawMapUI = (
    ctx: CanvasRenderingContext2D,
    canvas: HTMLCanvasElement,
    controls: MapControls
  ) => {
    // Draw coordinates text
    ctx.font = '12px monospace'
    ctx.fillStyle = 'rgba(0, 212, 255, 0.6)'
    ctx.fillText(`Zoom: ${controls.zoom.toFixed(1)}x`, 10, 20)
    ctx.fillText(`Pan: (${controls.panX.toFixed(0)}, ${controls.panY.toFixed(0)})`, 10, 35)
  }

  const handleZoom = (direction: 'in' | 'out') => {
    setControls(prev => ({
      ...prev,
      zoom: direction === 'in' ? Math.min(prev.zoom + 0.2, 5) : Math.max(prev.zoom - 0.2, 0.5),
    }))
  }

  const handleReset = () => {
    setControls({ zoom: 1, panX: 0, panY: 0 })
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true)
    setDragStart({ x: e.clientX, y: e.clientY })
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging) return

    const dx = e.clientX - dragStart.x
    const dy = e.clientY - dragStart.y

    setControls(prev => ({
      ...prev,
      panX: prev.panX + dx,
      panY: prev.panY + dy,
    }))

    setDragStart({ x: e.clientX, y: e.clientY })
  }

  const handleMouseUp = () => {
    setIsDragging(false)
  }

  return (
    <Card className="relative h-[600px] overflow-hidden">
      <canvas
        ref={canvasRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      />

      {/* Map Controls */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-2 z-10">
        <Button
          size="icon"
          variant="secondary"
          onClick={() => handleZoom('in')}
          title="Zoom In"
        >
          <ZoomIn size={18} />
        </Button>
        <Button
          size="icon"
          variant="secondary"
          onClick={() => handleZoom('out')}
          title="Zoom Out"
        >
          <ZoomOut size={18} />
        </Button>
        <Button
          size="icon"
          variant="secondary"
          onClick={handleReset}
          title="Reset View"
        >
          <RotateCcw size={18} />
        </Button>
      </div>

      {/* Info overlay */}
      <div className="absolute top-4 left-4 text-xs text-foreground/70 glass rounded px-3 py-2">
        <p>Drag to pan • Scroll wheel to zoom</p>
      </div>
    </Card>
  )
}
