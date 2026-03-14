'use client'

import { useRef, useEffect, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Html, Sphere, Line } from '@react-three/drei'
import * as THREE from 'three'
import { Card } from '@/components/ui/card'

// 3D Node component
function Node({ position, color, size = 0.3, label }: any) {
  const meshRef = useRef<THREE.Mesh>(null)
  const [hovered, setHovered] = useState(false)

  useFrame(() => {
    if (meshRef.current && hovered) {
      meshRef.current.scale.lerp(new THREE.Vector3(1.5, 1.5, 1.5), 0.1)
    } else if (meshRef.current) {
      meshRef.current.scale.lerp(new THREE.Vector3(1, 1, 1), 0.1)
    }
  })

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerEnter={() => setHovered(true)}
        onPointerLeave={() => setHovered(false)}
      >
        <sphereGeometry args={[size, 32, 32]} />
        <meshStandardMaterial
          color={color}
          emissive={hovered ? color : '#000000'}
          emissiveIntensity={hovered ? 0.8 : 0.2}
        />
      </mesh>
      {hovered && (
        <Html position={[0, size + 0.5, 0]} distanceFactor={1}>
          <div className="text-xs text-primary font-semibold bg-black/70 px-2 py-1 rounded whitespace-nowrap">
            {label}
          </div>
        </Html>
      )}
    </group>
  )
}

// Network connection line
function Connection({ start, end, color = '#00d4ff' }: any) {
  const points = [new THREE.Vector3(...start), new THREE.Vector3(...end)]
  
  return (
    <Line
      points={points}
      color={color}
      lineWidth={2}
      transparent
      opacity={0.6}
    />
  )
}

// Scene content
function NetworkScene() {
  // Generate network nodes
  const nodes = [
    { id: 1, position: [0, 0, 0], color: '#00d4ff', label: 'Central Hub', size: 0.5 },
    { id: 2, position: [3, 2, 1], color: '#a855f7', label: 'Analysis Node 1', size: 0.3 },
    { id: 3, position: [-3, 2, 1], color: '#a855f7', label: 'Analysis Node 2', size: 0.3 },
    { id: 4, position: [2, -2, 2], color: '#10b981', label: 'Processing Node 1', size: 0.3 },
    { id: 5, position: [-2, -2, 2], color: '#10b981', label: 'Processing Node 2', size: 0.3 },
    { id: 6, position: [0, 3, -2], color: '#f59e0b', label: 'Output Node', size: 0.3 },
  ]

  const connections = [
    { start: [0, 0, 0], end: [3, 2, 1] },
    { start: [0, 0, 0], end: [-3, 2, 1] },
    { start: [0, 0, 0], end: [2, -2, 2] },
    { start: [0, 0, 0], end: [-2, -2, 2] },
    { start: [0, 0, 0], end: [0, 3, -2] },
    { start: [3, 2, 1], end: [0, 3, -2] },
    { start: [-3, 2, 1], end: [0, 3, -2] },
  ]

  return (
    <>
      {/* Lighting */}
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1} color="#00d4ff" />
      <pointLight position={[-10, -10, 10]} intensity={0.6} color="#a855f7" />

      {/* Background */}
      <mesh position={[0, 0, -15]} scale={50}>
        <planeGeometry />
        <meshBasicMaterial color="#0f0f1e" />
      </mesh>

      {/* Grid */}
      <gridHelper args={[20, 20]} position={[0, -5, 0]} />

      {/* Nodes */}
      {nodes.map((node) => (
        <Node
          key={node.id}
          position={node.position}
          color={node.color}
          size={node.size}
          label={node.label}
        />
      ))}

      {/* Connections */}
      {connections.map((conn, idx) => (
        <Connection key={idx} start={conn.start} end={conn.end} />
      ))}

      {/* Orbit Controls */}
      <OrbitControls
        autoRotate
        autoRotateSpeed={2}
        enableZoom
        enablePan
        minDistance={10}
        maxDistance={30}
      />
    </>
  )
}

export function Network3D() {
  return (
    <Card className="relative h-[400px] overflow-hidden">
      <Canvas
        camera={{ position: [0, 5, 15], fov: 50 }}
        style={{ width: '100%', height: '100%' }}
      >
        <NetworkScene />
      </Canvas>

      {/* Info overlay */}
      <div className="absolute top-4 left-4 text-xs text-foreground/70 glass rounded px-3 py-2 z-10">
        <p>Drag to rotate • Scroll to zoom</p>
      </div>
    </Card>
  )
}
