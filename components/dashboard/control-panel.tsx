'use client'

import { useState, useRef } from 'react'
import { Upload, Play, Loader2, AlertCircle, CheckCircle } from 'lucide-react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'

type ProcessingStatus = 'idle' | 'uploading' | 'processing' | 'success' | 'error'

export function ControlPanel() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [status, setStatus] = useState<ProcessingStatus>('idle')
  const [message, setMessage] = useState('')
  const [processingTime, setProcessingTime] = useState(0)

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file)
      setMessage('')
    } else {
      setMessage('Please select a valid image file')
      setStatus('error')
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return

    try {
      setStatus('uploading')
      setMessage('Uploading image...')
      const filePath = await apiClient.uploadImage(selectedFile)
      setMessage('Image uploaded successfully')
      setStatus('idle')
      return filePath
    } catch (error) {
      setStatus('error')
      setMessage(`Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`)
      return null
    }
  }

  const handleProcess = async () => {
    if (!selectedFile) {
      setMessage('Please select an image first')
      setStatus('error')
      return
    }

    try {
      setStatus('processing')
      setMessage('Processing satellite imagery...')
      const startTime = Date.now()

      // Upload first
      const filePath = await handleUpload()
      if (!filePath) return

      // Then process
      const response = await apiClient.processImage({ image_path: filePath })
      
      const endTime = Date.now()
      const totalTime = endTime - startTime
      setProcessingTime(totalTime)

      if (response.success) {
        setStatus('success')
        setMessage(`Processing complete! Graph and solution generated. Time: ${(totalTime/1000).toFixed(2)}s`)
      } else {
        setStatus('error')
        setMessage(response.message || 'Processing failed')
      }
    } catch (error) {
      setStatus('error')
      setMessage(`Error: ${error instanceof Error ? error.message : 'Unknown error'}`)
    }
  }

  const getStatusIcon = () => {
    switch (status) {
      case 'uploading':
      case 'processing':
        return <Loader2 size={20} className="animate-spin text-primary" />
      case 'success':
        return <CheckCircle size={20} className="text-success" />
      case 'error':
        return <AlertCircle size={20} className="text-error" />
      default:
        return null
    }
  }

  const isProcessing = status === 'uploading' || status === 'processing'

  return (
    <Card className="space-y-4">
      <CardHeader>
        <CardTitle>Processing Controls</CardTitle>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* File Input */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          onChange={handleFileSelect}
          className="hidden"
        />

        {/* Selected File Display */}
        <div className="glass rounded-lg p-4 border glass-border">
          {selectedFile ? (
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-primary">{selectedFile.name}</p>
                <p className="text-xs text-foreground/60">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSelectedFile(null)
                  if (fileInputRef.current) fileInputRef.current.value = ''
                }}
              >
                Clear
              </Button>
            </div>
          ) : (
            <p className="text-sm text-foreground/70">No file selected</p>
          )}
        </div>

        {/* Upload Button */}
        <Button
          variant="secondary"
          className="w-full"
          onClick={() => fileInputRef.current?.click()}
          disabled={isProcessing}
        >
          <Upload size={18} className="mr-2" />
          Choose Image
        </Button>

        {/* Process Button */}
        <Button
          variant="default"
          className="w-full"
          onClick={handleProcess}
          disabled={!selectedFile || isProcessing}
        >
          {isProcessing ? (
            <>
              <Loader2 size={18} className="mr-2 animate-spin" />
              Processing...
            </>
          ) : (
            <>
              <Play size={18} className="mr-2" />
              Process Image
            </>
          )}
        </Button>

        {/* Status Message */}
        {message && (
          <div className={`flex items-start gap-3 glass rounded-lg p-3 border ${
            status === 'error' ? 'border-error/50' : 
            status === 'success' ? 'border-success/50' : 
            'glass-border'
          }`}>
            {getStatusIcon()}
            <div className="flex-1">
              <p className={`text-sm ${
                status === 'error' ? 'text-error' : 
                status === 'success' ? 'text-success' : 
                'text-foreground/80'
              }`}>
                {message}
              </p>
            </div>
          </div>
        )}

        {/* Processing Stats */}
        {processingTime > 0 && (
          <div className="glass rounded-lg p-3 border glass-border space-y-1">
            <p className="text-xs text-foreground/60">Processing Time</p>
            <p className="text-lg font-semibold text-primary">
              {(processingTime / 1000).toFixed(2)}s
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
