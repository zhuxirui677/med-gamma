"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { Upload, ZoomIn, ZoomOut, RotateCcw, Maximize2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
interface ImageViewerProps {
  imageUrl: string | null
  uploadedImage: string | null
  subjectId: string
  view: string
  onUploadImage: (file: File) => void
}

export function ImageViewer({
  imageUrl,
  uploadedImage,
  subjectId,
  view,
  onUploadImage,
}: ImageViewerProps) {
  const [zoom, setZoom] = useState(100)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const displayUrl = uploadedImage || imageUrl || null
  const [imgError, setImgError] = useState(false)
  useEffect(() => {
    setImgError(false)
  }, [displayUrl])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file && file.type.startsWith("image/")) {
        onUploadImage(file)
      }
    },
    [onUploadImage]
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) {
        onUploadImage(file)
      }
    },
    [onUploadImage]
  )

  return (
    <div className="flex flex-col h-full">
      {/* Image info bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-card">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-foreground">
            Subject: {subjectId}
          </span>
          <span className="rounded bg-secondary px-1.5 py-0.5 text-[10px] font-medium text-secondary-foreground">
            {view}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7"
            onClick={() => fileInputRef.current?.click()}
            title="Upload image"
          >
            <Upload className="h-3.5 w-3.5" />
          </Button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </div>
      </div>

      {/* Image display area */}
      <div
        className={`relative flex-1 flex items-center justify-center overflow-hidden bg-foreground/[0.03] ${
          isDragging ? "ring-2 ring-primary ring-inset" : ""
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        {(displayUrl && !imgError) ? (
          <div
            className="transition-transform duration-200"
            style={{ transform: `scale(${zoom / 100})` }}
          >
            /* eslint-disable-next-line @next/next/no-img-element */
            <img
              src={displayUrl}
              alt={`Chest X-Ray for subject ${subjectId}`}
              className="max-h-[calc(100vh-320px)] max-w-full object-contain"
              onError={() => setImgError(true)}
            />
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 rounded-xl border-2 border-dashed border-border bg-muted/30 px-8 py-10">
            <div className="flex h-20 w-20 items-center justify-center rounded-2xl border-2 border-dashed border-border bg-background">
              <Upload className="h-10 w-10 text-muted-foreground" />
            </div>
            <div className="text-center space-y-1">
              <p className="text-base font-semibold text-foreground">上传 CXR 图片进行 AI 分析</p>
              <p className="text-sm text-muted-foreground">拖拽到此处，或点击上方上传按钮</p>
              <p className="text-xs text-muted-foreground">选择左侧样本可查看报告，点击样本切换病例</p>
            </div>
          </div>
        )}

        {isDragging && (
          <div className="absolute inset-0 flex items-center justify-center bg-primary/5 backdrop-blur-[1px]">
            <div className="rounded-xl border-2 border-dashed border-primary bg-card/80 px-8 py-6 text-center">
              <Upload className="mx-auto mb-2 h-8 w-8 text-primary" />
              <p className="text-sm font-medium text-primary">Drop to upload</p>
            </div>
          </div>
        )}
      </div>

      {/* Zoom controls */}
      <div className="flex items-center gap-3 border-t border-border bg-card px-4 py-2">
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setZoom(Math.max(50, zoom - 25))}
        >
          <ZoomOut className="h-3.5 w-3.5" />
        </Button>
        <Slider
          value={[zoom]}
          min={50}
          max={300}
          step={25}
          onValueChange={([v]) => setZoom(v)}
          className="flex-1"
        />
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setZoom(Math.min(300, zoom + 25))}
        >
          <ZoomIn className="h-3.5 w-3.5" />
        </Button>
        <span className="w-10 text-right text-[10px] font-mono text-muted-foreground">
          {zoom}%
        </span>
        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7"
          onClick={() => setZoom(100)}
          title="Reset zoom"
        >
          <RotateCcw className="h-3.5 w-3.5" />
        </Button>
      </div>
    </div>
  )
}
