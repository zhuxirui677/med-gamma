"use client"

import { useState, useEffect, useCallback } from "react"
import { toast } from "sonner"
import { Header } from "@/components/header"
import { OnboardingTour } from "@/components/onboarding-tour"
import { ImageViewer } from "@/components/image-viewer"
import { ReportPanel } from "@/components/report-panel"
import { ChatInterface } from "@/components/chat-interface"
import { SampleBrowser } from "@/components/sample-browser"
import { SampleSelector } from "@/components/sample-selector"
import { Skeleton } from "@/components/ui/skeleton"
import type { ReportEntry, ChatMessage } from "@/lib/types"
import { compressImageIfNeeded } from "@/lib/utils"

const TOUR_KEY = "medgemma-tour-completed"

export default function Home() {
  const [reports, setReports] = useState<ReportEntry[]>([])
  const [selectedIndex, setSelectedIndex] = useState(0)
  const [isLoading, setIsLoading] = useState(true)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([])
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [showTour, setShowTour] = useState(false)
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const [uploadedImageB64, setUploadedImageB64] = useState<string | null>(null)

  // Load reports data
  useEffect(() => {
    async function loadReports() {
      try {
        const res = await fetch("/api/reports")
        const data = await res.json()
        if (data.reports) {
          setReports(data.reports)
        }
      } catch (error) {
        console.error("Failed to load reports:", error)
        toast.error("Failed to load sample data")
      } finally {
        setIsLoading(false)
      }
    }
    loadReports()
  }, [])

  // Show tour on first visit
  useEffect(() => {
    const hasCompletedTour = localStorage.getItem(TOUR_KEY)
    if (!hasCompletedTour) {
      setShowTour(true)
    }
  }, [])

  const handleCloseTour = useCallback(() => {
    setShowTour(false)
    localStorage.setItem(TOUR_KEY, "true")
  }, [])

  const handleSelectSample = useCallback((index: number) => {
    setSelectedIndex(index)
    setUploadedImage(null)
    setUploadedImageB64(null)
    setChatMessages([])
  }, [])

  const handleUploadImage = useCallback(async (file: File) => {
    try {
      // 超过 4MB 时压缩，避免 Vercel 4.5MB 限制
      const toUpload = await compressImageIfNeeded(file)

      // Create local preview
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
      }
      reader.readAsDataURL(toUpload)

      // Upload and get base64
      const formData = new FormData()
      formData.append("image", toUpload)
      const res = await fetch("/api/upload", { method: "POST", body: formData })
      const data = await res.json()
      if (data.success) {
        setUploadedImageB64(data.image_b64)
        toast.success(`Image uploaded: ${file.name}`)
        setChatMessages([])
      } else {
        toast.error(data.error || "Failed to upload image")
      }
    } catch (error) {
      console.error("Upload failed:", error)
      toast.error("Failed to upload image")
    }
  }, [])

  const handleSendMessage = useCallback(
    async (message: string) => {
      const userMsg: ChatMessage = {
        role: "user",
        content: message,
        timestamp: new Date(),
      }
      setChatMessages((prev) => [...prev, userMsg])
      setIsChatLoading(true)

      try {
        const currentReport = reports[selectedIndex]
        // 有上传图用上传图，否则用当前样本的图片 URL（API 会拉取转 base64）
        const imageToSend = uploadedImageB64 || undefined
        const imageUrlToSend = !uploadedImageB64 && currentReport?.image_path?.startsWith("http")
          ? currentReport.image_path
          : undefined
        const promptMessage = imageToSend || imageUrlToSend
          ? message
          : `[Context: CXR for subject ${currentReport?.subject_id}. Ground truth: ${currentReport?.ground_truth?.substring(0, 200)}...]\n\nQuestion: ${message}`

        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: promptMessage,
            image_b64: imageToSend,
            image_url: imageUrlToSend,
            max_tokens: 512,
            temperature: 0.3,
          }),
        })

        const data = await res.json()
        const assistantMsg: ChatMessage = {
          role: "assistant",
          content: data.response || "Unable to generate a response.",
          timestamp: new Date(),
        }
        setChatMessages((prev) => [...prev, assistantMsg])
      } catch (error) {
        console.error("Chat error:", error)
        const errorMsg: ChatMessage = {
          role: "assistant",
          content:
            "Sorry, I encountered an error processing your request. Please check that the backend is configured and try again.",
          timestamp: new Date(),
        }
        setChatMessages((prev) => [...prev, errorMsg])
      } finally {
        setIsChatLoading(false)
      }
    },
    [reports, selectedIndex, uploadedImageB64]
  )

  const currentReport = reports[selectedIndex]

  if (isLoading) {
    return (
      <div className="flex h-screen flex-col bg-background">
        <div className="flex items-center gap-3 border-b border-border px-6 py-3">
          <Skeleton className="h-9 w-9 rounded-lg" />
          <div className="space-y-1">
            <Skeleton className="h-4 w-40" />
            <Skeleton className="h-3 w-56" />
          </div>
        </div>
        <div className="flex flex-1">
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-3">
              <Skeleton className="mx-auto h-48 w-48 rounded-lg" />
              <Skeleton className="h-3 w-32 mx-auto" />
            </div>
          </div>
          <div className="w-96 border-l border-border p-4 space-y-3">
            <Skeleton className="h-5 w-24" />
            <Skeleton className="h-24 w-full rounded-lg" />
            <Skeleton className="h-24 w-full rounded-lg" />
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-screen flex-col bg-background overflow-hidden">
      <Header onOpenGuide={() => setShowTour(true)} />

      <OnboardingTour isOpen={showTour} onClose={handleCloseTour} />

      {/* Sample selector bar */}
      {reports.length > 0 && (
        <SampleSelector
          reports={reports}
          selectedIndex={selectedIndex}
          onSelectSample={handleSelectSample}
        />
      )}

      {/* Main content */}
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {/* Left: Image viewer */}
        <div className="flex flex-1 flex-col min-w-0">
          <div className="flex-1 min-h-0">
            <ImageViewer
              imageUrl={
                currentReport?.image_path?.startsWith("http")
                  ? currentReport.image_path
                  : null
              }
              uploadedImage={uploadedImage}
              subjectId={currentReport?.subject_id || "N/A"}
              view={currentReport?.view || "N/A"}
              onUploadImage={handleUploadImage}
            />
          </div>

          {/* Sample thumbnails */}
          {reports.length > 0 && (
            <SampleBrowser
              reports={reports}
              selectedIndex={selectedIndex}
              onSelectSample={handleSelectSample}
            />
          )}
        </div>

        {/* Right: Reports + Chat */}
        <div className="flex w-[420px] flex-col border-l border-border min-h-0">
          <div className="flex-1 min-h-0 overflow-hidden">
            <ReportPanel
              groundTruth={currentReport?.ground_truth || ""}
              generatedReport={currentReport?.generated_report || ""}
              subjectId={currentReport?.subject_id || "N/A"}
            />
          </div>

          <ChatInterface
            messages={chatMessages}
            isLoading={isChatLoading}
            onSendMessage={handleSendMessage}
          />
        </div>
      </div>
    </div>
  )
}
