"use client"

import { useState, useEffect } from "react"
import { X, ChevronRight, ChevronLeft, ImageIcon, FileText, MessageSquare, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"

const TOUR_STEPS = [
  {
    title: "Welcome to MedGemma CXR Analysis",
    description:
      "This interactive demo lets you explore AI-powered chest X-ray interpretation. Browse 233 pre-loaded CXR cases, compare ground truth reports with AI-generated reports, and ask questions about the images.",
    icon: ImageIcon,
    target: "welcome",
  },
  {
    title: "Browse Sample Cases",
    description:
      "Use the sample browser on the left to scroll through 233 CXR cases from the MIMIC-CXR dataset. Click any thumbnail to load the case. You can also use the dropdown to jump to a specific sample index.",
    icon: ImageIcon,
    target: "samples",
  },
  {
    title: "Upload Your Own Image",
    description:
      "Click the upload button to analyze your own chest X-ray image. The uploaded image will replace the current demo image and you can generate reports and ask questions about it.",
    icon: Upload,
    target: "upload",
  },
  {
    title: "Compare Reports",
    description:
      "The right panel shows two reports side by side: the Ground Truth (human radiologist report) and the AI-Generated Report from our student model. Compare them to evaluate the model performance.",
    icon: FileText,
    target: "reports",
  },
  {
    title: "Ask Clinical Questions",
    description:
      "Use the chat interface at the bottom to ask any clinical question about the current image. The AI model will analyze the image and provide a detailed response. Try example questions or type your own.",
    icon: MessageSquare,
    target: "chat",
  },
]

interface OnboardingTourProps {
  isOpen: boolean
  onClose: () => void
}

export function OnboardingTour({ isOpen, onClose }: OnboardingTourProps) {
  const [step, setStep] = useState(0)

  useEffect(() => {
    if (isOpen) setStep(0)
  }, [isOpen])

  if (!isOpen) return null

  const currentStep = TOUR_STEPS[step]
  const Icon = currentStep.icon

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-foreground/40 backdrop-blur-sm">
      <div className="mx-4 w-full max-w-md rounded-xl bg-card p-6 shadow-lg border border-border">
        <div className="flex items-start justify-between mb-4">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
            <Icon className="h-5 w-5 text-primary" />
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8 text-muted-foreground">
            <X className="h-4 w-4" />
          </Button>
        </div>

        <h3 className="text-lg font-semibold text-foreground mb-2">
          {currentStep.title}
        </h3>
        <p className="text-sm text-muted-foreground leading-relaxed mb-6">
          {currentStep.description}
        </p>

        {/* Progress dots */}
        <div className="flex items-center justify-center gap-1.5 mb-5">
          {TOUR_STEPS.map((_, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              className={`h-1.5 rounded-full transition-all ${
                i === step ? "w-6 bg-primary" : "w-1.5 bg-border"
              }`}
            />
          ))}
        </div>

        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setStep(Math.max(0, step - 1))}
            disabled={step === 0}
            className="gap-1"
          >
            <ChevronLeft className="h-4 w-4" />
            Back
          </Button>

          <span className="text-xs text-muted-foreground">
            {step + 1} / {TOUR_STEPS.length}
          </span>

          {step < TOUR_STEPS.length - 1 ? (
            <Button
              size="sm"
              onClick={() => setStep(step + 1)}
              className="gap-1"
            >
              Next
              <ChevronRight className="h-4 w-4" />
            </Button>
          ) : (
            <Button size="sm" onClick={onClose}>
              Get Started
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
