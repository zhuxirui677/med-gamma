"use client"

import { Activity, HelpCircle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface HeaderProps {
  onOpenGuide: () => void
}

export function Header({ onOpenGuide }: HeaderProps) {
  return (
    <header className="sticky top-0 z-50 flex items-center justify-between border-b border-border bg-card px-6 py-3">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary">
          <Activity className="h-5 w-5 text-primary-foreground" />
        </div>
        <div>
          <h1 className="text-base font-semibold text-foreground leading-tight">
            MedGemma CXR Analysis
          </h1>
          <p className="text-xs text-muted-foreground leading-tight">
            Powered by MedGemma 1.5 &middot; LoRA fine-tuned &middot; 233 CXR cases
          </p>
        </div>
      </div>
      <div className="flex items-center gap-2">
        <Button
          variant="ghost"
          size="sm"
          onClick={onOpenGuide}
          className="gap-1.5 text-muted-foreground hover:text-foreground"
        >
          <HelpCircle className="h-4 w-4" />
          <span className="hidden sm:inline">Guide</span>
        </Button>
      </div>
    </header>
  )
}
