"use client"

import { useRef, useEffect } from "react"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import type { ReportEntry } from "@/lib/types"

interface SampleBrowserProps {
  reports: ReportEntry[]
  selectedIndex: number
  onSelectSample: (index: number) => void
}

export function SampleBrowser({ reports, selectedIndex, onSelectSample }: SampleBrowserProps) {
  const scrollRef = useRef<HTMLDivElement>(null)
  const itemRefs = useRef<(HTMLButtonElement | null)[]>([])

  useEffect(() => {
    const el = itemRefs.current[selectedIndex]
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" })
    }
  }, [selectedIndex])

  return (
    <div className="border-t border-border bg-card">
      <div className="flex items-center justify-between px-4 py-2">
        <span className="text-xs font-medium text-muted-foreground">
          Samples ({reports.length})
        </span>
        <span className="text-xs text-muted-foreground">
          Case {selectedIndex + 1} of {reports.length}
        </span>
      </div>
      <ScrollArea className="w-full">
        <div ref={scrollRef} className="flex gap-2 px-4 pb-3">
          {reports.map((report, i) => (
            <button
              key={report.subject_id}
              ref={(el) => { itemRefs.current[i] = el }}
              onClick={() => onSelectSample(i)}
              className={`flex-shrink-0 flex flex-col items-center gap-1 rounded-lg border p-2 transition-all ${
                i === selectedIndex
                  ? "border-primary bg-primary/5 ring-1 ring-primary/20"
                  : "border-border bg-secondary/30 hover:border-primary/30 hover:bg-secondary"
              }`}
            >
              <div className="flex h-12 w-16 items-center justify-center rounded bg-muted text-[10px] text-muted-foreground font-mono">
                CXR
              </div>
              <span className={`text-[10px] font-medium leading-none ${
                i === selectedIndex ? "text-primary" : "text-muted-foreground"
              }`}>
                {String(i + 1).padStart(3, "0")}
              </span>
            </button>
          ))}
        </div>
        <ScrollBar orientation="horizontal" />
      </ScrollArea>
    </div>
  )
}
