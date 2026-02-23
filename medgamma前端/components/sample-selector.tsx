"use client"

import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import type { ReportEntry } from "@/lib/types"

interface SampleSelectorProps {
  reports: ReportEntry[]
  selectedIndex: number
  onSelectSample: (index: number) => void
}

export function SampleSelector({ reports, selectedIndex, onSelectSample }: SampleSelectorProps) {
  return (
    <div className="flex items-center gap-2 px-4 py-2 border-b border-border bg-card">
      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => onSelectSample(Math.max(0, selectedIndex - 1))}
        disabled={selectedIndex === 0}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>

      <Select
        value={String(selectedIndex)}
        onValueChange={(v) => onSelectSample(Number(v))}
      >
        <SelectTrigger className="h-8 w-40 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent className="max-h-64">
          {reports.map((r, i) => (
            <SelectItem key={r.subject_id} value={String(i)} className="text-xs">
              Case {String(i + 1).padStart(3, "0")} - {r.subject_id}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>

      <Button
        variant="ghost"
        size="icon"
        className="h-7 w-7"
        onClick={() => onSelectSample(Math.min(reports.length - 1, selectedIndex + 1))}
        disabled={selectedIndex >= reports.length - 1}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>

      <div className="ml-auto flex items-center gap-2">
        <div className="h-1.5 w-24 rounded-full bg-secondary overflow-hidden">
          <div
            className="h-full rounded-full bg-primary transition-all duration-300"
            style={{ width: `${((selectedIndex + 1) / reports.length) * 100}%` }}
          />
        </div>
        <span className="text-[10px] text-muted-foreground font-mono">
          {selectedIndex + 1}/{reports.length}
        </span>
      </div>
    </div>
  )
}
