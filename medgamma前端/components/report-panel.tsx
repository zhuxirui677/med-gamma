"use client"

import { FileText, Sparkles, AlertTriangle, Copy, Check } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

interface ReportPanelProps {
  groundTruth: string
  generatedReport: string
  subjectId: string
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Button variant="ghost" size="icon" className="h-6 w-6" onClick={handleCopy} title="Copy report">
      {copied ? <Check className="h-3 w-3 text-primary" /> : <Copy className="h-3 w-3" />}
    </Button>
  )
}

export function ReportPanel({ groundTruth, generatedReport, subjectId }: ReportPanelProps) {
  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-border">
        <h3 className="text-sm font-semibold text-foreground">Analysis</h3>
        <p className="text-xs text-muted-foreground">Subject {subjectId}</p>
      </div>

      <Tabs defaultValue="reports" className="flex-1 flex flex-col min-h-0">
        <TabsList className="mx-4 mt-2 w-auto self-start">
          <TabsTrigger value="reports" className="text-xs gap-1.5">
            <FileText className="h-3 w-3" />
            Reports
          </TabsTrigger>
        </TabsList>

        <TabsContent value="reports" className="flex-1 min-h-0 px-4 pb-4">
          <ScrollArea className="h-full">
            <div className="space-y-4 pr-2 pt-2">
              {/* Ground Truth */}
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <FileText className="h-3.5 w-3.5 text-muted-foreground" />
                    <span className="text-xs font-semibold text-foreground uppercase tracking-wider">
                      Ground Truth
                    </span>
                  </div>
                  <CopyButton text={groundTruth} />
                </div>
                <p className="text-sm leading-relaxed text-foreground/80">
                  {groundTruth || "No ground truth report available for this case."}
                </p>
              </div>

              {/* Generated Report */}
              <div className="rounded-lg border border-primary/20 bg-primary/[0.02] p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Sparkles className="h-3.5 w-3.5 text-primary" />
                    <span className="text-xs font-semibold text-primary uppercase tracking-wider">
                      AI Generated Report
                    </span>
                  </div>
                  <CopyButton text={generatedReport} />
                </div>
                <p className="text-sm leading-relaxed text-foreground/80">
                  {generatedReport || "No AI-generated report available for this case."}
                </p>
              </div>

              {/* Disclaimer */}
              <div className="flex items-start gap-2 rounded-lg border border-warning/30 bg-warning/10 p-3">
                <AlertTriangle className="h-3.5 w-3.5 mt-0.5 flex-shrink-0 text-warning-foreground" />
                <p className="text-[11px] leading-relaxed text-warning-foreground">
                  This demonstration is for illustrative purposes of the model&apos;s baseline capabilities only.
                  It does not represent a finished or approved product, is not intended to diagnose or
                  suggest treatment of any disease or condition, and should not be used for medical advice.
                </p>
              </div>
            </div>
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  )
}
