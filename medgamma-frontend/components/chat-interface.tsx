"use client"

import { useState, useRef, useEffect } from "react"
import { Send, Sparkles, User, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import type { ChatMessage } from "@/lib/types"
import { DEMO_QUESTIONS } from "@/lib/types"

interface ChatInterfaceProps {
  messages: ChatMessage[]
  isLoading: boolean
  onSendMessage: (message: string) => void
}

export function ChatInterface({ messages, isLoading, onSendMessage }: ChatInterfaceProps) {
  const [input, setInput] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [messages, isLoading])

  const handleSend = () => {
    const trimmed = input.trim()
    if (!trimmed || isLoading) return
    onSendMessage(trimmed)
    setInput("")
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col border-t border-border bg-card">
      {/* Chat header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-border">
        <div className="flex items-center gap-2">
          <Sparkles className="h-3.5 w-3.5 text-primary" />
          <span className="text-xs font-semibold text-foreground">Ask a question about the image</span>
        </div>
        {messages.length > 0 && (
          <span className="text-[10px] text-muted-foreground">
            {messages.filter((m) => m.role === "user").length} questions asked
          </span>
        )}
      </div>

      {/* Messages area */}
      <ScrollArea className="max-h-48 min-h-0">
        <div ref={scrollRef} className="space-y-3 p-4">
          {messages.length === 0 && !isLoading && (
            <div className="text-center py-3">
              <p className="text-xs text-muted-foreground mb-3">
                Select a sample or upload your own image, then ask any clinical question.
              </p>
              <div className="flex flex-wrap gap-1.5 justify-center">
                {DEMO_QUESTIONS.slice(0, 4).map((q) => (
                  <button
                    key={q}
                    onClick={() => onSendMessage(q)}
                    className="rounded-full border border-border bg-secondary/50 px-3 py-1 text-[11px] text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`flex gap-2 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              {msg.role === "assistant" && (
                <div className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-primary/10">
                  <Sparkles className="h-3 w-3 text-primary" />
                </div>
              )}
              <div
                className={`max-w-[80%] rounded-xl px-3 py-2 text-xs leading-relaxed ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-secondary text-secondary-foreground"
                }`}
              >
                {msg.content}
              </div>
              {msg.role === "user" && (
                <div className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-foreground/10">
                  <User className="h-3 w-3 text-foreground/60" />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-2">
              <div className="flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-full bg-primary/10">
                <Sparkles className="h-3 w-3 text-primary" />
              </div>
              <div className="flex flex-col gap-0.5 rounded-xl bg-secondary px-3 py-2 text-xs text-muted-foreground">
                <span className="flex items-center gap-1.5">
                  <Loader2 className="h-3 w-3 animate-spin" />
                  Analyzing... (约 2–3 分钟)
                </span>
                <a
                  href="https://maxsine2025-medical-image-analysis.hf.space/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[10px] text-primary hover:underline"
                >
                  若超时，可在此直接使用 HF Space 分析
                </a>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Input area */}
      <div className="flex items-center gap-2 border-t border-border px-4 py-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type a clinical question..."
          disabled={isLoading}
          className="flex-1 rounded-lg border border-input bg-background px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-primary disabled:opacity-50"
        />
        <Button
          size="icon"
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
          className="h-9 w-9 flex-shrink-0"
        >
          <Send className="h-4 w-4" />
        </Button>
      </div>
    </div>
  )
}
