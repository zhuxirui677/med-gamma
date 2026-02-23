import { NextResponse } from "next/server"
import fs from "fs"
import path from "path"

export interface ReportEntry {
  index: number
  subject_id: string
  view: string
  image_path: string
  ground_truth: string
  generated_report: string
}

function parseCSVLine(line: string): string[] {
  const result: string[] = []
  let current = ""
  let inQuotes = false

  for (let i = 0; i < line.length; i++) {
    const char = line[i]
    if (char === '"') {
      if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
        current += '"'
        i++
      } else {
        inQuotes = !inQuotes
      }
    } else if (char === "," && !inQuotes) {
      result.push(current.trim())
      current = ""
    } else {
      current += char
    }
  }
  result.push(current.trim())
  return result
}

export async function GET() {
  try {
    const csvPath = path.join(process.cwd(), "lib", "reports-data.csv")
    const csvContent = fs.readFileSync(csvPath, "utf-8")
    const lines = csvContent.split("\n").filter((l) => l.trim())

    // Skip header
    const entries: ReportEntry[] = []
    for (let i = 1; i < lines.length; i++) {
      const cols = parseCSVLine(lines[i])
      if (cols.length >= 5) {
        entries.push({
          index: i - 1,
          subject_id: cols[0],
          view: cols[1],
          image_path: cols[2],
          ground_truth: cols[3],
          generated_report: cols[4],
        })
      }
    }

    return NextResponse.json({ reports: entries, total: entries.length })
  } catch (error) {
    console.error("Failed to parse CSV:", error)
    return NextResponse.json({ error: "Failed to load reports" }, { status: 500 })
  }
}
