import { NextRequest, NextResponse } from "next/server"

// This endpoint handles uploaded images - stores base64 temporarily
// In production, this would connect to the backend for real analysis
export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData()
    const file = formData.get("image") as File | null

    if (!file) {
      return NextResponse.json({ error: "No image provided" }, { status: 400 })
    }

    const bytes = await file.arrayBuffer()
    const base64 = Buffer.from(bytes).toString("base64")

    return NextResponse.json({
      success: true,
      image_b64: base64,
      filename: file.name,
      size: file.size,
    })
  } catch (error) {
    console.error("Upload error:", error)
    return NextResponse.json({ error: "Upload failed" }, { status: 500 })
  }
}
