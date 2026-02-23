import { NextRequest, NextResponse } from "next/server"

// 支持两种后端：Modal 或 HF Gradio Space
const MODAL_API_URL = process.env.MODAL_API_URL || ""
const HF_GRADIO_SPACE = process.env.HF_GRADIO_SPACE || "" // 如 "username/medgemma-assistant"

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const { message, image_b64, max_tokens = 512, temperature = 0.3 } = body

    // 1. 优先使用 Modal 后端
    if (MODAL_API_URL) {
      const res = await fetch(`${MODAL_API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, image_b64, max_tokens, temperature }),
      })
      if (res.ok) {
        const data = await res.json()
        return NextResponse.json(data)
      }
    }

    // 2. 使用 HF Gradio Space 后端（两步：POST 获取 event_id，GET 获取结果）
    if (HF_GRADIO_SPACE) {
      const gradioUrl = `https://${HF_GRADIO_SPACE.replace("/", "-")}.hf.space`
      const apiName = image_b64 ? "analyze" : "chat"
      const callUrl = `${gradioUrl}/call/${apiName}`

      // 构建 data 数组（Gradio Image 需 {"path": url}，data URL 可用）
      const imageDataUrl = image_b64
        ? (image_b64.startsWith("data:") ? image_b64 : `data:image/jpeg;base64,${image_b64}`)
        : null
      const imagePayload = imageDataUrl ? { path: imageDataUrl } : null
      const dataArr = image_b64
        ? [imagePayload, message || "Describe the key findings in this chest X-ray.", max_tokens, temperature]
        : [message, [], null, max_tokens, temperature]

      // Step 1: POST 获取 event_id
      const postRes = await fetch(callUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ data: dataArr }),
      })
      if (!postRes.ok) throw new Error(`Gradio POST returned ${postRes.status}`)
      const { event_id } = await postRes.json()
      if (!event_id) throw new Error("No event_id from Gradio")

      // Step 2: GET 获取结果（轮询直到 complete）
      const getUrl = `${callUrl}/${event_id}`
      let responseText = ""
      const getRes = await fetch(getUrl, { headers: { Accept: "text/event-stream" } })
      if (!getRes.ok) throw new Error(`Gradio GET returned ${getRes.status}`)
      const text = await getRes.text()
      const completeMatch = text.match(/event: complete\s+data: (\[.*\]|".*")/s)
      if (completeMatch) {
        try {
          const parsed = JSON.parse(completeMatch[1])
          responseText = Array.isArray(parsed) ? (Array.isArray(parsed[0]) ? parsed[0][parsed[0].length - 1]?.[1] || parsed[0] : String(parsed[0])) : String(parsed)
        } catch {
          responseText = completeMatch[1]
        }
      }
      if (!responseText) throw new Error("Could not parse Gradio response")

      return NextResponse.json({ response: responseText })
    }

    // 3. 无后端配置时返回 mock
    return NextResponse.json({
      response: generateMockResponse(message),
    })
  } catch (error) {
    console.error("Chat API error:", error)
    return NextResponse.json({
      response: "The backend model is currently unavailable. Please set MODAL_API_URL or HF_GRADIO_SPACE to connect your model.",
    })
  }
}

function generateMockResponse(message: string): string {
  const lower = message.toLowerCase()
  if (lower.includes("finding") || lower.includes("describe") || lower.includes("report")) {
    return "Based on the provided chest X-ray image, the findings include: The heart size appears within normal limits. The mediastinal contours are unremarkable. The lungs are clear bilaterally. Impression: No acute cardiopulmonary process identified. (Demo response - connect HF_GRADIO_SPACE for real AI analysis.)"
  }
  if (lower.includes("cardiomegaly") || lower.includes("heart")) {
    return "Cardiomegaly refers to an enlarged heart. Common causes include hypertension, valvular disease, cardiomyopathy. (Demo response.)"
  }
  return "Thank you for your question. Connect the MedGemma backend (HF_GRADIO_SPACE) for real AI-powered analysis. (Demo response.)"
}
