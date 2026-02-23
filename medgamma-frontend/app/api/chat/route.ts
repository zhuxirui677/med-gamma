import { NextRequest, NextResponse } from "next/server"

// 延长超时：MedGemma 推理约 30–60 秒
export const maxDuration = 60

// 支持两种后端：Modal 或 HF Gradio Space
const MODAL_API_URL = process.env.MODAL_API_URL || ""
const HF_GRADIO_SPACE = process.env.HF_GRADIO_SPACE || ""
const HF_TOKEN = process.env.HF_TOKEN || ""

export async function POST(req: NextRequest) {
  let message = ""
  try {
    const body = await req.json()
    message = body?.message ?? ""
    let { image_b64, image_url, max_tokens = 512, temperature = 0.3 } = body

    // 若有 image_url（样本图片）但无 image_b64，服务端拉取并转 base64
    if (!image_b64 && image_url && typeof image_url === "string" && image_url.startsWith("http")) {
      try {
        const imgRes = await fetch(image_url, { headers: { "User-Agent": "MedGemma-Demo/1.0" } })
        if (imgRes.ok) {
          const buf = await imgRes.arrayBuffer()
          image_b64 = Buffer.from(buf).toString("base64")
        }
      } catch (e) {
        console.warn("Failed to fetch image_url:", e)
      }
    }

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

    // 2. 使用 HF Gradio Space 后端（Gradio 6.x 使用 /gradio_api 前缀）
    if (HF_GRADIO_SPACE) {
      const gradioUrl = `https://${HF_GRADIO_SPACE.replace("/", "-")}.hf.space`
      const apiName = image_b64 ? "analyze" : "chat"
      const callUrl = `${gradioUrl}/gradio_api/call/${apiName}`

      // 构建 data 数组（Gradio ImageData 对 data URL 只检查 url 字段，path 会触发 File name too long）
      const imageDataUrl = image_b64
        ? (image_b64.startsWith("data:") ? image_b64 : `data:image/jpeg;base64,${image_b64}`)
        : null
      const imagePayload = imageDataUrl ? { url: imageDataUrl } : null
      const dataArr = image_b64
        ? [imagePayload, message || "Describe the key findings in this chest X-ray.", max_tokens, temperature]
        : [message, [], null, max_tokens, temperature]

      const headers: Record<string, string> = { "Content-Type": "application/json" }
      if (HF_TOKEN) headers["Authorization"] = `Bearer ${HF_TOKEN}`

      const ctrl = new AbortController()
      const timeout = setTimeout(() => ctrl.abort(), 55000)
      const postRes = await fetch(callUrl, {
        method: "POST",
        headers,
        body: JSON.stringify({ data: dataArr }),
        signal: ctrl.signal,
      })
      clearTimeout(timeout)
      if (!postRes.ok) throw new Error(`Gradio POST returned ${postRes.status}`)
      const { event_id } = await postRes.json()
      if (!event_id) throw new Error("No event_id from Gradio")

      // Step 2: GET 获取结果（SSE 流，等待 complete）
      const getUrl = `${callUrl}/${event_id}`
      let responseText = ""
      const ctrl2 = new AbortController()
      const timeout2 = setTimeout(() => ctrl2.abort(), 55000)
      const getHeaders: Record<string, string> = { Accept: "text/event-stream" }
      if (HF_TOKEN) getHeaders["Authorization"] = `Bearer ${HF_TOKEN}`
      const getRes = await fetch(getUrl, {
        headers: getHeaders,
        signal: ctrl2.signal,
      })
      clearTimeout(timeout2)
      if (!getRes.ok) throw new Error(`Gradio GET returned ${getRes.status}`)
      const text = await getRes.text()

      // 检测 error 事件（HF Space 模型错误、OOM 等）
      const errorMatch = text.match(/event:\s*error\s+data:\s*(.+?)(?=\r?\n\r?\n|$)/s)
      if (errorMatch) {
        let errMsg = "HF Space returned an error"
        try {
          const errData = errorMatch[1].trim()
          if (errData && errData !== "null") {
            const parsed = JSON.parse(errData)
            errMsg = typeof parsed === "string" ? parsed : (parsed?.message || parsed?.error || JSON.stringify(parsed))
          } else {
            errMsg = "Model may be overloaded or restarting. Please try again in a moment."
          }
        } catch {
          errMsg = errorMatch[1] || errMsg
        }
        throw new Error(errMsg)
      }

      // 解析 complete 事件（支持 \n 和 \r\n）
      const completeMatch = text.match(/event:\s*complete\s+data:\s*(\S[\s\S]*?)(?=\r?\n\r?\n|$)/s)
      if (completeMatch) {
        try {
          const dataStr = completeMatch[1].trim()
          const parsed = JSON.parse(dataStr)
          // analyze: 返回 [string]；chat: 返回 [[msg,reply], [msg,reply]]，取最后一条 reply
          if (Array.isArray(parsed)) {
            const last = parsed[parsed.length - 1]
            responseText = Array.isArray(last) ? (last[1] ?? String(last)) : String(last ?? parsed[0])
          } else {
            responseText = String(parsed)
          }
        } catch {
          responseText = completeMatch[1].trim()
        }
      }
      if (!responseText) {
        // 兜底：尝试逐行解析 SSE
        const lines = text.split(/\r?\n/)
        let lastEvent = ""
        let lastData = ""
        for (let i = 0; i < lines.length; i++) {
          if (lines[i].startsWith("event:")) lastEvent = lines[i].replace(/^event:\s*/, "").trim()
          if (lines[i].startsWith("data:")) lastData = lines[i].replace(/^data:\s*/, "").trim()
        }
        if (lastEvent === "error") {
          throw new Error(lastData && lastData !== "null" ? lastData : "Model may be overloaded or restarting. Please try again in a moment.")
        }
        if (lastEvent === "complete" && lastData) {
          try {
            const parsed = JSON.parse(lastData)
            const last = Array.isArray(parsed) ? parsed[parsed.length - 1] : parsed
            responseText = Array.isArray(last) ? (last[1] ?? String(last[0])) : String(parsed)
          } catch {
            responseText = lastData
          }
        }
        if (!responseText) throw new Error("Could not parse Gradio response")
      }

      return NextResponse.json({ response: responseText })
    }

    // 3. 无后端配置时返回 mock
    return NextResponse.json({
      response: generateMockResponse(message),
    })
  } catch (error) {
    console.error("Chat API error:", error)
    const msg = error instanceof Error ? error.message : String(error)
    const isAbort = error instanceof Error && error.name === "AbortError"
    if (isAbort) {
      return NextResponse.json({
        response: "请求超时。HF Space 推理较慢（模型加载约 2–5 分钟），请稍后重试。",
      })
    }
    if (msg.includes("overloaded") || msg.includes("restarting") || msg.includes("Could not parse Gradio")) {
      const mock = generateMockResponse(message)
      return NextResponse.json({
        response: `${mock}\n\n_(HF Space 当前繁忙或响应异常。请确认：1) 打开 https://maxsine2025-medical-image-analysis.hf.space/ 确认 Space 已 Running；2) Vercel 环境变量 HF_GRADIO_SPACE=Maxsine2025/medical-image-analysis 已设置。)_`,
      })
    }
    return NextResponse.json({
      response: `Backend error: ${msg}\n\n请检查 HF Space 是否 Running：https://maxsine2025-medical-image-analysis.hf.space/`,
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
