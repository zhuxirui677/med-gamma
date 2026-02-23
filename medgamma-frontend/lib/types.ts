export interface ReportEntry {
  index: number
  subject_id: string
  view: string
  image_path: string
  ground_truth: string
  generated_report: string
}

export interface ChatMessage {
  role: "user" | "assistant"
  content: string
  timestamp: Date
}

export const DEMO_QUESTIONS = [
  "Describe the key findings in this chest X-ray.",
  "Is there evidence of cardiomegaly?",
  "Are there any signs of pulmonary edema?",
  "Is there pleural effusion present?",
  "What is the overall impression of this image?",
  "Are there any signs of pneumothorax?",
  "Describe the mediastinal contours.",
  "Is there focal consolidation visible?",
]

// Configurable backend URL
export const API_BASE = process.env.NEXT_PUBLIC_MODAL_API_URL || ""
