import type { Metadata, Viewport } from 'next'
import { DM_Sans, DM_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import { Toaster } from 'sonner'
import './globals.css'

const dmSans = DM_Sans({ subsets: ["latin"], variable: "--font-sans" })
const dmMono = DM_Mono({ weight: ["400", "500"], subsets: ["latin"], variable: "--font-mono" })

export const metadata: Metadata = {
  title: 'MedGemma CXR Analysis - AI-Powered Chest X-Ray Interpretation',
  description: 'Interactive demo for AI-assisted chest X-ray report generation using a fine-tuned MedGemma model with LoRA adapter. Browse 233 CXR cases, compare reports, and ask clinical questions.',
  generator: 'v0.app',
  icons: {
    icon: [
      {
        url: '/icon-light-32x32.png',
        media: '(prefers-color-scheme: light)',
      },
      {
        url: '/icon-dark-32x32.png',
        media: '(prefers-color-scheme: dark)',
      },
      {
        url: '/icon.svg',
        type: 'image/svg+xml',
      },
    ],
    apple: '/apple-icon.png',
  },
}

export const viewport: Viewport = {
  themeColor: '#4a6cf7',
  userScalable: true,
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${dmSans.variable} ${dmMono.variable}`}>
      <body className="font-sans antialiased">
        {children}
        <Toaster position="bottom-right" richColors />
        <Analytics />
      </body>
    </html>
  )
}
