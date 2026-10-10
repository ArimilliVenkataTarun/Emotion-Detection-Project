import { ScanFace } from "lucide-react"
import { EmotionApp } from "@/components/emotion-app"

export default function Page() {
  return (
    <main className="mx-auto flex min-h-dvh max-w-6xl flex-col gap-8 px-4 py-8 md:px-8 md:py-12">
      <header className="flex flex-col gap-3">
        <div className="flex items-center gap-2 text-sm font-medium text-accent">
          <ScanFace className="size-4" aria-hidden="true" />
          <span>OpenCV + FER+</span>
        </div>
        <h1 className="text-balance text-3xl font-semibold tracking-tight md:text-4xl">Emotion Detection</h1>
        <p className="max-w-2xl text-pretty leading-relaxed text-muted">
          Faces are located with an OpenCV Haar cascade and classified into eight emotions by the FER+ network on
          OpenCV&apos;s DNN module. Everything runs in your browser, so no images are uploaded.
        </p>
      </header>
      <EmotionApp />
    </main>
  )
}
