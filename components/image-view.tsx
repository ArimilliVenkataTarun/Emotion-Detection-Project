"use client"

import { useRef, useState } from "react"
import { ImageUp, Sparkles } from "lucide-react"
import { detectEmotions, type Engine } from "@/lib/opencv"
import { drawResults } from "@/lib/draw"
import type { FaceResult } from "@/lib/emotions"

const MAX_SIDE = 1024

export function ImageView({ engine, onResults }: { engine: Engine; onResults: (results: FaceResult[] | null) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const [hasImage, setHasImage] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const analyze = async (blob: Blob) => {
    setError(null)
    let bitmap: ImageBitmap
    try {
      bitmap = await createImageBitmap(blob)
    } catch {
      setError("Could not read that image. Please try a JPG or PNG file.")
      return
    }
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d", { willReadFrequently: true })
    if (!canvas || !ctx) return
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height))
    canvas.width = Math.round(bitmap.width * scale)
    canvas.height = Math.round(bitmap.height * scale)
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    bitmap.close()
    setHasImage(true)

    try {
      const faces = await detectEmotions(engine, canvas)
      drawResults(ctx, faces)
      onResults(faces)
    } catch (err) {
      setError(`Detection failed: ${err instanceof Error ? err.message : "unknown error"}`)
    }
  }

  const loadSample = async () => {
    try {
      const response = await fetch("/sample-face.png")
      if (!response.ok) throw new Error()
      await analyze(await response.blob())
    } catch {
      setError("Could not load the sample image.")
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <div
        onDragOver={(event) => {
          event.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault()
          setDragging(false)
          const file = event.dataTransfer.files[0]
          if (file?.type.startsWith("image/")) analyze(file)
        }}
        className={`relative flex aspect-video items-center justify-center overflow-hidden rounded-lg border bg-surface-2 transition-colors ${
          dragging ? "border-accent" : "border-transparent"
        }`}
      >
        <canvas
          ref={canvasRef}
          className={`max-h-full max-w-full ${hasImage ? "" : "hidden"}`}
          role="img"
          aria-label="Analyzed image with detected faces"
        />
        {!hasImage && (
          <div className="flex flex-col items-center gap-2 p-6 text-center text-sm text-muted">
            <ImageUp className="size-8" aria-hidden="true" />
            <p>Drop an image here or choose a file below.</p>
          </div>
        )}
      </div>

      {error && (
        <p role="alert" className="text-sm text-red-400">
          {error}
        </p>
      )}

      <div className="flex flex-wrap gap-2">
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="sr-only"
          id="image-upload"
          onChange={(event) => {
            const file = event.target.files?.[0]
            if (file) analyze(file)
            event.target.value = ""
          }}
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          className="inline-flex items-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90"
        >
          <ImageUp className="size-4" aria-hidden="true" />
          Upload image
        </button>
        <button
          type="button"
          onClick={loadSample}
          className="inline-flex items-center gap-2 rounded-md border border-border bg-surface-2 px-4 py-2 text-sm font-medium transition-colors hover:bg-border"
        >
          <Sparkles className="size-4" aria-hidden="true" />
          Try sample
        </button>
      </div>
    </div>
  )
}
