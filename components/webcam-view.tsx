"use client"

import { useEffect, useRef, useState } from "react"
import { Camera, CameraOff } from "lucide-react"
import { detectEmotions, type Engine } from "@/lib/opencv"
import { drawResults } from "@/lib/draw"
import type { FaceResult } from "@/lib/emotions"

const ANALYSIS_WIDTH = 480
const FRAME_INTERVAL_MS = 120

export function WebcamView({ engine, onResults }: { engine: Engine; onResults: (results: FaceResult[] | null) => void }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const workCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<number | null>(null)
  const onResultsRef = useRef(onResults)
  const [state, setState] = useState<"idle" | "starting" | "live">("idle")
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    onResultsRef.current = onResults
  }, [onResults])

  const release = () => {
    if (timerRef.current) window.clearTimeout(timerRef.current)
    timerRef.current = null
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
  }

  useEffect(() => release, [])

  const analyzeFrame = () => {
    const video = videoRef.current
    const overlay = overlayRef.current
    if (!video || !overlay || !streamRef.current) return

    if (video.readyState >= 2 && video.videoWidth > 0) {
      const work = (workCanvasRef.current ??= document.createElement("canvas"))
      const scale = Math.min(1, ANALYSIS_WIDTH / video.videoWidth)
      work.width = Math.round(video.videoWidth * scale)
      work.height = Math.round(video.videoHeight * scale)
      work.getContext("2d", { willReadFrequently: true })?.drawImage(video, 0, 0, work.width, work.height)

      const faces = detectEmotions(engine, work)
      overlay.width = video.videoWidth
      overlay.height = video.videoHeight
      const ctx = overlay.getContext("2d")
      if (ctx) {
        ctx.clearRect(0, 0, overlay.width, overlay.height)
        drawResults(ctx, faces, 1 / scale)
      }
      onResultsRef.current(faces)
    }
    timerRef.current = window.setTimeout(analyzeFrame, FRAME_INTERVAL_MS)
  }

  const start = async () => {
    setError(null)
    setState("starting")
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      })
      streamRef.current = stream
      const video = videoRef.current
      if (!video) return
      video.srcObject = stream
      await video.play()
      setState("live")
      analyzeFrame()
    } catch (err) {
      release()
      setState("idle")
      setError(
        err instanceof DOMException && err.name === "NotAllowedError"
          ? "Camera access was blocked. Allow camera permission in your browser, or open the app in a new tab."
          : "Could not start the camera. Make sure one is connected and not in use by another app.",
      )
    }
  }

  const stop = () => {
    release()
    setState("idle")
    const overlay = overlayRef.current
    overlay?.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height)
    onResultsRef.current(null)
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="relative flex aspect-video items-center justify-center overflow-hidden rounded-lg bg-surface-2">
        <video
          ref={videoRef}
          playsInline
          muted
          className={`size-full object-contain ${state === "live" ? "" : "invisible"}`}
          aria-label="Webcam preview"
        />
        <canvas
          ref={overlayRef}
          className="pointer-events-none absolute inset-0 size-full object-contain"
          aria-hidden="true"
        />
        {state !== "live" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 p-6 text-center text-sm text-muted">
            <Camera className="size-8" aria-hidden="true" />
            <p>{state === "starting" ? "Starting camera..." : "Your camera feed is analyzed locally and never leaves your device."}</p>
          </div>
        )}
      </div>

      {error && (
        <p role="alert" className="text-sm text-red-400">
          {error}
        </p>
      )}

      <div>
        {state === "live" ? (
          <button
            type="button"
            onClick={stop}
            className="inline-flex items-center gap-2 rounded-md border border-border bg-surface-2 px-4 py-2 text-sm font-medium transition-colors hover:bg-border"
          >
            <CameraOff className="size-4" aria-hidden="true" />
            Stop camera
          </button>
        ) : (
          <button
            type="button"
            onClick={start}
            disabled={state === "starting"}
            className="inline-flex items-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90 disabled:opacity-60"
          >
            <Camera className="size-4" aria-hidden="true" />
            Start camera
          </button>
        )}
      </div>
    </div>
  )
}
