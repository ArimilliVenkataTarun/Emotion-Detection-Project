"use client"

import { useEffect, useState } from "react"
import { Camera, ImageIcon, Loader2, RotateCw } from "lucide-react"
import { loadEngine, type Engine, type EngineStatus } from "@/lib/opencv"
import type { FaceResult } from "@/lib/emotions"
import { ImageView } from "./image-view"
import { WebcamView } from "./webcam-view"
import { ResultsPanel } from "./results-panel"

type Mode = "image" | "webcam"

const MODES: { id: Mode; label: string; icon: typeof Camera }[] = [
  { id: "image", label: "Image", icon: ImageIcon },
  { id: "webcam", label: "Webcam", icon: Camera },
]

export function EmotionApp() {
  const [engine, setEngine] = useState<Engine | null>(null)
  const [status, setStatus] = useState<EngineStatus>({ stage: "Loading OpenCV.js and emotion model" })
  const [loadError, setLoadError] = useState<string | null>(null)
  const [mode, setMode] = useState<Mode>("image")
  const [results, setResults] = useState<FaceResult[] | null>(null)

  const [attempt, setAttempt] = useState(0)

  useEffect(() => {
    let active = true
    loadEngine((next) => {
      if (active) setStatus(next)
    })
      .then((loaded) => {
        if (active) setEngine(loaded)
      })
      .catch((error: Error) => {
        if (active) setLoadError(error.message)
      })
    return () => {
      active = false
    }
  }, [attempt])

  const retry = () => {
    setLoadError(null)
    setEngine(null)
    setStatus({ stage: "Starting detector", progress: 0 })
    setAttempt((value) => value + 1)
  }

  const switchMode = (next: Mode) => {
    setMode(next)
    setResults(null)
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_340px]">
      <section aria-label="Input" className="flex flex-col gap-4 rounded-xl border border-border bg-surface p-4 md:p-5">
        <div className="flex items-center justify-between gap-4">
          <div role="tablist" aria-label="Input source" className="inline-flex rounded-md bg-surface-2 p-1">
            {MODES.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={mode === id}
                onClick={() => switchMode(id)}
                className={`inline-flex items-center gap-2 rounded px-3 py-1.5 text-sm font-medium transition-colors ${
                  mode === id ? "bg-border text-foreground" : "text-muted hover:text-foreground"
                }`}
              >
                <Icon className="size-4" aria-hidden="true" />
                {label}
              </button>
            ))}
          </div>
          <span className="flex items-center gap-2 text-xs text-muted">
            <span
              className={`size-2 rounded-full ${loadError ? "bg-red-500" : engine ? "bg-emerald-500" : "bg-accent"}`}
              aria-hidden="true"
            />
            {loadError ? "Engine error" : engine ? "OpenCV ready" : "Loading"}
          </span>
        </div>

        {loadError ? (
          <div role="alert" className="flex aspect-video flex-col items-center justify-center gap-2 rounded-lg bg-surface-2 p-6 text-center">
            <p className="font-medium">Could not start the detector</p>
            <p className="text-sm text-muted">{loadError}</p>
            <button
              type="button"
              onClick={retry}
              className="mt-2 inline-flex items-center gap-2 rounded-md bg-accent px-4 py-2 text-sm font-medium text-background transition-opacity hover:opacity-90"
            >
              <RotateCw className="size-4" aria-hidden="true" />
              Retry
            </button>
          </div>
        ) : !engine ? (
          <div
            role="status"
            className="flex aspect-video flex-col items-center justify-center gap-4 rounded-lg bg-surface-2 p-6 text-center"
          >
            <Loader2 className="size-6 animate-spin text-accent" aria-hidden="true" />
            <div className="flex w-full max-w-xs flex-col gap-2">
              <p className="text-sm">{status.stage}</p>
              {status.progress !== undefined && (
                <>
                  <div className="h-1.5 overflow-hidden rounded-full bg-border">
                    <div
                      className="h-full rounded-full bg-accent transition-[width]"
                      style={{ width: `${Math.round(status.progress * 100)}%` }}
                    />
                  </div>
                  <p className="font-mono text-xs text-muted">{Math.round(status.progress * 100)}%</p>
                </>
              )}
            </div>
          </div>
        ) : mode === "image" ? (
          <ImageView engine={engine} onResults={setResults} />
        ) : (
          <WebcamView engine={engine} onResults={setResults} />
        )}
      </section>

      <ResultsPanel results={results} />
    </div>
  )
}
