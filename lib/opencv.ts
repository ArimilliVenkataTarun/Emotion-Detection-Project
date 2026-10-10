import type { FaceResult } from "./emotions"

export type EngineStatus = { stage: string; progress?: number }
export type Engine = { detect: (image: ImageData) => Promise<FaceResult[]> }

const WORKER_URL = "/workers/emotion-worker.js"
const STALL_TIMEOUT_MS = 45_000
const DETECT_TIMEOUT_MS = 15_000

type WorkerMessage =
  | { type: "status"; stage: string; progress?: number }
  | { type: "ready" }
  | { type: "error"; message: string }
  | { type: "result"; id: number; faces: FaceResult[] }
  | { type: "detectError"; id: number; message: string }

type Pending = { resolve: (faces: FaceResult[]) => void; reject: (error: Error) => void; timer: number }

let worker: Worker | null = null
let enginePromise: Promise<Engine> | null = null
let statusListener: ((status: EngineStatus) => void) | null = null
let nextRequestId = 0
const pending = new Map<number, Pending>()

function teardown(reason: string) {
  worker?.terminate()
  worker = null
  enginePromise = null
  for (const request of pending.values()) {
    window.clearTimeout(request.timer)
    request.reject(new Error(reason))
  }
  pending.clear()
}

function createEngine(target: Worker): Engine {
  return {
    detect(image) {
      if (worker !== target) return Promise.reject(new Error("The detector stopped. Retry loading it."))
      const id = ++nextRequestId
      return new Promise<FaceResult[]>((resolve, reject) => {
        const timer = window.setTimeout(() => {
          pending.delete(id)
          reject(new Error("Detection timed out"))
        }, DETECT_TIMEOUT_MS)
        pending.set(id, { resolve, reject, timer })
        target.postMessage(
          { type: "detect", id, width: image.width, height: image.height, data: image.data },
          [image.data.buffer],
        )
      })
    },
  }
}

/**
 * Starts the OpenCV worker once and shares the result. A failed load is not retried
 * automatically; call this again (e.g. from a Retry button) to make a fresh attempt.
 */
export function loadEngine(onStatus: (status: EngineStatus) => void): Promise<Engine> {
  statusListener = onStatus
  if (enginePromise) return enginePromise

  enginePromise = new Promise<Engine>((resolve, reject) => {
    const target = new Worker(WORKER_URL)
    worker = target
    let settled = false
    let stallTimer = 0

    const fail = (message: string) => {
      window.clearTimeout(stallTimer)
      if (worker === target) teardown(message)
      if (!settled) {
        settled = true
        reject(new Error(message))
      }
    }
    const armStallTimer = () => {
      window.clearTimeout(stallTimer)
      stallTimer = window.setTimeout(
        () => fail(`Loading stalled: no progress for ${STALL_TIMEOUT_MS / 1000} seconds`),
        STALL_TIMEOUT_MS,
      )
    }

    target.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data
      switch (message.type) {
        case "status":
          armStallTimer()
          statusListener?.({ stage: message.stage, progress: message.progress })
          break
        case "ready":
          window.clearTimeout(stallTimer)
          settled = true
          resolve(createEngine(target))
          break
        case "error":
          fail(message.message)
          break
        case "result":
        case "detectError": {
          const request = pending.get(message.id)
          if (!request) break
          pending.delete(message.id)
          window.clearTimeout(request.timer)
          if (message.type === "result") request.resolve(message.faces)
          else request.reject(new Error(message.message))
          break
        }
      }
    }
    target.onerror = (event) => {
      event.preventDefault()
      fail(event.message || "The detector worker crashed")
    }

    armStallTimer()
    statusListener?.({ stage: "Starting detector", progress: 0 })
    target.postMessage({ type: "init" })
  })

  return enginePromise
}

export function detectEmotions(engine: Engine, canvas: HTMLCanvasElement): Promise<FaceResult[]> {
  const ctx = canvas.getContext("2d", { willReadFrequently: true })
  if (!ctx) return Promise.reject(new Error("Canvas is not available"))
  return engine.detect(ctx.getImageData(0, 0, canvas.width, canvas.height))
}
