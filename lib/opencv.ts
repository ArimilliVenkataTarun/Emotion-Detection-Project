import { EMOTIONS, type Emotion, type FaceResult } from "./emotions"

/* eslint-disable @typescript-eslint/no-explicit-any -- OpenCV.js ships without type definitions */
declare global {
  interface Window {
    cv?: any
  }
}

export type Engine = { cv: any; cascade: any; net: any }
export type EngineStatus = { stage: string; progress?: number }

const ASSET_BASE = "/opencv"
const CASCADE_FILE = "haarcascade_frontalface_default.xml"
const MODEL_FILE = "emotion-ferplus-8.onnx"

let enginePromise: Promise<Engine> | null = null
let statusListener: ((status: EngineStatus) => void) | null = null

const report = (status: EngineStatus) => statusListener?.(status)

function loadScript(src: string) {
  return new Promise<void>((resolve, reject) => {
    if (document.querySelector("script[data-opencv]")) {
      resolve()
      return
    }
    const script = document.createElement("script")
    script.src = src
    script.async = true
    script.dataset.opencv = "true"
    script.onload = () => resolve()
    script.onerror = () => reject(new Error("Failed to load OpenCV.js"))
    document.body.appendChild(script)
  })
}

async function waitForRuntime() {
  for (let attempt = 0; attempt < 900; attempt++) {
    const cv = window.cv
    if (cv instanceof Promise) {
      window.cv = await cv
      continue
    }
    if (cv?.Mat && cv.FS_createDataFile && cv.readNet) return cv
    await new Promise((resolve) => setTimeout(resolve, 100))
  }
  throw new Error("OpenCV.js took too long to initialize")
}

async function fetchBytes(url: string, onProgress?: (fraction: number) => void) {
  const response = await fetch(url)
  if (!response.ok) throw new Error(`Failed to download ${url}`)
  const total = Number(response.headers.get("content-length")) || 0
  if (!response.body || !total || !onProgress) return new Uint8Array(await response.arrayBuffer())

  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let received = 0
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    received += value.length
    onProgress(Math.min(1, received / total))
  }
  const bytes = new Uint8Array(received)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.length
  }
  return bytes
}

export function loadEngine(onStatus: (status: EngineStatus) => void) {
  statusListener = onStatus
  if (!enginePromise) {
    enginePromise = (async () => {
      report({ stage: "Loading OpenCV.js and emotion model", progress: 0 })
      const [cv, cascadeBytes, modelBytes] = await Promise.all([
        loadScript(`${ASSET_BASE}/opencv.js`).then(waitForRuntime),
        fetchBytes(`${ASSET_BASE}/${CASCADE_FILE}`),
        fetchBytes(`${ASSET_BASE}/${MODEL_FILE}`, (progress) =>
          report({ stage: "Downloading emotion model", progress }),
        ),
      ])

      report({ stage: "Initializing networks" })
      cv.FS_createDataFile("/", CASCADE_FILE, cascadeBytes, true, false, false)
      cv.FS_createDataFile("/", MODEL_FILE, modelBytes, true, false, false)

      const cascade = new cv.CascadeClassifier()
      if (!cascade.load(CASCADE_FILE)) throw new Error("Could not load the face detector")
      const net = cv.readNet(MODEL_FILE)
      return { cv, cascade, net }
    })().catch((error) => {
      enginePromise = null
      throw error
    })
  }
  return enginePromise
}

function softmax(logits: number[]) {
  const max = Math.max(...logits)
  const exps = logits.map((value) => Math.exp(value - max))
  const sum = exps.reduce((total, value) => total + value, 0)
  return exps.map((value) => value / sum)
}

export function detectEmotions({ cv, cascade, net }: Engine, canvas: HTMLCanvasElement): FaceResult[] {
  const src = cv.imread(canvas)
  const gray = new cv.Mat()
  const faces = new cv.RectVector()
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY)
    const minSide = Math.max(30, Math.floor(Math.min(gray.rows, gray.cols) / 12))
    cascade.detectMultiScale(gray, faces, 1.1, 5, 0, new cv.Size(minSide, minSide), new cv.Size(0, 0))

    const results: FaceResult[] = []
    for (let i = 0; i < faces.size(); i++) {
      const face = faces.get(i)
      const pad = Math.floor(face.width * 0.1)
      const x0 = Math.max(0, face.x - pad)
      const y0 = Math.max(0, face.y - pad)
      const x1 = Math.min(gray.cols, face.x + face.width + pad)
      const y1 = Math.min(gray.rows, face.y + face.height + pad)

      const roi = gray.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0))
      const resized = new cv.Mat()
      cv.resize(roi, resized, new cv.Size(64, 64))
      const blob = cv.blobFromImage(resized, 1, new cv.Size(64, 64), new cv.Scalar(0, 0, 0, 0), false, false)
      net.setInput(blob)
      const output = net.forward()
      const probabilities = softmax(Array.from(output.data32F as Float32Array))
      roi.delete()
      resized.delete()
      blob.delete()
      output.delete()

      const scores = Object.fromEntries(EMOTIONS.map((emotion, index) => [emotion, probabilities[index]])) as Record<
        Emotion,
        number
      >
      const topIndex = probabilities.indexOf(Math.max(...probabilities))
      results.push({
        box: { x: face.x, y: face.y, width: face.width, height: face.height },
        emotion: EMOTIONS[topIndex],
        confidence: probabilities[topIndex],
        scores,
      })
    }
    return results.sort((a, b) => a.box.x - b.box.x)
  } finally {
    src.delete()
    gray.delete()
    faces.delete()
  }
}
