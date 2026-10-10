/* eslint-disable no-restricted-globals -- runs as a classic Web Worker */
"use strict"

const ASSETS = {
  script: "/opencv/opencv.js",
  cascade: "/opencv/haarcascade_frontalface_default.xml",
  model: "/opencv/emotion-ferplus-8.onnx",
}
const CASCADE_FILE = "haarcascade_frontalface_default.xml"
const MODEL_FILE = "emotion-ferplus-8.onnx"
const EMOTIONS = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
const PROGRESS_INTERVAL_MS = 100

let cv = null
let cascade = null
let net = null
let initPromise = null

const post = (message) => self.postMessage(message)

async function downloadAll() {
  const entries = Object.entries(ASSETS)
  const responses = await Promise.all(
    entries.map(async ([, url]) => {
      const response = await fetch(url)
      if (!response.ok) throw new Error(`Failed to download ${url} (HTTP ${response.status})`)
      return response
    }),
  )

  const sizes = responses.map((response) => Number(response.headers.get("content-length")) || 0)
  const totalKnown = sizes.every((size) => size > 0)
  const total = sizes.reduce((sum, size) => sum + size, 0)
  let received = 0
  let lastReport = 0

  const reportProgress = (force) => {
    const now = Date.now()
    if (!force && now - lastReport < PROGRESS_INTERVAL_MS) return
    lastReport = now
    const megabytes = (received / 1048576).toFixed(1)
    post(
      totalKnown
        ? { type: "status", stage: "Downloading OpenCV.js and emotion model", progress: Math.min(1, received / total) }
        : { type: "status", stage: `Downloading OpenCV.js and emotion model (${megabytes} MB)` },
    )
  }

  reportProgress(true)
  const buffers = await Promise.all(
    responses.map(async (response) => {
      if (!response.body) {
        const buffer = new Uint8Array(await response.arrayBuffer())
        received += buffer.length
        reportProgress(false)
        return buffer
      }
      const reader = response.body.getReader()
      const chunks = []
      let length = 0
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        chunks.push(value)
        length += value.length
        received += value.length
        reportProgress(false)
      }
      const buffer = new Uint8Array(length)
      let offset = 0
      for (const chunk of chunks) {
        buffer.set(chunk, offset)
        offset += chunk.length
      }
      return buffer
    }),
  )
  reportProgress(true)

  return Object.fromEntries(entries.map(([key], index) => [key, buffers[index]]))
}

// The Emscripten Module exposes a custom `then` that calls back with the Module itself.
// Awaiting or returning it from an async function makes promise resolution recurse forever,
// so it is removed and readiness is detected via onRuntimeInitialized instead.
function waitForRuntime(module) {
  delete module.then
  return new Promise((resolve, reject) => {
    if (module.calledRun || module.Mat) {
      resolve()
      return
    }
    module.onAbort = (reason) => reject(new Error(`OpenCV.js aborted: ${reason}`))
    module.onRuntimeInitialized = () => resolve()
  })
}

async function initialize() {
  const assets = await downloadAll()

  post({ type: "status", stage: "Compiling OpenCV runtime" })
  const scriptUrl = URL.createObjectURL(new Blob([assets.script], { type: "text/javascript" }))
  try {
    importScripts(scriptUrl)
  } finally {
    URL.revokeObjectURL(scriptUrl)
  }
  const module = self.cv
  if (!module) throw new Error("OpenCV.js did not register a runtime")
  await waitForRuntime(module)
  cv = module

  post({ type: "status", stage: "Initializing face detector and emotion network" })
  cv.FS_createDataFile("/", CASCADE_FILE, assets.cascade, true, false, false)
  cv.FS_createDataFile("/", MODEL_FILE, assets.model, true, false, false)
  cascade = new cv.CascadeClassifier()
  if (!cascade.load(CASCADE_FILE)) throw new Error("Could not load the face detector")
  net = cv.readNet(MODEL_FILE)
  if (net.empty && net.empty()) throw new Error("Could not load the emotion model")
}

function softmax(logits) {
  const max = Math.max(...logits)
  const exps = logits.map((value) => Math.exp(value - max))
  const sum = exps.reduce((total, value) => total + value, 0)
  return exps.map((value) => value / sum)
}

function detect({ width, height, data }) {
  const src = cv.matFromImageData({ width, height, data })
  const gray = new cv.Mat()
  const faces = new cv.RectVector()
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY)
    const minSide = Math.max(30, Math.floor(Math.min(gray.rows, gray.cols) / 12))
    cascade.detectMultiScale(gray, faces, 1.1, 5, 0, new cv.Size(minSide, minSide), new cv.Size(0, 0))

    const results = []
    for (let i = 0; i < faces.size(); i++) {
      const face = faces.get(i)
      const pad = Math.floor(face.width * 0.1)
      const x0 = Math.max(0, face.x - pad)
      const y0 = Math.max(0, face.y - pad)
      const x1 = Math.min(gray.cols, face.x + face.width + pad)
      const y1 = Math.min(gray.rows, face.y + face.height + pad)

      const roi = gray.roi(new cv.Rect(x0, y0, x1 - x0, y1 - y0))
      const resized = new cv.Mat()
      let blob = null
      let output = null
      try {
        cv.resize(roi, resized, new cv.Size(64, 64))
        blob = cv.blobFromImage(resized, 1, new cv.Size(64, 64), new cv.Scalar(0, 0, 0, 0), false, false)
        net.setInput(blob)
        output = net.forward()
        const probabilities = softmax(Array.from(output.data32F))
        const topIndex = probabilities.indexOf(Math.max(...probabilities))
        results.push({
          box: { x: face.x, y: face.y, width: face.width, height: face.height },
          emotion: EMOTIONS[topIndex],
          confidence: probabilities[topIndex],
          scores: Object.fromEntries(EMOTIONS.map((emotion, index) => [emotion, probabilities[index]])),
        })
      } finally {
        roi.delete()
        resized.delete()
        if (blob) blob.delete()
        if (output) output.delete()
      }
    }
    return results.sort((a, b) => a.box.x - b.box.x)
  } finally {
    src.delete()
    gray.delete()
    faces.delete()
  }
}

self.onmessage = (event) => {
  const message = event.data
  if (message.type === "init") {
    initPromise ??= initialize()
    initPromise.then(
      () => post({ type: "ready" }),
      (error) => post({ type: "error", message: error && error.message ? error.message : String(error) }),
    )
    return
  }
  if (message.type === "detect") {
    if (!cv || !cascade || !net) {
      post({ type: "detectError", id: message.id, message: "Detector is not ready" })
      return
    }
    try {
      post({ type: "result", id: message.id, faces: detect(message) })
    } catch (error) {
      post({ type: "detectError", id: message.id, message: error && error.message ? error.message : String(error) })
    }
  }
}
