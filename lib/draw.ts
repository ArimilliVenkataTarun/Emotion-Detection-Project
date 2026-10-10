import { EMOTION_META, type FaceResult } from "./emotions"

export function drawResults(ctx: CanvasRenderingContext2D, faces: FaceResult[], scale = 1) {
  const size = Math.min(ctx.canvas.width, ctx.canvas.height)
  const lineWidth = Math.max(2, Math.round(size / 250))
  const fontSize = Math.max(12, Math.round(size / 34))
  ctx.font = `600 ${fontSize}px ui-sans-serif, system-ui, sans-serif`
  ctx.textBaseline = "middle"

  for (const face of faces) {
    const x = face.box.x * scale
    const y = face.box.y * scale
    const width = face.box.width * scale
    const height = face.box.height * scale
    const { color, label } = EMOTION_META[face.emotion]

    ctx.strokeStyle = color
    ctx.lineWidth = lineWidth
    ctx.strokeRect(x, y, width, height)

    const text = `${label} ${Math.round(face.confidence * 100)}%`
    const padding = fontSize * 0.45
    const labelHeight = fontSize * 1.6
    const labelWidth = ctx.measureText(text).width + padding * 2
    const labelX = x - lineWidth / 2
    const labelY = y - labelHeight >= 0 ? y - labelHeight : y + height

    ctx.fillStyle = color
    ctx.fillRect(labelX, labelY, labelWidth, labelHeight)
    ctx.fillStyle = "#0b0d10"
    ctx.fillText(text, labelX + padding, labelY + labelHeight / 2)
  }
}
