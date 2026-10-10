export const EMOTIONS = [
  "neutral",
  "happiness",
  "surprise",
  "sadness",
  "anger",
  "disgust",
  "fear",
  "contempt",
] as const

export type Emotion = (typeof EMOTIONS)[number]

export const EMOTION_META: Record<Emotion, { label: string; color: string }> = {
  neutral: { label: "Neutral", color: "#c8c8c8" },
  happiness: { label: "Happiness", color: "#ffd700" },
  surprise: { label: "Surprise", color: "#00bfff" },
  sadness: { label: "Sadness", color: "#4f8dff" },
  anger: { label: "Anger", color: "#ef4444" },
  disgust: { label: "Disgust", color: "#4cbb4c" },
  fear: { label: "Fear", color: "#b06ad8" },
  contempt: { label: "Contempt", color: "#c48a8a" },
}

export type FaceResult = {
  box: { x: number; y: number; width: number; height: number }
  emotion: Emotion
  confidence: number
  scores: Record<Emotion, number>
}
