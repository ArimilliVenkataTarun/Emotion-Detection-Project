import { EMOTIONS, EMOTION_META, type FaceResult } from "@/lib/emotions"

export function ResultsPanel({ results }: { results: FaceResult[] | null }) {
  return (
    <section aria-labelledby="results-heading" className="flex flex-col gap-4">
      <div className="flex items-baseline justify-between">
        <h2 id="results-heading" className="text-sm font-medium uppercase tracking-wider text-muted">
          Results
        </h2>
        {results && results.length > 0 && (
          <span className="font-mono text-xs text-muted">
            {results.length} {results.length === 1 ? "face" : "faces"}
          </span>
        )}
      </div>

      {results === null && (
        <p className="rounded-lg border border-dashed border-border p-6 text-sm leading-relaxed text-muted">
          Upload a photo, try the sample, or start your webcam to see emotion scores here.
        </p>
      )}

      {results?.length === 0 && (
        <p className="rounded-lg border border-dashed border-border p-6 text-sm leading-relaxed text-muted">
          No faces detected. Try a well-lit, front-facing photo.
        </p>
      )}

      <ul className="flex flex-col gap-3" aria-live="polite">
        {results?.map((face, index) => {
          const top = EMOTION_META[face.emotion]
          const ranked = [...EMOTIONS].sort((a, b) => face.scores[b] - face.scores[a])
          return (
            <li key={index} className="flex flex-col gap-4 rounded-lg border border-border bg-surface p-4">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span className="size-2.5 rounded-full" style={{ backgroundColor: top.color }} aria-hidden="true" />
                  <span className="font-medium">{top.label}</span>
                  <span className="text-xs text-muted">Face {index + 1}</span>
                </div>
                <span className="font-mono text-sm tabular-nums">{Math.round(face.confidence * 100)}%</span>
              </div>
              <dl className="flex flex-col gap-2">
                {ranked.map((emotion) => {
                  const value = face.scores[emotion]
                  return (
                    <div key={emotion} className="flex items-center gap-3 text-xs">
                      <dt className="w-20 shrink-0 text-muted">{EMOTION_META[emotion].label}</dt>
                      <dd className="flex flex-1 items-center gap-2">
                        <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                          <div
                            className="h-full rounded-full transition-[width] duration-200"
                            style={{ width: `${value * 100}%`, backgroundColor: EMOTION_META[emotion].color }}
                          />
                        </div>
                        <span className="w-9 text-right font-mono tabular-nums text-muted">
                          {(value * 100).toFixed(0)}%
                        </span>
                      </dd>
                    </div>
                  )
                })}
              </dl>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
