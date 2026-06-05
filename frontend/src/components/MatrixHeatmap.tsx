type MatrixHeatmapProps = {
  z: number[][]
  xLabels?: string[]
  yLabels?: string[]
  xAxisTitle?: string
  yAxisTitle?: string
  height?: number
  domain?: [number, number]
  showValues?: boolean
  valueFormatter?: (value: number) => string
  compactLabels?: boolean
}

function finiteValues(z: number[][]) {
  return z.flat().filter((value) => Number.isFinite(value))
}

function defaultFormatter(value: number) {
  if (Math.abs(value) >= 10) return value.toFixed(0)
  if (Math.abs(value) >= 1) return value.toFixed(1)
  return value.toFixed(2)
}

function pickTicks(count: number) {
  if (count <= 12) return Array.from({ length: count }, (_, i) => i)
  const ticks = new Set<number>()
  for (let i = 0; i <= 5; i++) {
    ticks.add(Math.round(((count - 1) * i) / 5))
  }
  return Array.from(ticks).sort((a, b) => a - b)
}

function shortenLabel(label: string) {
  return label
    .replace(/\s*\((var|metric)_\d+\)\s*$/i, '')
    .replace(/\s*（(var|metric)_\d+）\s*$/i, '')
}

export default function MatrixHeatmap({
  z,
  xLabels,
  yLabels,
  xAxisTitle,
  yAxisTitle,
  height = 320,
  domain,
  showValues = false,
  valueFormatter = defaultFormatter,
  compactLabels = true,
}: MatrixHeatmapProps) {
  const rows = z.length
  const cols = Math.max(0, ...z.map((row) => row.length))
  const values = finiteValues(z)
  const min = domain?.[0] ?? Math.min(...values, 0)
  const max = domain?.[1] ?? Math.max(...values, 1)
  const span = max - min || 1
  const rawY = yLabels ?? Array.from({ length: rows }, (_, i) => `row_${i}`)
  const rawX = xLabels ?? Array.from({ length: cols }, (_, i) => `col_${i}`)
  const y = compactLabels ? rawY.map(shortenLabel) : rawY
  const x = compactLabels ? rawX.map(shortenLabel) : rawX
  const ticks = pickTicks(cols)
  const cellMin = cols > 80 ? 3 : cols > 32 ? 8 : 28
  const labelColumnWidth = rows <= 8 ? 132 : 104

  function colorFor(value: number | undefined) {
    if (!Number.isFinite(value)) return '#f8fafc'
    const t = Math.max(0, Math.min(1, ((value ?? 0) - min) / span))
    return `rgba(37, 99, 235, ${0.08 + t * 0.82})`
  }

  return (
    <div className="w-full">
      <div
        className="grid gap-x-2"
        style={{
          gridTemplateColumns: `${labelColumnWidth}px minmax(0, 1fr)`,
          gridTemplateRows: `${height}px auto`,
        }}
      >
        <div
          className="grid min-h-0 text-[11px] leading-tight text-slate-500"
          style={{ gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))` }}
        >
          {y.map((label, i) => (
            <div key={`${label}-${i}`} className="flex min-w-0 items-center justify-end pr-2">
              <span className="line-clamp-2 text-right" title={rawY[i]}>
                {label}
              </span>
            </div>
          ))}
        </div>

        <div className="min-w-0 overflow-hidden rounded-lg border border-slate-200 bg-white">
          <div
            className="grid h-full w-full"
            style={{
              gridTemplateColumns: `repeat(${cols}, minmax(${cellMin}px, 1fr))`,
              gridTemplateRows: `repeat(${rows}, minmax(0, 1fr))`,
            }}
          >
            {z.flatMap((row, r) =>
              Array.from({ length: cols }, (_, c) => {
                const value = row[c] ?? 0
                const normalized = Math.max(0, Math.min(1, (value - min) / span))
                return (
                  <div
                    key={`${r}-${c}`}
                    className="flex items-center justify-center border-b border-r border-white/70 text-[10px] font-semibold"
                    style={{
                      backgroundColor: colorFor(value),
                      color: normalized > 0.55 ? 'white' : '#334155',
                    }}
                    title={`${rawY[r] ?? r} <- ${rawX[c] ?? c}: ${valueFormatter(value)}`}
                  >
                    {showValues ? valueFormatter(value) : ''}
                  </div>
                )
              }),
            )}
          </div>
        </div>

        <div className="relative flex items-center justify-center pt-1 text-[11px] font-medium text-slate-500">
          {yAxisTitle}
        </div>
        <div className="relative min-h-14 text-[11px] leading-tight text-slate-500">
          {ticks.map((tick) => (
            <span
              key={tick}
              className="absolute top-1 max-w-24 -translate-x-1/2 text-center"
              style={{ left: cols <= 1 ? '0%' : `${(tick / (cols - 1)) * 100}%` }}
              title={rawX[tick]}
            >
              {x[tick]}
            </span>
          ))}
          {xAxisTitle && (
            <span className="absolute bottom-0 left-1/2 -translate-x-1/2 font-medium text-slate-600">
              {xAxisTitle}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
