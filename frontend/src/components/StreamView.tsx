// @ts-nocheck
import { useEffect, useRef, useState, useCallback } from 'react'
import Plot from './Plot'
import { useAppStore } from '../store'
import { startStream, stopStream, injectAnomaly, openStreamWS } from '../api'
import type { StreamTick } from '../types'

const VAR_NAMES = ['x (var_0)', 'w (var_1)', 'y (var_2)', 'z (var_3)']
const COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
const WINDOW = 200

export default function StreamView() {
  const session = useAppStore((s) => s.session)
  const runStatus = useAppStore((s) => s.runStatus)
  const setToast = useAppStore((s) => s.setToast)

  const [buffer, setBuffer] = useState<number[][]>([])
  const [injectionGlobalTs, setInjectionGlobalTs] = useState<number[]>([])
  const [detectionGlobalTs, setDetectionGlobalTs] = useState<number[]>([])
  const [currentScores, setCurrentScores] = useState<number[]>([0, 0, 0, 0])
  const [currentDetected, setCurrentDetected] = useState<boolean[]>([false, false, false, false])
  const [maxScores, setMaxScores] = useState<number[]>([0, 0, 0, 0])
  const [varDetectionCounts, setVarDetectionCounts] = useState<number[]>([0, 0, 0, 0])
  const [isRunning, setIsRunning] = useState(false)
  const [injecting, setInjecting] = useState(false)
  const [latencyRecords, setLatencyRecords] = useState<Array<{
    injT: number
    detT: number | null
    ticks: number | null
    anomalyVars: number[] | null
    anomalyAmp: number | null
  }>>([])

  const totalTicksRef = useRef(0)
  const wsRef = useRef<WebSocket | null>(null)
  const pendingInjectionsRef = useRef<number[]>([])
  const lastDetectionTRef = useRef<number>(-99)

  useEffect(() => {
    return () => { wsRef.current?.close() }
  }, [])

  const openWS = useCallback(() => {
    if (!session) return
    wsRef.current?.close()
    const ws = openStreamWS(session.session_id)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      try {
        const msg: StreamTick = JSON.parse(ev.data)
        if (msg.type === 'ping' || msg.type === 'hello') return
        if (msg.type === 'stopped') { setIsRunning(false); return }
        if (msg.type === 'tick' && msg.values && msg.t !== undefined) {
          totalTicksRef.current = msg.t + 1
          setBuffer((prev) => {
            const next = [...prev, msg.values!]
            return next.length > WINDOW ? next.slice(-WINDOW) : next
          })
          if (msg.is_anomaly_step) {
            setInjectionGlobalTs((prev) => [...prev, msg.t!])
            pendingInjectionsRef.current.push(msg.t!)
            setLatencyRecords((prev) => [...prev, {
              injT: msg.t!, detT: null, ticks: null,
              anomalyVars: msg.anomaly_vars ?? null,
              anomalyAmp: msg.anomaly_amp ?? null,
            }])
          }
          const s = msg.scores ?? [0, 0, 0, 0]
          const d = msg.detected ?? [false, false, false, false]
          setCurrentScores(s)
          setCurrentDetected(d)
          setMaxScores((prev) => prev.map((m, i) => Math.max(m, s[i])))
          if (d.some(Boolean) && msg.t! - lastDetectionTRef.current > 2) {
            lastDetectionTRef.current = msg.t!
            setDetectionGlobalTs((prev) => [...prev, msg.t!])
            setVarDetectionCounts((prev) => prev.map((c, i) => c + (d[i] ? 1 : 0)))
            // 匹配最早的待处理注入
            if (pendingInjectionsRef.current.length > 0) {
              const injT = pendingInjectionsRef.current[0]
              if (msg.t! >= injT) {
                pendingInjectionsRef.current.shift()
                const ticks = msg.t! - injT
                setLatencyRecords((prev) => {
                  const idx = prev.findIndex((r) => r.injT === injT && r.detT === null)
                  if (idx === -1) return prev
                  const updated = [...prev]
                  updated[idx] = { ...updated[idx], detT: msg.t!, ticks }
                  return updated
                })
              }
            }
          }
        }
      } catch { /* ignore */ }
    }
    ws.onerror = () => setToast({ kind: 'error', text: '流式 WebSocket 连接出错' })
    ws.onclose = () => {}
  }, [session, setToast])

  function resetState() {
    setBuffer([])
    setInjectionGlobalTs([])
    setDetectionGlobalTs([])
    setCurrentScores([0, 0, 0, 0])
    setCurrentDetected([false, false, false, false])
    setMaxScores([0, 0, 0, 0])
    setVarDetectionCounts([0, 0, 0, 0])
    setLatencyRecords([])
    pendingInjectionsRef.current = []
    lastDetectionTRef.current = -99
    totalTicksRef.current = 0
  }

  async function handleStart() {
    if (!session) return
    try {
      resetState()
      await startStream(session.session_id)
      openWS()
      setIsRunning(true)
    } catch (e: any) {
      setToast({ kind: 'error', text: `启动失败：${e.message}` })
    }
  }

  async function handleStop() {
    if (!session) return
    try {
      await stopStream(session.session_id)
      setIsRunning(false)
      wsRef.current?.close()
    } catch (e: any) {
      setToast({ kind: 'error', text: `停止失败：${e.message}` })
    }
  }

  async function handleInject() {
    if (!session || !isRunning) return
    setInjecting(true)
    try {
      await injectAnomaly(session.session_id)
    } catch (e: any) {
      setToast({ kind: 'error', text: `注入失败：${e.message}` })
    } finally {
      setTimeout(() => setInjecting(false), 400)
    }
  }

  if (!session || session.dataset_name !== 'linear' || runStatus !== 'done') return null

  // ── chart helpers ──────────────────────────────────────────────────────────
  const xAxis = buffer.map((_, i) => i)
  const currentT = totalTicksRef.current
  const bufLen = buffer.length

  function globalToLocal(gt: number) {
    const offset = currentT - 1 - gt
    return bufLen - 1 - offset
  }

  const seriesTraces = VAR_NAMES.map((name, v) => ({
    x: xAxis,
    y: buffer.map((row) => row[v]),
    mode: 'lines',
    name,
    line: { color: COLORS[v], width: 1.5 },
  }))

  const injLocalXs = injectionGlobalTs.map(globalToLocal).filter((x) => x >= 0 && x < bufLen)
  const detLocalXs = detectionGlobalTs.map(globalToLocal).filter((x) => x >= 0 && x < bufLen)

  const injMarker = injLocalXs.length > 0 ? {
    x: injLocalXs,
    y: injLocalXs.map((xi) => { const row = buffer[xi]; return row ? Math.max(...row) : 0 }),
    mode: 'markers',
    name: '注入点 ▼',
    marker: { color: '#dc2626', size: 11, symbol: 'triangle-down' },
  } : null

  // 检测标记：x = 报警时间步，y 固定用垂直线（shape）表示，marker 仅做图例占位
  const detShapes = detLocalXs.map((x) => ({
    type: 'line',
    x0: x, x1: x,
    y0: 0, y1: 1,
    yref: 'paper',
    line: { color: '#f97316', width: 1.5, dash: 'dot' },
  }))

  const detMarker = detLocalXs.length > 0 ? {
    x: detLocalXs,
    y: detLocalXs.map(() => null), // 不在图上绘制实际点，只用于图例
    mode: 'markers',
    name: '检测到 |',
    marker: { color: '#f97316', size: 10, symbol: 'line-ns' },
  } : null

  const allTraces = [
    ...seriesTraces,
    ...(injMarker ? [injMarker] : []),
    ...(detMarker ? [detMarker] : []),
  ]

  // summary stats
  const totalInjected = injectionGlobalTs.length
  const totalDetected = detectionGlobalTs.length
  const detectionRate = totalInjected > 0
    ? Math.round((Math.min(totalDetected, totalInjected) / totalInjected) * 100)
    : 0

  return (
    <div className="glass-panel p-4 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="section-title mb-0 text-sm">
          🌊 实时流式异常检测
          <span className="ml-2 text-xs font-normal text-slate-500">（Linear 数据集）</span>
        </h3>
        <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-medium ${isRunning ? 'bg-emerald-100 text-emerald-700' : 'bg-slate-100 text-slate-500'}`}>
          <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'}`} />
          {isRunning ? '运行中' : '已停止'}
        </span>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-center gap-3">
        {!isRunning ? (
          <button onClick={handleStart} className="px-4 py-2 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 text-white text-sm font-semibold shadow hover:shadow-md transition">
            ▶ 开始流式生成
          </button>
        ) : (
          <button onClick={handleStop} className="px-4 py-2 rounded-lg bg-rose-500 text-white text-sm font-semibold shadow hover:bg-rose-600 transition">
            ⏹ 停止生成
          </button>
        )}
        <button
          disabled={!isRunning}
          onClick={handleInject}
          className={`px-4 py-2 rounded-lg text-sm font-semibold shadow transition ${isRunning ? injecting ? 'bg-orange-400 text-white scale-95' : 'bg-orange-500 hover:bg-orange-600 text-white' : 'bg-slate-200 text-slate-400 cursor-not-allowed'}`}
        >
          💥 插入一个异常值
        </button>
        {currentT > 0 && (
          <span className="text-xs text-slate-500 ml-auto">已生成 {currentT} 个数据点</span>
        )}
      </div>

      {/* Summary banner (appears after at least 1 injection) */}
      {totalInjected > 0 && (
        <div className="flex items-center gap-4 px-4 py-2.5 rounded-lg bg-slate-50 border border-slate-200 text-sm">
          <span className="text-slate-700">
            注入 <b className="text-red-600">{totalInjected}</b> 次
          </span>
          <span className="text-slate-300">|</span>
          <span className="text-slate-700">
            检测到 <b className="text-orange-600">{totalDetected}</b> 个异常 tick
          </span>
          <span className="text-slate-300">|</span>
          <span className="text-slate-700">
            检出率约 <b className={detectionRate >= 50 ? 'text-emerald-600' : 'text-rose-600'}>{detectionRate}%</b>
          </span>
          <span className="text-[11px] text-slate-400 ml-auto">
            🔴 红色倒三角 = 注入点 &nbsp; 🟠 橙色虚线 = 模型检测线
          </span>
        </div>
      )}

      {/* Latency table */}
      {latencyRecords.length > 0 && (
        <div className="rounded-lg border border-slate-200 overflow-hidden text-xs">
          <div className="bg-slate-50 px-3 py-1.5 font-medium text-slate-600 border-b border-slate-200">
            ⏱ 注入 → 检测延迟记录
          </div>
          <div className="divide-y divide-slate-100">
            {latencyRecords.map((r, i) => {
              const ms = r.ticks !== null ? r.ticks * 300 : null
              return (
                <div key={i} className="flex items-center gap-3 px-3 py-2">
                  <span className="text-slate-400 w-14 shrink-0">第 {i + 1} 次</span>
                  <span className="text-slate-500">
                    注入 t={r.injT}
                    {r.anomalyVars && (
                      <span className="ml-1.5 text-rose-500">
                        （{r.anomalyVars.map((v) => ['x','w','y','z'][v]).join('+')} +{r.anomalyAmp?.toFixed(1)}）
                      </span>
                    )}
                  </span>
                  <span className="text-slate-300">→</span>
                  {r.detT !== null ? (
                    <>
                      <span className="text-slate-500">检测 t={r.detT}</span>
                      <span className={`ml-auto font-semibold px-2 py-0.5 rounded ${
                        r.ticks === 0
                          ? 'bg-emerald-100 text-emerald-700'
                          : r.ticks! <= 2
                          ? 'bg-orange-100 text-orange-700'
                          : 'bg-rose-100 text-rose-700'
                      }`}>
                        {r.ticks === 0
                          ? '即时检测 (<300ms)'
                          : `延迟 ${r.ticks} tick · ${ms}ms`}
                      </span>
                    </>
                  ) : (
                    <span className="ml-auto text-slate-400 italic">
                      {isRunning ? '等待检测中…' : '未检出'}
                    </span>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Chart */}
      {buffer.length > 1 ? (
        <Plot
          data={allTraces}
          layout={{
            height: 300,
            margin: { t: 10, b: 50, l: 55, r: 20 },
            xaxis: { title: '时间步（滚动窗口最近 200 步）', gridcolor: '#e5e7eb' },
            yaxis: { title: '变量值', gridcolor: '#e5e7eb' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            legend: { orientation: 'h', y: -0.3 },
            shapes: detShapes,
          }}
          style={{ width: '100%' }}
          useResizeHandler
          config={{ responsive: true, displayModeBar: false }}
        />
      ) : (
        <div className="h-24 flex items-center justify-center text-slate-400 text-sm border border-dashed border-slate-200 rounded-lg">
          {isRunning ? '数据生成中，稍候…' : '点击「开始流式生成」启动'}
        </div>
      )}

      {/* Per-variable score cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {VAR_NAMES.map((name, v) => {
          const cur = currentScores[v] ?? 0
          const peak = maxScores[v] ?? 0
          const detCount = varDetectionCounts[v] ?? 0
          const isNowDetected = currentDetected[v] ?? false
          const everDetected = detCount > 0
          const pct = Math.min(100, Math.max(0, (peak / 6) * 100))
          return (
            <div key={v} className={`rounded-lg p-3 border transition ${isNowDetected ? 'border-red-400 bg-red-50' : everDetected ? 'border-orange-300 bg-orange-50/60' : 'border-slate-200 bg-white/60'}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-semibold" style={{ color: COLORS[v] }}>{name}</span>
                {isNowDetected
                  ? <span className="text-[10px] font-bold text-red-600 bg-red-100 px-1.5 py-0.5 rounded">⚠ 异常</span>
                  : everDetected
                  ? <span className="text-[10px] font-medium text-orange-600 bg-orange-100 px-1.5 py-0.5 rounded">历史 {detCount}次</span>
                  : null
                }
              </div>
              <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden mb-1">
                <div className="h-full rounded-full transition-all duration-300"
                  style={{
                    width: `${pct}%`,
                    backgroundColor: isNowDetected ? '#ef4444' : everDetected ? '#f97316' : '#10b981',
                  }}
                />
              </div>
              <div className="flex justify-between text-[10px] text-slate-500">
                <span>当前: {cur.toFixed(2)}</span>
                <span className="font-medium">峰值: {peak.toFixed(2)}</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
