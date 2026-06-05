// @ts-nocheck
import { useEffect, useRef, useState, useCallback } from 'react'
import Plot from './Plot'
import { useAppStore } from '../store'
import { startStream, stopStream, injectAnomaly, openStreamWS, listFaultScenarios } from '../api'
import type { FaultScenario, StreamTick } from '../types'
import { formatVariableList, getVariableNames } from '../domain/variableNames'
import { Activity, ChevronDown, ChevronUp, Loader2, Play, PlusCircle, Square } from 'lucide-react'

const COLORS = ['#2563eb', '#ea580c', '#16a34a', '#dc2626', '#7c3aed', '#0891b2', '#be123c', '#4d7c0f']
const STREAM_DATASET_LABELS: Record<string, string> = {
  linear: '线性工业过程',
  nonlinear: '非线性耦合过程',
  swat: 'SWaT 水处理过程',
}
const SUPPORTED_STREAM_DATASETS = new Set(['linear', 'nonlinear', 'swat'])
const WINDOW = 200

type LatencyRecord = {
  injT: number
  detT: number | null
  ticks: number | null
  anomalyVars: number[] | null
  anomalyAmp: number | null
  faultId: string | null
  faultName: string | null
  faultEffect: string | null
  faultRootCause: string | null
  faultConfidence: number | null
  inferenceMs: number | null
  detectionSource: string | null
}

type DetectionRecord = {
  detT: number
  detectedVars: number[]
  inferenceMs: number | null
  detectionSource: string | null
  matchedInjT: number | null
}

function fillArray<T>(count: number, value: T) {
  return Array.from({ length: Math.max(1, count) }, () => value)
}

function formatInferenceMs(value: number | null) {
  if (value === null || !Number.isFinite(value)) return '推理耗时 --'
  if (value < 0.1) return `推理 ${value.toFixed(3)}ms`
  if (value < 10) return `推理 ${value.toFixed(2)}ms`
  return `推理 ${value.toFixed(1)}ms`
}

function formatTickDelay(ticks: number | null) {
  if (ticks === null) return '等待检测'
  if (ticks === 0) return '同 tick 检出'
  return `延迟 ${ticks} tick · ${ticks * 300}ms`
}

function formatDetectionSource(source: string | null) {
  if (source === 'model+rule') return '模型+规则'
  if (source === 'rule') return '规则兜底'
  if (source === 'model') return '模型检测'
  return '检测来源 --'
}

export default function StreamView() {
  const session = useAppStore((s) => s.session)
  const runStatus = useAppStore((s) => s.runStatus)
  const setToast = useAppStore((s) => s.setToast)

  const [buffer, setBuffer] = useState<number[][]>([])
  const [injectionGlobalTs, setInjectionGlobalTs] = useState<number[]>([])
  const [detectionGlobalTs, setDetectionGlobalTs] = useState<number[]>([])
  const [currentScores, setCurrentScores] = useState<number[]>([])
  const [currentDetected, setCurrentDetected] = useState<boolean[]>([])
  const [maxScores, setMaxScores] = useState<number[]>([])
  const [varDetectionCounts, setVarDetectionCounts] = useState<number[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [isStopping, setIsStopping] = useState(false)
  const [injecting, setInjecting] = useState(false)
  const [latencyRecords, setLatencyRecords] = useState<LatencyRecord[]>([])
  const [detectionRecords, setDetectionRecords] = useState<DetectionRecord[]>([])
  const [faultScenarios, setFaultScenarios] = useState<FaultScenario[]>([])
  const [showExtraAlarms, setShowExtraAlarms] = useState(false)

  const totalTicksRef = useRef(0)
  const wsRef = useRef<WebSocket | null>(null)
  const pendingInjectionsRef = useRef<number[]>([])
  const lastDetectionTRef = useRef<number>(-99)

  const datasetName = session?.dataset_name ?? 'linear'
  const numVars = Math.max(1, Number(session?.num_vars ?? buffer[0]?.length ?? 4))
  const visibleVarCount = Math.min(numVars, 8)
  const visibleVarIndexes = Array.from({ length: visibleVarCount }, (_, i) => i)
  const sessionFaultId = String(session?.fault_id ?? session?.options_summary?.fault_id ?? '')
  const selectedFaultId = sessionFaultId || faultScenarios[0]?.id || ''
  const selectedFault = faultScenarios.find((f) => f.id === selectedFaultId)
  const varNames = getVariableNames(datasetName, numVars, selectedFaultId)
  const streamSupported = SUPPORTED_STREAM_DATASETS.has(datasetName)

  useEffect(() => {
    return () => { wsRef.current?.close() }
  }, [])

  useEffect(() => {
    wsRef.current?.close()
    setIsRunning(false)
    setIsStopping(false)
    resetState()
  }, [session?.session_id])

  useEffect(() => {
    if (!session || !streamSupported) {
      setFaultScenarios([])
      return
    }

    let cancelled = false
    listFaultScenarios(datasetName)
      .then((faults) => {
        if (cancelled) return
        setFaultScenarios(faults)
      })
      .catch((e: any) => {
        if (!cancelled) setToast({ kind: 'error', text: `加载故障场景失败：${e.message}` })
      })

    return () => { cancelled = true }
  }, [datasetName, session?.session_id, setToast, streamSupported])

  const openWS = useCallback(() => {
    if (!session) return
    wsRef.current?.close()
    const ws = openStreamWS(session.session_id)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      try {
        const msg: StreamTick = JSON.parse(ev.data)
        if (msg.type === 'ping' || msg.type === 'hello') return
        if (msg.type === 'stopped') {
          setIsRunning(false)
          setIsStopping(false)
          return
        }
        if (msg.type === 'tick' && msg.values && msg.t !== undefined) {
          totalTicksRef.current = msg.t + 1
          setBuffer((prev) => {
            const next = [...prev, msg.values!]
            return next.length > WINDOW ? next.slice(-WINDOW) : next
          })
          if (msg.is_anomaly_step) {
            const fault = msg.fault
            setInjectionGlobalTs((prev) => [...prev, msg.t!])
            pendingInjectionsRef.current.push(msg.t!)
            setLatencyRecords((prev) => [...prev, {
              injT: msg.t!, detT: null, ticks: null,
              anomalyVars: msg.anomaly_vars ?? null,
              anomalyAmp: msg.anomaly_amp ?? null,
              faultId: fault?.fault_id ?? fault?.id ?? null,
              faultName: fault?.fault_name ?? fault?.name ?? null,
              faultEffect: fault?.fault_effect ?? fault?.effect ?? null,
              faultRootCause: fault?.fault_root_cause ?? fault?.root_cause ?? null,
              faultConfidence: fault?.fault_confidence ?? fault?.confidence ?? null,
              inferenceMs: null,
              detectionSource: null,
            }])
          }

          const n = msg.values?.length ?? msg.scores?.length ?? numVars
          const scores = Array.from({ length: n }, (_, i) => Number(msg.scores?.[i] ?? 0))
          const detected = Array.from({ length: n }, (_, i) => Boolean(msg.detected?.[i]))
          setCurrentScores(scores)
          setCurrentDetected(detected)
          setMaxScores((prev) => Array.from({ length: n }, (_, i) => Math.max(prev[i] ?? 0, scores[i] ?? 0)))

          if (detected.some(Boolean) && msg.t! - lastDetectionTRef.current > 2) {
            lastDetectionTRef.current = msg.t!
            setDetectionGlobalTs((prev) => [...prev, msg.t!])
            setVarDetectionCounts((prev) => Array.from({ length: n }, (_, i) => (prev[i] ?? 0) + (detected[i] ? 1 : 0)))
            const inferenceMs = typeof msg.inference_ms === 'number' ? msg.inference_ms : null
            const detectionSource = msg.detection_source ?? null
            const detectedVars = detected
              .map((isDetected, i) => (isDetected ? i : -1))
              .filter((i) => i >= 0)
            let matchedInjT: number | null = null

            if (pendingInjectionsRef.current.length > 0) {
              const injT = pendingInjectionsRef.current[0]
              if (msg.t! >= injT) {
                pendingInjectionsRef.current.shift()
                matchedInjT = injT
                const ticks = msg.t! - injT
                setLatencyRecords((prev) => {
                  const idx = prev.findIndex((r) => r.injT === injT && r.detT === null)
                  if (idx === -1) return prev
                  const updated = [...prev]
                  updated[idx] = {
                    ...updated[idx],
                    detT: msg.t!,
                    ticks,
                    inferenceMs,
                    detectionSource,
                  }
                  return updated
                })
              }
            }
            setDetectionRecords((prev) => [...prev, {
              detT: msg.t!,
              detectedVars,
              inferenceMs,
              detectionSource,
              matchedInjT,
            }])
          }
        }
      } catch {
        // ignore malformed stream messages
      }
    }
    ws.onerror = () => setToast({ kind: 'error', text: '流式 WebSocket 连接出错' })
    ws.onclose = () => {}
  }, [numVars, session, setToast])

  function resetState() {
    setBuffer([])
    setInjectionGlobalTs([])
    setDetectionGlobalTs([])
    setCurrentScores(fillArray(numVars, 0))
    setCurrentDetected(fillArray(numVars, false))
    setMaxScores(fillArray(numVars, 0))
    setVarDetectionCounts(fillArray(numVars, 0))
    setLatencyRecords([])
    setDetectionRecords([])
    setShowExtraAlarms(false)
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
      setIsStopping(false)
    } catch (e: any) {
      setToast({ kind: 'error', text: `启动失败：${e.message}` })
    }
  }

  async function handleStop() {
    if (!session) return
    try {
      await stopStream(session.session_id)
      setIsStopping(true)
      setToast({ kind: 'info', text: '已请求停止生成，正在完成当前故障段检测...' })
    } catch (e: any) {
      setIsStopping(false)
      setToast({ kind: 'error', text: `停止失败：${e.message}` })
    }
  }

  async function handleInject() {
    if (!session || !isRunning || isStopping || !selectedFaultId) return
    setInjecting(true)
    try {
      await injectAnomaly(session.session_id, selectedFaultId)
    } catch (e: any) {
      setToast({ kind: 'error', text: `注入失败：${e.message}` })
    } finally {
      setTimeout(() => setInjecting(false), 400)
    }
  }

  if (!session || !streamSupported || runStatus !== 'done') return null

  const xAxis = buffer.map((_, i) => i)
  const currentT = totalTicksRef.current
  const bufLen = buffer.length

  function globalToLocal(gt: number) {
    const offset = currentT - 1 - gt
    return bufLen - 1 - offset
  }

  function formatVarList(vars: number[], faultId = selectedFaultId) {
    return formatVariableList(datasetName, vars, numVars, faultId)
  }

  const seriesTraces = visibleVarIndexes.map((v) => ({
    x: xAxis,
    y: buffer.map((row) => row[v] ?? null),
    mode: 'lines',
    name: varNames[v],
    line: { color: COLORS[v % COLORS.length], width: 1.5 },
  }))

  const injLocalXs = injectionGlobalTs.map(globalToLocal).filter((x) => x >= 0 && x < bufLen)
  const detLocalXs = detectionGlobalTs.map(globalToLocal).filter((x) => x >= 0 && x < bufLen)

  const injMarker = injLocalXs.length > 0 ? {
    x: injLocalXs,
    y: injLocalXs.map((xi) => {
      const row = buffer[xi]
      const values = row ? visibleVarIndexes.map((v) => row[v]).filter((v) => typeof v === 'number') : []
      return values.length > 0 ? Math.max(...values) : 0
    }),
    mode: 'markers',
    name: '注入点 ▼',
    marker: { color: '#dc2626', size: 11, symbol: 'triangle-down' },
  } : null

  const detShapes = detLocalXs.map((x) => ({
    type: 'line',
    x0: x, x1: x,
    y0: 0, y1: 1,
    yref: 'paper',
    line: { color: '#f97316', width: 1.5, dash: 'dot' },
  }))

  const detMarker = detLocalXs.length > 0 ? {
    x: detLocalXs,
    y: detLocalXs.map(() => null),
    mode: 'markers',
    name: '检测到 |',
    marker: { color: '#f97316', size: 10, symbol: 'line-ns' },
  } : null

  const allTraces = [
    ...seriesTraces,
    ...(injMarker ? [injMarker] : []),
    ...(detMarker ? [detMarker] : []),
  ]

  const totalInjected = injectionGlobalTs.length
  const totalDetected = detectionRecords.length
  const matchedDetected = latencyRecords.filter((r) => r.detT !== null).length
  const extraAlarmRecords = detectionRecords.filter((r) => r.matchedInjT === null)
  const extraAlarms = extraAlarmRecords.length
  const latestMatched = [...latencyRecords].reverse().find((r) => r.detT !== null)

  return (
    <div className="glass-panel p-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="section-title mb-0 text-sm">
          <Activity className="h-4 w-4 text-blue-600" />
          实时流式异常检测
          <span className="ml-2 text-xs font-normal text-slate-500">（{STREAM_DATASET_LABELS[datasetName] ?? datasetName}）</span>
        </h3>
        <span className={`status-badge ${isRunning ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-slate-50 text-slate-500'}`}>
          <span className={`w-2 h-2 rounded-full ${isRunning ? 'bg-emerald-500 animate-pulse' : 'bg-slate-400'}`} />
          {isStopping ? '收尾检测中' : isRunning ? '运行中' : '已停止'}
        </span>
      </div>

      <div className="grid gap-3 lg:grid-cols-[auto_auto_1fr] lg:items-end">
        {!isRunning ? (
          <button onClick={handleStart} className="btn-primary">
            <Play className="h-4 w-4" />
            开始流式生成
          </button>
        ) : (
          <button onClick={handleStop} disabled={isStopping} className="btn-danger">
            {isStopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
            {isStopping ? '检测收尾中' : '停止生成'}
          </button>
        )}
        <button
          disabled={!isRunning || isStopping || !selectedFaultId}
          onClick={handleInject}
          className={`${isRunning && !isStopping && selectedFaultId ? 'btn-warning' : 'btn-secondary'} ${injecting ? 'scale-[0.98]' : ''}`}
        >
          {injecting ? <Loader2 className="h-4 w-4 animate-spin" /> : <PlusCircle className="h-4 w-4" />}
          注入故障
        </button>
        {currentT > 0 && (
          <span className="pb-2 text-xs text-slate-500 lg:text-right">已生成 {currentT} 个数据点</span>
        )}
      </div>

      {selectedFault && (
        <div className="rounded-lg border border-blue-100 bg-blue-50/60 p-3 text-xs leading-5 text-slate-700">
          <div className="mb-1 flex flex-wrap items-center gap-2">
            <span className="font-semibold text-slate-900">{selectedFault.name}</span>
            <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold text-blue-700">{selectedFault.category}</span>
            <span className="rounded-full bg-white px-2 py-0.5 font-mono text-[10px] text-slate-500">{selectedFault.pattern}</span>
          </div>
          <p>信号表现：{selectedFault.effect}</p>
          <p>影响测点：{formatVarList(selectedFault.affected_vars)}</p>
          <p className="text-slate-500">预期根因：{selectedFault.root_cause}</p>
        </div>
      )}

      {totalInjected > 0 && (
        <div className="flex flex-wrap items-center gap-4 rounded-lg border border-slate-200 bg-slate-50 px-4 py-2.5 text-sm">
          <span className="text-slate-700">
            注入 <b className="text-red-600">{totalInjected}</b> 次
          </span>
          <span className="text-slate-300">|</span>
          <span className="text-slate-700">
            匹配检出 <b className="text-emerald-600">{matchedDetected}</b> 次
          </span>
          <span className="text-slate-300">|</span>
          <span className="text-slate-700">
            额外报警 <b className={extraAlarms > 0 ? 'text-orange-600' : 'text-slate-600'}>{extraAlarms}</b> 次
          </span>
          <span className="ml-auto text-[11px] text-slate-500">
            红色倒三角 = 注入点 · 橙色虚线 = 检测线
          </span>
        </div>
      )}

      {latestMatched && (
        <div className="grid gap-3 rounded-lg border border-emerald-200 bg-emerald-50/60 p-3 text-sm lg:grid-cols-[160px_1fr]">
          <div>
            <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-emerald-700">根因推断</p>
            <p className="mt-1 font-semibold text-slate-900">{latestMatched.faultName ?? '未知故障'}</p>
          </div>
          <div className="space-y-1 text-xs leading-5 text-slate-700">
            <p>第一步：模型在 t={latestMatched.detT} 检测到异常信号。</p>
            <p>第二步：结合故障场景库推断为：{latestMatched.faultRootCause ?? '暂无根因解释'}</p>
            {latestMatched.faultConfidence !== null && (
              <p className="text-slate-500">规则置信度：{Math.round(latestMatched.faultConfidence * 100)}%</p>
            )}
          </div>
        </div>
      )}

      {(latencyRecords.length > 0 || detectionRecords.length > 0) && (
        <div className="overflow-hidden rounded-lg border border-slate-200 text-xs">
          <div className="border-b border-slate-200 bg-slate-50 px-3 py-2 font-semibold text-slate-700">
            注入匹配与模型报警记录
          </div>
          <div className="divide-y divide-slate-100">
            {latencyRecords.map((r, i) => {
              return (
                <div key={i} className="flex flex-wrap items-center gap-3 px-3 py-2">
                  <span className="text-slate-400 w-14 shrink-0">第 {i + 1} 次</span>
                  <span className="min-w-0 text-slate-500">
                    <span className="font-semibold text-slate-700">{r.faultName ?? '故障注入'}</span>
                    <span className="ml-1">t={r.injT}</span>
                    {r.anomalyVars && (
                      <span className="ml-1.5 text-rose-500">
                        （影响 {formatVarList(r.anomalyVars, r.faultId ?? selectedFaultId)}）
                      </span>
                    )}
                    {r.faultEffect && (
                      <span className="ml-2 hidden text-slate-400 md:inline">{r.faultEffect}</span>
                    )}
                  </span>
                  <span className="text-slate-300">→</span>
                  {r.detT !== null ? (
                    <>
                      <span className="text-slate-500">检测 t={r.detT}</span>
                      <span className={`ml-auto shrink-0 rounded-full px-2 py-0.5 font-semibold ${
                        r.ticks === 0
                          ? 'bg-emerald-100 text-emerald-700'
                          : r.ticks! <= 2
                          ? 'bg-orange-100 text-orange-700'
                          : 'bg-rose-100 text-rose-700'
                      }`}>
                        {formatDetectionSource(r.detectionSource)} · {formatInferenceMs(r.inferenceMs)} · {formatTickDelay(r.ticks)}
                      </span>
                    </>
                  ) : (
                    <span className="ml-auto text-slate-400 italic">
                      {isRunning ? '等待匹配中...' : '该次注入未匹配报警'}
                    </span>
                  )}
                </div>
              )
            })}
            {extraAlarmRecords.length > 0 && (
              <div className="bg-orange-50/60 px-3 py-2">
                <button
                  type="button"
                  onClick={() => setShowExtraAlarms((v) => !v)}
                  className="flex w-full items-center justify-between gap-3 text-left text-[11px] font-semibold text-orange-700"
                >
                  <span>
                    额外报警 {extraAlarmRecords.length} 次
                    <span className="ml-2 font-normal text-orange-600">
                      默认收起：模型检测到异常，但没有匹配到本次故障注入
                    </span>
                  </span>
                  <span className="inline-flex items-center gap-1 rounded-full bg-white px-2 py-0.5 text-orange-700">
                    {showExtraAlarms ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                    {showExtraAlarms ? '收起' : '展开'}
                  </span>
                </button>
              </div>
            )}
            {showExtraAlarms && extraAlarmRecords.map((r, i) => (
              <div key={`extra-${r.detT}-${i}`} className="flex flex-wrap items-center gap-3 px-3 py-2">
                <span className="text-orange-500 w-20 shrink-0">额外报警 {i + 1}</span>
                <span className="font-semibold text-slate-700">检测 t={r.detT}</span>
                {r.detectedVars.length > 0 && (
                  <span className="text-orange-600">
                    （检测变量 {formatVarList(r.detectedVars)}）
                  </span>
                )}
                <span className="ml-auto shrink-0 rounded-full bg-orange-100 px-2 py-0.5 font-semibold text-orange-700">
                  {formatDetectionSource(r.detectionSource)} · {formatInferenceMs(r.inferenceMs)} · 未匹配注入
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {numVars > visibleVarCount && (
        <p className="text-[11px] text-slate-500">
          当前数据集共有 {numVars} 个变量，实时趋势图展示前 {visibleVarCount} 个变量；检测与根因记录仍按全量变量计算。
        </p>
      )}

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
          {isRunning ? '数据生成中，稍候...' : '点击「开始流式生成」启动'}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {visibleVarIndexes.map((v) => {
          const name = varNames[v]
          const cur = currentScores[v] ?? 0
          const peak = maxScores[v] ?? 0
          const detCount = varDetectionCounts[v] ?? 0
          const isNowDetected = currentDetected[v] ?? false
          const everDetected = detCount > 0
          const pct = Math.min(100, Math.max(0, (peak / 6) * 100))
          return (
            <div key={v} className={`rounded-lg border p-3 transition ${isNowDetected ? 'border-red-300 bg-red-50' : everDetected ? 'border-amber-300 bg-amber-50/70' : 'border-slate-200 bg-white'}`}>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-semibold" style={{ color: COLORS[v % COLORS.length] }}>{name}</span>
                {isNowDetected
                  ? <span className="rounded-full bg-red-100 px-1.5 py-0.5 text-[10px] font-bold text-red-600">异常</span>
                  : everDetected
                  ? <span className="rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-700">历史 {detCount}次</span>
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
