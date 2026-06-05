import { useEffect, useMemo, useRef, useState } from 'react'
import Sidebar from './components/Sidebar'
import DataView from './components/DataView'
import ResultsView from './components/ResultsView'
import StreamView from './components/StreamView'
import Plot from './components/Plot'
import { useAppStore } from './store'
import { getResults, openProgressWS, runModel, stopTraining } from './api'
import { Activity, AlertCircle, CheckCircle2, Loader2, Play, RadioTower, Square } from 'lucide-react'

export default function App() {
  const toast = useAppStore((s) => s.toast)
  const setToast = useAppStore((s) => s.setToast)
  const session = useAppStore((s) => s.session)
  const runStatus = useAppStore((s) => s.runStatus)
  const setRunStatus = useAppStore((s) => s.setRunStatus)
  const runError = useAppStore((s) => s.runError)
  const setRunError = useAppStore((s) => s.setRunError)
  const progress = useAppStore((s) => s.progressLog)
  const appendProgress = useAppStore((s) => s.appendProgress)
  const resetProgress = useAppStore((s) => s.resetProgress)
  const setResults = useAppStore((s) => s.setResults)

  const [epochs, setEpochs] = useState(50)
  const [lr, setLr] = useState(0.001)
  const [isStopping, setIsStopping] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  // 自动消失 toast
  useEffect(() => {
    if (!toast) return
    const t = setTimeout(() => setToast(null), 4500)
    return () => clearTimeout(t)
  }, [toast, setToast])

  // 关闭已有 WS
  useEffect(() => {
    return () => {
      wsRef.current?.close()
    }
  }, [])

  function ensureWebSocket() {
    if (!session) return
    wsRef.current?.close()
    const ws = openProgressWS(session.session_id)
    wsRef.current = ws
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (msg.phase === 'ping') return
        if (msg.phase === 'hello') {
          // The socket can connect just before /run flips the backend status to running.
          // Ignore that initial idle echo so it does not downgrade the local running state.
          if (msg.run_status && msg.run_status !== 'idle') setRunStatus(msg.run_status)
          return
        }
        if (msg.phase === 'training' || msg.phase === 'training_batch') {
          setRunStatus('running')
        }
        appendProgress(msg)
        if (msg.phase === 'done') {
          setIsStopping(false)
          setRunStatus('done')
          getResults(session.session_id)
            .then((r) => setResults(r))
            .catch((e) => setToast({ kind: 'error', text: `获取结果失败：${e.message}` }))
        }
        if (msg.phase === 'stopped') {
          setIsStopping(false)
          setRunStatus('stopped')
          setRunError(null)
          setToast({ kind: 'info', text: msg.message ?? '训练已停止' })
        }
        if (msg.phase === 'error') {
          setIsStopping(false)
          setRunStatus('failed')
          setRunError(msg.message ?? 'Unknown error')
        }
      } catch (e) {
        // ignore
      }
    }
    ws.onclose = () => {}
    ws.onerror = () => {
      setToast({ kind: 'error', text: 'WebSocket 连接错误，将退化为轮询' })
    }
  }

  async function handleRun() {
    if (!session) return
    setRunError(null)
    setIsStopping(false)
    resetProgress()
    setRunStatus('running')
    ensureWebSocket()
    try {
      await runModel(session.session_id, { epochs, lr, training_aerca: true })
      setToast({ kind: 'info', text: '模型训练已启动，等待 WebSocket 推送进度…' })
    } catch (e: any) {
      setRunStatus('failed')
      setRunError(e.message)
      setToast({ kind: 'error', text: `启动失败：${e.message}` })
    }
  }

  async function handleStopTraining() {
    if (!session || runStatus !== 'running') return
    setIsStopping(true)
    try {
      await stopTraining(session.session_id)
      setToast({ kind: 'info', text: '正在停止训练，当前计算批次结束后生效...' })
    } catch (e: any) {
      setIsStopping(false)
      setToast({ kind: 'error', text: `停止失败：${e.message}` })
    }
  }

  const lossData = useMemo(() => {
    const trainX: number[] = []
    const train: number[] = []
    const valX: number[] = []
    const val: number[] = []
    for (const p of progress) {
      if ((p.phase === 'training' || p.phase === 'training_batch') && typeof p.epoch === 'number') {
        const totalBatches = Math.max(1, p.total_batches ?? 1)
        const batch = p.phase === 'training_batch' ? Math.max(1, p.batch ?? 1) : totalBatches
        const x = p.epoch - 1 + batch / totalBatches
        if (typeof p.train_loss === 'number') {
          trainX.push(x)
          train.push(p.train_loss)
        }
        if (p.phase === 'training' && typeof p.val_loss === 'number') {
          valX.push(p.epoch)
          val.push(p.val_loss)
        }
      }
    }
    return { trainX, train, valX, val }
  }, [progress])

  const lastTraining = [...progress].reverse().find((p) => p.phase === 'training' || p.phase === 'training_batch')
  const trainingProgressPct = (() => {
    if (!lastTraining?.total_epochs || !lastTraining.epoch) return 0
    const totalBatches = Math.max(1, lastTraining.total_batches ?? 1)
    const batchProgress =
      lastTraining.phase === 'training_batch'
        ? Math.min(totalBatches, Math.max(1, lastTraining.batch ?? 1)) / totalBatches
        : 1
    return Math.round(((lastTraining.epoch - 1 + batchProgress) * 100) / lastTraining.total_epochs)
  })()
  const showLossPanel = runStatus === 'running' || lossData.trainX.length > 0 || Boolean(runError)

  useEffect(() => {
    if (lastTraining && runStatus === 'idle') {
      setRunStatus('running')
    }
  }, [lastTraining, runStatus, setRunStatus])

  return (
    <div className="app-shell">
      <div className="flex min-h-screen flex-col gap-3 p-3 lg:h-screen lg:flex-row">
        <Sidebar />

        <main className="min-w-0 flex-1 space-y-3 overflow-y-auto pr-1">
          <header className="glass-panel p-4">
            <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
              <div className="min-w-0">
                <p className="section-kicker">AERCA Analytics Workspace</p>
                <h1 className="mt-1 text-xl font-semibold text-slate-950">
                  多变量时间序列根因分析
                </h1>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`status-badge ${
                    runStatus === 'idle'
                      ? 'border-slate-200 bg-slate-50 text-slate-600'
                      : runStatus === 'running'
                      ? 'border-amber-200 bg-amber-50 text-amber-700'
                      : runStatus === 'done'
                      ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                      : runStatus === 'stopped'
                      ? 'border-slate-200 bg-slate-50 text-slate-600'
                      : 'border-rose-200 bg-rose-50 text-rose-700'
                  }`}
                >
                  {runStatus === 'running' ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : runStatus === 'done' ? <CheckCircle2 className="h-3.5 w-3.5" /> : runStatus === 'failed' ? <AlertCircle className="h-3.5 w-3.5" /> : <Activity className="h-3.5 w-3.5" />}
                  {runStatus.toUpperCase()}
                </span>
                {session && (
                  <span className="status-badge border-blue-200 bg-blue-50 text-blue-700">
                    当前会话 {session.dataset_name.toUpperCase()} · {session.num_vars} 变量 · T={session.T}
                  </span>
                )}
              </div>
            </div>

            {runStatus === 'running' && (
              <div className="mt-4 h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
                <div
                  className="h-full bg-blue-600 transition-all"
                  style={{ width: `${trainingProgressPct}%` }}
                />
              </div>
            )}
          </header>

          <DataView />

          {session && (
            <div className="glass-panel p-4">
              <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
                <div>
                  <h2 className="section-title mb-0 text-sm">
                    <RadioTower className="h-4 w-4 text-blue-600" />
                    训练控制
                  </h2>
                  <p className="mt-1 text-xs text-slate-500">
                    当前训练数据：{session.dataset_name} · {session.num_vars} 变量 · T={session.T} · session={session.session_id.slice(0, 6)}
                  </p>
                </div>
                {runStatus === 'running' && (
                  <span className="text-xs font-medium text-amber-700">实时同步中</span>
                )}
              </div>
              <div className="flex flex-wrap items-end gap-3">
                <label className="w-28">
                  <span className="field-label">Epochs</span>
                  <input
                    type="number"
                    min={1}
                    max={5000}
                    step={1}
                    className="field-control"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value || '1', 10))}
                  />
                </label>
                <label className="w-28">
                  <span className="field-label">LR</span>
                  <input
                    type="number"
                    step={0.0001}
                    className="field-control"
                    value={lr}
                    onChange={(e) => setLr(parseFloat(e.target.value || '0.001'))}
                  />
                </label>
                <button
                  disabled={runStatus === 'running'}
                  onClick={handleRun}
                  className="btn-primary"
                >
                  {runStatus === 'running' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                  {runStatus === 'running' ? '训练中' : '启动训练'}
                </button>
                {runStatus === 'running' && (
                  <button
                    type="button"
                    disabled={isStopping}
                    onClick={handleStopTraining}
                    className="btn-danger"
                  >
                    {isStopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
                    {isStopping ? '停止中' : '停止训练'}
                  </button>
                )}
              </div>
            </div>
          )}

          {showLossPanel && (
            <div className="glass-panel p-4 space-y-3">
              {session && (
                <div className="flex items-center gap-3 text-xs">
                  <span
                    className={`status-badge ${
                      runStatus === 'idle'
                        ? 'border-slate-200 bg-slate-50 text-slate-600'
                        : runStatus === 'running'
                        ? 'border-amber-200 bg-amber-50 text-amber-700'
                        : runStatus === 'done'
                        ? 'border-emerald-200 bg-emerald-50 text-emerald-700'
                        : runStatus === 'stopped'
                        ? 'border-slate-200 bg-slate-50 text-slate-600'
                        : 'border-rose-200 bg-rose-50 text-rose-700'
                    }`}
                  >
                    {runStatus.toUpperCase()}
                  </span>
                  <span className="status-badge border-blue-200 bg-blue-50 text-blue-700">
                    当前会话 {session.dataset_name.toUpperCase()} · {session.num_vars} 变量
                  </span>
                  {lastTraining && (
                    <span className="text-slate-600">
                      Epoch {lastTraining.epoch}/{lastTraining.total_epochs} ·
                      {lastTraining.phase === 'training_batch' && lastTraining.batch && lastTraining.total_batches
                        ? ` batch=${lastTraining.batch}/${lastTraining.total_batches} ·`
                        : ''}
                      train={lastTraining.train_loss?.toFixed(4)}
                      {typeof lastTraining.val_loss === 'number' ? ` · val=${lastTraining.val_loss.toFixed(4)}` : ''}
                    </span>
                  )}
                  {runError && (
                    <span className="text-rose-600">错误：{runError}</span>
                  )}
                </div>
              )}
              <p className="text-[11px] text-slate-500">
                这里显示的是 AERCA 训练目标值，不是原始时序数据；目标包含分布/正则项，数值可能为负，主要观察是否收敛。
              </p>

              {lossData.trainX.length > 0 ? (
                <Plot
                  data={[
                    {
                      x: lossData.trainX,
                      y: lossData.train,
                      mode: 'lines+markers',
                      name: 'train_loss',
                      line: { color: '#3b82f6' },
                    },
                    {
                      x: lossData.valX,
                      y: lossData.val,
                      mode: 'lines+markers',
                      name: 'val_loss',
                      line: { color: '#10b981' },
                    },
                  ]}
                  layout={{
                    height: 260,
                    margin: { t: 20, b: 40, l: 50, r: 20 },
                    legend: { orientation: 'h' },
                    xaxis: { title: 'Epoch' },
                    yaxis: { title: '训练目标值' },
                  }}
                  style={{ width: '100%' }}
                  useResizeHandler
                  config={{ responsive: true, displayModeBar: false }}
                />
              ) : (
                <div className="flex h-40 items-center justify-center rounded-lg border border-dashed border-slate-200 text-sm text-slate-500">
                  正在等待第一批训练 loss...
                </div>
              )}
            </div>
          )}

          <ResultsView />

          <StreamView />
        </main>
      </div>

      {toast && (
        <div
          className={`fixed bottom-6 right-6 max-w-sm rounded-lg px-4 py-3 text-sm font-medium text-white shadow-lg
            ${toast.kind === 'success' ? 'bg-emerald-700' : toast.kind === 'error' ? 'bg-rose-700' : 'bg-slate-900'}`}
        >
          {toast.text}
        </div>
      )}
    </div>
  )
}
