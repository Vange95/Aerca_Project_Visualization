import { useEffect, useMemo, useRef, useState } from 'react'
import Sidebar from './components/Sidebar'
import DataView from './components/DataView'
import ResultsView from './components/ResultsView'
import StreamView from './components/StreamView'
import Plot from './components/Plot'
import { useAppStore } from './store'
import { getResults, openProgressWS, runModel } from './api'

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
          if (msg.run_status) setRunStatus(msg.run_status)
          return
        }
        appendProgress(msg)
        if (msg.phase === 'done') {
          setRunStatus('done')
          getResults(session.session_id)
            .then((r) => setResults(r))
            .catch((e) => setToast({ kind: 'error', text: `获取结果失败：${e.message}` }))
        }
        if (msg.phase === 'error') {
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

  const lossData = useMemo(() => {
    const epochsArr: number[] = []
    const train: number[] = []
    const val: number[] = []
    for (const p of progress) {
      if (p.phase === 'training' && typeof p.epoch === 'number') {
        epochsArr.push(p.epoch)
        train.push(typeof p.train_loss === 'number' ? p.train_loss : NaN)
        val.push(typeof p.val_loss === 'number' ? p.val_loss : NaN)
      }
    }
    return { epochsArr, train, val }
  }, [progress])

  const lastTraining = [...progress].reverse().find((p) => p.phase === 'training')
  const trainingProgressPct =
    lastTraining && lastTraining.total_epochs
      ? Math.round(((lastTraining.epoch ?? 0) * 100) / lastTraining.total_epochs)
      : 0

  return (
    <div className="app-shell">
      <div className="flex h-full gap-2 px-2 py-2">
        <Sidebar />

        <main className="flex-1 overflow-y-auto space-y-2">
          <header className="glass-panel p-4 flex flex-col gap-3 border border-white/40">
            <div className="flex items-center justify-center">
              <h1 className="text-xl font-bold text-slate-900">
                多变量时间序列 · 根因分析与可视化
              </h1>
            </div>

            {runStatus === 'running' && (
              <div className="w-full bg-slate-100 rounded-full h-1.5 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-emerald-500 to-cyan-500 h-full transition-all"
                  style={{ width: `${trainingProgressPct}%` }}
                />
              </div>
            )}
          </header>

          <DataView />

          {session && (
            <div className="glass-panel p-4 flex flex-col gap-3">
              <div className="flex items-center justify-center gap-2">
                <span className="text-xs font-semibold text-slate-700">训练控制</span>
                {runStatus === 'running' && (
                  <span className="text-[11px] text-orange-600">实时同步中…</span>
                )}
              </div>
              <div className="flex flex-wrap items-center justify-center gap-3">
                <label className="text-xs text-slate-600 flex items-center gap-1">
                  Epochs
                  <input
                    type="number"
                    min={1}
                    max={5000}
                    step={1}
                    className="w-24 border rounded-lg px-3 py-2 text-xs bg-white/80"
                    value={epochs}
                    onChange={(e) => setEpochs(parseInt(e.target.value || '1', 10))}
                  />
                </label>
                <label className="text-xs text-slate-600 flex items-center gap-1">
                  LR
                  <input
                    type="number"
                    step={0.0001}
                    className="w-24 border rounded-lg px-3 py-2 text-xs bg-white/80"
                    value={lr}
                    onChange={(e) => setLr(parseFloat(e.target.value || '0.001'))}
                  />
                </label>
                <button
                  disabled={runStatus === 'running'}
                  onClick={handleRun}
                  className="px-5 py-2 rounded-lg bg-gradient-to-r from-emerald-500 to-cyan-500 text-white text-sm font-semibold shadow-lg shadow-emerald-200 hover:shadow-emerald-300 transition disabled:opacity-60"
                >
                  {runStatus === 'running' ? '⏳ 训练中…' : '🚀 启动训练'}
                </button>
              </div>
            </div>
          )}

          {lossData.epochsArr.length > 0 && (
            <div className="glass-panel p-4 space-y-3">
              {session && (
                <div className="flex items-center gap-3 text-xs">
                  <span
                    className={`inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium ${
                      runStatus === 'idle'
                        ? 'bg-slate-200 text-slate-700'
                        : runStatus === 'running'
                        ? 'bg-orange-200 text-orange-800'
                        : runStatus === 'done'
                        ? 'bg-emerald-200 text-emerald-800'
                        : 'bg-rose-200 text-rose-800'
                    }`}
                  >
                    {runStatus.toUpperCase()}
                  </span>
                  {lastTraining && (
                    <span className="text-slate-600">
                      Epoch {lastTraining.epoch}/{lastTraining.total_epochs} ·
                      train={lastTraining.train_loss?.toFixed(4)} ·
                      val={lastTraining.val_loss?.toFixed(4)}
                    </span>
                  )}
                  {runError && (
                    <span className="text-rose-600">错误：{runError}</span>
                  )}
                </div>
              )}

              <Plot
                data={[
                  {
                    x: lossData.epochsArr,
                    y: lossData.train,
                    mode: 'lines+markers',
                    name: 'train_loss',
                    line: { color: '#3b82f6' },
                  },
                  {
                    x: lossData.epochsArr,
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
                  yaxis: { title: 'Loss' },
                }}
                style={{ width: '100%' }}
                useResizeHandler
                config={{ responsive: true, displayModeBar: false }}
              />
            </div>
          )}

          <ResultsView />

          <StreamView />
        </main>
      </div>

      {toast && (
        <div
          className={`fixed bottom-6 right-6 px-4 py-3 rounded shadow-lg text-sm text-white max-w-sm
            ${toast.kind === 'success' ? 'bg-emerald-600' : toast.kind === 'error' ? 'bg-rose-600' : 'bg-slate-800'}`}
        >
          {toast.text}
        </div>
      )}
    </div>
  )
}
