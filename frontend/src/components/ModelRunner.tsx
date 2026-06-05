import { useEffect, useMemo, useRef, useState } from 'react'
import Plot from './Plot'
import { useAppStore } from '../store'
import { getResults, openProgressWS, runModel, stopTraining } from '../api'
import { Loader2, Play, RadioTower, Square } from 'lucide-react'

export default function ModelRunner() {
  const session = useAppStore((s) => s.session)
  const runStatus = useAppStore((s) => s.runStatus)
  const setRunStatus = useAppStore((s) => s.setRunStatus)
  const runError = useAppStore((s) => s.runError)
  const setRunError = useAppStore((s) => s.setRunError)
  const progress = useAppStore((s) => s.progressLog)
  const appendProgress = useAppStore((s) => s.appendProgress)
  const resetProgress = useAppStore((s) => s.resetProgress)
  const setResults = useAppStore((s) => s.setResults)
  const setToast = useAppStore((s) => s.setToast)

  const [epochs, setEpochs] = useState(50)
  const [lr, setLr] = useState(0.001)
  const [isStopping, setIsStopping] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  // 关闭已有 WS（卸载或会话变化时）
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
          setIsStopping(false)
          setRunStatus('done')
          // 拉取最终结果
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
    ws.onclose = () => {
      // 仅记录
    }
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

  // 训练曲线数据
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
    <div className="glass-panel p-5 space-y-5">
      <h3 className="section-title mb-0">
        <RadioTower className="h-4 w-4 text-blue-600" />
        AERCA 模型训练与根因分析
      </h3>

      {session && (
        <>
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-[minmax(110px,1fr)_minmax(110px,1fr)_minmax(290px,auto)]">
            <div>
              <label className="field-label">训练轮数 (epochs)</label>
              <input
                type="number"
                min={1}
                max={5000}
                step={1}
                className="field-control"
                value={epochs}
                onChange={(e) => setEpochs(parseInt(e.target.value || '1', 10))}
              />
            </div>
            <div>
              <label className="field-label">学习率</label>
              <input
                type="number"
                step={0.0001}
                className="field-control"
                value={lr}
                onChange={(e) => setLr(parseFloat(e.target.value || '0.001'))}
              />
            </div>
            <div className="flex items-end gap-2">
              <button
                disabled={runStatus === 'running'}
                onClick={handleRun}
                className="btn-primary flex-1"
              >
                {runStatus === 'running' ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {runStatus === 'running' ? '训练中' : '启动训练 + 根因分析'}
              </button>
              {runStatus === 'running' && (
                <button
                  type="button"
                  disabled={isStopping}
                  onClick={handleStopTraining}
                  className="btn-danger shrink-0"
                >
                  {isStopping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Square className="h-4 w-4" />}
                  {isStopping ? '停止中' : '停止训练'}
                </button>
              )}
            </div>
          </div>

          {/* 状态栏 */}
          <div className="flex items-center gap-3 text-sm">
            <span
              className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
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
            {lastTraining && (
              <span className="text-slate-600">
                Epoch {lastTraining.epoch}/{lastTraining.total_epochs} ·
                train={lastTraining.train_loss?.toFixed(4)} ·
                val={lastTraining.val_loss?.toFixed(4)} ·
                best={lastTraining.best_val_loss?.toFixed(4)}
              </span>
            )}
          </div>

          {runStatus === 'running' && (
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className="h-full bg-blue-600 transition-all"
                style={{ width: `${trainingProgressPct}%` }}
              />
            </div>
          )}

          {runError && (
            <div className="bg-rose-50 border border-rose-200 text-rose-700 p-3 rounded text-xs">
              <b>错误：</b> {runError}
            </div>
          )}

          {/* 训练曲线 */}
          {lossData.epochsArr.length > 0 && (
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
                  line: { color: '#ef4444' },
                },
              ]}
              layout={{
                title: { text: '实时训练曲线', font: { size: 16 } },
                height: 320,
                xaxis: { title: 'Epoch', gridcolor: '#e5e7eb' },
                yaxis: { title: 'Loss', gridcolor: '#e5e7eb' },
                margin: { t: 50, b: 50, l: 60, r: 30 },
                legend: { orientation: 'h' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)',
              }}
              style={{ width: '100%' }}
              useResizeHandler
              config={{ responsive: true, displayModeBar: false }}
            />
          )}

          {/* 阶段日志 */}
          {progress.length > 0 && (
            <details className="text-xs">
              <summary className="cursor-pointer text-slate-600 hover:text-slate-800">阶段日志 ({progress.length})</summary>
              <div className="mt-2 max-h-44 overflow-y-auto rounded-lg bg-slate-50 p-2 font-mono text-[11px] text-slate-700">
                {progress.map((p, i) => (
                  <div key={i}>
                    [{p.phase}] {p.epoch ? `epoch=${p.epoch}/${p.total_epochs} ` : ''}
                    {p.message ?? ''}{' '}
                    {p.train_loss !== undefined ? `train=${p.train_loss.toFixed(4)}` : ''}
                    {p.val_loss !== undefined ? ` val=${p.val_loss.toFixed(4)}` : ''}
                  </div>
                ))}
              </div>
            </details>
          )}
        </>
      )}
    </div>
  )
}
