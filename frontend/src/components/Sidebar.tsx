import { useEffect, useState } from 'react'
import { useAppStore } from '../store'
import { createSession, getCausal, getSample, listDatasets, listFaultScenarios } from '../api'
import type { DatasetInfo, FaultScenario } from '../types'
import { ChevronDown, ChevronUp, Database, Loader2, Settings2 } from 'lucide-react'

const DEFAULT_TRAINING_SIZE = 100
const DEFAULT_TESTING_SIZE = 50

function patternToPreviewAdtype(pattern?: string) {
  if (pattern === 'step_up' || pattern === 'step_down' || pattern === 'drop_to_zero' || pattern === 'signal_zero' || pattern === 'stuck_value') return 'step'
  if (pattern === 'gradual_drift' || pattern === 'oscillation') return 'causal'
  return 'spike'
}

function patternLabel(pattern?: string) {
  const labels: Record<string, string> = {
    step_up: '阶跃升高',
    step_down: '阶跃降低',
    drop_to_zero: '多路归零',
    signal_zero: '信号丢失',
    gradual_drift: '渐进漂移',
    stuck_value: '卡滞死值',
    oscillation: '异常振荡',
  }
  return pattern ? (labels[pattern] ?? pattern) : '自动'
}

export default function Sidebar() {
  const datasets = useAppStore((s) => s.datasets)
  const setDatasets = useAppStore((s) => s.setDatasets)
  const setSession = useAppStore((s) => s.setSession)
  const setSessionInfo = useAppStore((s) => s.setSessionInfo)
  const setCurrentSample = useAppStore((s) => s.setCurrentSample)
  const setCurrentSampleIdx = useAppStore((s) => s.setCurrentSampleIdx)
  const setTrueCausalMatrix = useAppStore((s) => s.setTrueCausalMatrix)
  const setRunStatus = useAppStore((s) => s.setRunStatus)
  const setResults = useAppStore((s) => s.setResults)
  const resetProgress = useAppStore((s) => s.resetProgress)
  const setToast = useAppStore((s) => s.setToast)
  const setPreviewAdtype = useAppStore((s) => s.setPreviewAdtype)
  const setPreviewFault = useAppStore((s) => s.setPreviewFault)
  const activeSession = useAppStore((s) => s.session)

  const [datasetName, setDatasetName] = useState<string>('linear')
  const [adtype, setAdtype] = useState<string>('spike')
  const [faultScenarios, setFaultScenarios] = useState<FaultScenario[]>([])
  const [faultScenarioId, setFaultScenarioId] = useState<string>('')
  const [preprocessing, setPreprocessing] = useState<number>(1)
  const [seed, setSeed] = useState<number>(42)
  const [trainingSize, setTrainingSize] = useState<number>(DEFAULT_TRAINING_SIZE)
  const [testingSize, setTestingSize] = useState<number>(DEFAULT_TESTING_SIZE)
  const [T, setT] = useState<number>(200)
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    listDatasets()
      .then((d: DatasetInfo[]) => setDatasets(d))
      .catch((e: any) => setToast({ kind: 'error', text: `加载数据集列表失败：${e.message}` }))
  }, [setDatasets, setToast])

  const datasetList = Array.isArray(datasets) ? datasets : []
  const currentDataset: DatasetInfo | undefined = datasetList.find((d) => d.name === datasetName)
  const hasFaultClosedLoop = Boolean(currentDataset?.fault_closed_loop)
  const selectedFault = faultScenarios.find((f) => f.id === faultScenarioId)

  useEffect(() => {
    let cancelled = false
    setFaultScenarios([])
    setFaultScenarioId('')
    if (!hasFaultClosedLoop) {
      setPreviewAdtype('none')
      setPreviewFault(null)
      return () => { cancelled = true }
    }
    listFaultScenarios(datasetName)
      .then((faults) => {
        if (cancelled) return
        setFaultScenarios(faults)
        const nextFault = faults[0]
        setFaultScenarioId(nextFault?.id ?? '')
        if (nextFault) {
          setPreviewAdtype(patternToPreviewAdtype(nextFault.pattern))
          setPreviewFault(nextFault)
        } else {
          setPreviewFault(null)
        }
      })
      .catch(() => {
        if (!cancelled) {
          setFaultScenarios([])
          setFaultScenarioId('')
          setPreviewFault(null)
        }
      })
    return () => { cancelled = true }
  }, [datasetName, hasFaultClosedLoop, setPreviewAdtype, setPreviewFault])

  async function handleRun() {
    setBusy(true)
    try {
      const payload: any = {
        dataset_name: datasetName,
        preprocessing_data: preprocessing,
        seed,
        training_size: trainingSize,
        testing_size: testingSize,
        T,
      }
      if (hasFaultClosedLoop && faultScenarioId) payload.fault_id = faultScenarioId
      else if (currentDataset?.supports_adtype) payload.adtype = adtype

      const session = await createSession(payload)
      setSession(session)
      setSessionInfo({
        session_id: session.session_id,
        dataset_name: session.dataset_name,
        run_status: 'idle',
        n_total_samples: session.training_size + session.testing_size,
        training_size: session.training_size,
        testing_size: session.testing_size,
        use_slice: session.use_slice,
        supports_adtype: session.supports_adtype,
        num_vars: session.num_vars,
        T: session.T,
        has_causal_struct: session.has_causal_struct,
        fault_id: session.fault_id,
        fault_name: session.fault_name,
        fault_category: session.fault_category,
        fault_pattern: session.fault_pattern,
      })

      // 重置依赖于上一次会话的状态
      resetProgress()
      setRunStatus('idle')
      setResults(null)
      setCurrentSampleIdx(0)

      // 拉取首个样本和真实因果矩阵
      const sample = await getSample(session.session_id, 0, 'auto')
      setCurrentSample(sample)
      if (session.has_causal_struct) {
        const m = await getCausal(session.session_id)
        setTrueCausalMatrix(m)
      } else {
        setTrueCausalMatrix(null)
      }
      setToast({ kind: 'success', text: `数据准备完成 (session=${session.session_id.slice(0, 6)}...)` })
    } catch (e: any) {
      setToast({ kind: 'error', text: `数据准备失败：${e.message}` })
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside className="glass-panel max-h-[44vh] w-full shrink-0 overflow-y-auto p-4 lg:max-h-none lg:w-72">
      <div className="mb-5 flex items-center justify-between border-b border-slate-200 pb-4">
        <div>
          <p className="section-kicker">Control Panel</p>
          <h2 className="mt-1 text-base font-semibold text-slate-950">数据配置</h2>
        </div>
        <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-blue-100 bg-blue-50 text-blue-700">
          <Database className="h-4 w-4" />
        </span>
      </div>

      <div className="space-y-4">
        <div>
          <label className="field-label">仿真/数据场景（准备下一个会话）</label>
          <select
            className="field-control"
            value={datasetName}
            onChange={(e) => setDatasetName(e.target.value)}
          >
            {datasetList.map((d) => (
              <option key={d.name} value={d.name}>
                {d.display_name ? `${d.display_name} (${d.name})` : d.name}
              </option>
            ))}
          </select>
          {currentDataset && (
            <div className="mt-2 rounded-lg border border-blue-100 bg-blue-50/60 p-3 text-xs leading-5 text-slate-700">
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="font-semibold text-slate-900">{currentDataset.display_name ?? currentDataset.name}</span>
                <span className="rounded-full bg-white px-2 py-0.5 font-mono text-[10px] text-blue-700">{currentDataset.name}</span>
              </div>
              {currentDataset.scenario && (
                <p>{currentDataset.scenario}</p>
              )}
              {currentDataset.signal_profile && (
                <p className="mt-1 text-slate-500">信号特征：{currentDataset.signal_profile}</p>
              )}
              {currentDataset.typical_faults && currentDataset.typical_faults.length > 0 && (
                <p className="mt-1 text-slate-500">典型故障：{currentDataset.typical_faults.join(' / ')}</p>
              )}
            </div>
          )}
          {activeSession && activeSession.dataset_name !== datasetName && (
            <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 p-2 text-[11px] leading-5 text-amber-800">
              当前主视图仍是 {activeSession.dataset_name}；点击“准备数据集”后才会切换到 {datasetName}。
            </div>
          )}
        </div>

        {hasFaultClosedLoop && faultScenarios.length > 0 ? (
          <div>
            <label className="field-label">故障场景（生成/切片与实时注入）</label>
            <select
              className="field-control"
              value={faultScenarioId}
              onChange={(e) => {
                const next = e.target.value
                const fault = faultScenarios.find((f) => f.id === next)
                setFaultScenarioId(next)
                setPreviewAdtype(patternToPreviewAdtype(fault?.pattern))
                setPreviewFault(fault ?? null)
              }}
            >
              {faultScenarios.map((fault) => (
                <option key={fault.id} value={fault.id}>
                  {fault.name} · {fault.category}
                </option>
              ))}
            </select>
            {selectedFault && (
              <div className="mt-2 rounded-lg border border-blue-100 bg-blue-50/60 p-3 text-xs leading-5 text-slate-700">
                <div className="mb-1 flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-slate-900">{selectedFault.name}</span>
                  <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold text-blue-700">{selectedFault.category}</span>
                  <span className="rounded-full bg-white px-2 py-0.5 text-[10px] font-semibold text-slate-600">{patternLabel(selectedFault.pattern)}</span>
                </div>
                <p>信号表现：{selectedFault.effect}</p>
                <p className="mt-1 text-slate-500">预期根因：{selectedFault.root_cause}</p>
              </div>
            )}
          </div>
        ) : !hasFaultClosedLoop ? (
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs leading-5 text-slate-500">
            当前数据类型尚未建立业务故障闭环，仅用于算法或真实异常验证；因此不展示故障场景和业务测点含义。
          </div>
        ) : currentDataset?.supports_adtype && (
          <div>
            <label className="field-label">异常形态模板</label>
            <select
              className="field-control"
              value={adtype}
              onChange={(e) => {
                setAdtype(e.target.value)
                setPreviewAdtype(e.target.value)
                setPreviewFault(null)
              }}
            >
              {currentDataset.adtypes.map((a) => (
                <option key={a} value={a}>
                  {a === 'spike' ? 'Spike（突然尖峰异常）' : a === 'step' ? 'Step（阶跃持续异常）' : a === 'causal' ? 'Causal Propagation（因果传播异常）' : a}
                </option>
              ))}
            </select>
          </div>
        )}

        <div>
          <label className="field-label">数据处理方式</label>
          <div className="grid grid-cols-2 rounded-lg border border-slate-300 bg-slate-50 p-1">
            <button
              type="button"
              className={`rounded-md px-2 py-1.5 text-xs font-semibold transition ${preprocessing === 1 ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}
              onClick={() => setPreprocessing(1)}
            >
              生成新数据
            </button>
            <button
              type="button"
              className={`rounded-md px-2 py-1.5 text-xs font-semibold transition ${preprocessing === 0 ? 'bg-white text-blue-700 shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}
              onClick={() => setPreprocessing(0)}
            >
              加载已有数据
            </button>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
          <div className="flex items-center justify-between">
            <span className="inline-flex items-center gap-2 text-xs font-semibold text-slate-700">
              <Settings2 className="h-3.5 w-3.5 text-slate-500" />
              高级选项
            </span>
            <button
              type="button"
              className="inline-flex items-center gap-1 rounded-md px-1.5 py-1 text-xs font-semibold text-blue-700 hover:bg-blue-50"
              onClick={() => setShowAdvanced((v) => !v)}
            >
              {showAdvanced ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
              {showAdvanced ? '收起' : '展开'}
            </button>
          </div>

          {showAdvanced && (
            <div className="mt-3 space-y-3">
              <p className="rounded-md bg-white px-2 py-1.5 text-[11px] leading-5 text-slate-500">
                默认用于正式演示：训练 {DEFAULT_TRAINING_SIZE} / 测试 {DEFAULT_TESTING_SIZE}；快速调试可手动调低。
              </p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="field-label">训练样本</label>
                  <input
                    type="number"
                    className="field-control"
                    value={trainingSize}
                    min={1}
                    onChange={(e) => setTrainingSize(parseInt(e.target.value || '1', 10))}
                  />
                </div>
                <div>
                  <label className="field-label">测试样本</label>
                  <input
                    type="number"
                    className="field-control"
                    value={testingSize}
                    min={1}
                    onChange={(e) => setTestingSize(parseInt(e.target.value || '1', 10))}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="field-label">序列长度 T</label>
                  <input
                    type="number"
                    className="field-control"
                    value={T}
                    min={10}
                    onChange={(e) => setT(parseInt(e.target.value || '10', 10))}
                  />
                </div>
                <div>
                  <label className="field-label">随机种子</label>
                  <input
                    type="number"
                    className="field-control"
                    value={seed}
                    onChange={(e) => setSeed(parseInt(e.target.value || '0', 10))}
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        <button
          className="btn-primary mt-2 w-full"
          disabled={busy}
          onClick={handleRun}
        >
          {busy && <Loader2 className="h-4 w-4 animate-spin" />}
          {busy ? '准备中' : '准备数据集'}
        </button>
      </div>

      <p className="mt-6 border-t border-slate-200 pt-4 text-[11px] leading-5 text-slate-500">
        AERCA 多变量时间序列根因分析 · React + FastAPI
      </p>
    </aside>
  )
}
