// @ts-nocheck
import { useEffect, useMemo, useState } from 'react'
import Plot from './Plot'
import MatrixHeatmap from './MatrixHeatmap'
import { useAppStore } from '../store'
import { getSample } from '../api'
import { getVariableNames } from '../domain/variableNames'
import { BarChart3, GitBranch, LineChart, Search } from 'lucide-react'

const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
  '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', '#c49c94',
  '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5',
]

function patternPreviewLabel(pattern?: string) {
  const labels: Record<string, string> = {
    spike: '尖峰',
    step: '阶跃',
    step_up: '阶跃升高',
    step_down: '阶跃降低',
    drop_to_zero: '多路归零',
    signal_zero: '信号丢失',
    gradual_drift: '渐进漂移',
    stuck_value: '卡滞死值',
    oscillation: '异常振荡',
    causal: '因果传播',
    none: '数据波动',
  }
  return pattern ? (labels[pattern] ?? pattern) : '异常'
}

function buildPreviewSeries(pattern?: string) {
  const x = Array.from({ length: 80 }, (_, i) => i)
  const base = x.map((i) => 0.12 * Math.sin(i / 7) + 0.04 * Math.sin(i / 3.5))
  const y = base.map((v) => v)
  const start = 34
  const end = 58

  if (pattern === 'none') {
    // dataset-level preview only
  } else if (pattern === 'step_up' || pattern === 'step') {
    for (let i = start; i < x.length; i++) y[i] += 1.0
  } else if (pattern === 'step_down') {
    for (let i = start; i < x.length; i++) y[i] -= 0.85
  } else if (pattern === 'drop_to_zero' || pattern === 'signal_zero') {
    for (let i = start; i < end; i++) y[i] = 0
  } else if (pattern === 'gradual_drift' || pattern === 'causal') {
    for (let i = start; i < x.length; i++) y[i] += Math.min(1.0, (i - start) * 0.035)
  } else if (pattern === 'stuck_value') {
    const hold = y[start]
    for (let i = start; i < end; i++) y[i] = hold
  } else if (pattern === 'oscillation') {
    for (let i = start; i < end; i++) y[i] += 0.55 * Math.sin((i - start) * 0.85)
  } else {
    y[start] += 1.2
    y[start + 18] -= 0.9
  }

  return { x, y, start, end }
}

export default function DataView() {
  const session = useAppStore((s) => s.session)
  const sessionInfo = useAppStore((s) => s.sessionInfo)
  const previewAdtype = useAppStore((s) => s.previewAdtype)
  const previewFault = useAppStore((s) => s.previewFault)
  const sample = useAppStore((s) => s.currentSample)
  const idx = useAppStore((s) => s.currentSampleIdx)
  const setSample = useAppStore((s) => s.setCurrentSample)
  const setIdx = useAppStore((s) => s.setCurrentSampleIdx)
  const trueCausal = useAppStore((s) => s.trueCausalMatrix)
  const runStatus = useAppStore((s) => s.runStatus)
  const results = useAppStore((s) => s.results)

  const [showCompare, setShowCompare] = useState(false)

  // 测试集大小
  const testSize = useMemo(() => {
    if (results) return results.test_size
    if (sessionInfo?.use_slice) return sessionInfo.testing_size
    return sessionInfo?.n_total_samples ?? 0
  }, [results, sessionInfo])

  // 索引切换 → 重新拉取样本
  useEffect(() => {
    if (!session) return
    let cancelled = false
    getSample(session.session_id, idx, 'auto')
      .then((s) => {
        if (!cancelled) setSample(s)
      })
      .catch(() => {/* ignore */})
    return () => {
      cancelled = true
    }
  }, [session, idx, runStatus, setSample])

  if (!session || !sample) {
    const previewPattern = previewFault?.pattern ?? previewAdtype
    const preview = buildPreviewSeries(previewPattern)
    const showFaultShape = previewPattern !== 'none'
    const previewTitle = previewFault ? `${previewFault.name}预览` : showFaultShape ? `${patternPreviewLabel(previewPattern)}示例` : '数据波动预览'

    return (
      <div className="min-h-[560px] space-y-3">
        <div className="content-card flex items-center justify-between">
          <div>
            <p className="section-kicker">Dataset Preview</p>
            <h3 className="mt-1 text-base font-semibold text-slate-900">等待数据集</h3>
          </div>
          <span className="status-badge border-slate-200 bg-slate-50 text-slate-600">Idle</span>
        </div>
        <div className="glass-panel p-4">
          <div className="flex items-center justify-between mb-1">
            <h3 className="section-title mb-0 text-sm">
              <LineChart className="h-4 w-4 text-blue-600" />
              {previewTitle}
            </h3>
            <span className="text-[11px] text-slate-500">
              {previewFault ? `${previewFault.category} · ${patternPreviewLabel(previewPattern)}` : '预览'}
            </span>
          </div>
          <Plot
            data={[
              {
                x: preview.x,
                y: preview.y,
                mode: 'lines',
                name: previewFault?.name ?? patternPreviewLabel(previewPattern),
                line: { color: '#0f766e', width: 2 },
              },
            ]}
            layout={{
              height: 360,
              margin: { t: 20, b: 30, l: 40, r: 20 },
              xaxis: { title: 'Time', gridcolor: '#e5e7eb' },
              yaxis: { title: 'Value', gridcolor: '#e5e7eb' },
              shapes: showFaultShape ? [
                {
                  type: 'rect',
                  xref: 'x',
                  yref: 'paper',
                  x0: preview.start,
                  x1: preview.end,
                  y0: 0,
                  y1: 1,
                  fillcolor: '#f97316',
                  opacity: 0.12,
                  line: { width: 0 },
                },
              ] : [],
              annotations: previewFault ? [
                {
                  x: preview.start,
                  y: 1,
                  yref: 'paper',
                  text: previewFault.effect,
                  showarrow: false,
                  xanchor: 'left',
                  yanchor: 'bottom',
                  font: { size: 11, color: '#475569' },
                },
              ] : [],
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
            }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      </div>
    )
  }

  const varNames = getVariableNames(
    session.dataset_name,
    sample.num_vars,
    session.fault_id ?? String(session.options_summary?.fault_id ?? ''),
  )
  const timeSteps = Array.from({ length: sample.T }, (_, i) => i)

  // ===== 正常 vs 异常对比 =====
  const compareTraces: any[] = []
  for (let v = 0; v < sample.num_vars; v++) {
    compareTraces.push({
      x: timeSteps,
      y: sample.x_n.map((row) => row[v]),
      mode: 'lines',
      name: `${varNames[v]} (正常)`,
      line: { color: COLORS[v % COLORS.length], width: 1.5, dash: 'dot' },
    })
    compareTraces.push({
      x: timeSteps,
      y: sample.x_ab.map((row) => row[v]),
      mode: 'lines',
      name: `${varNames[v]} (异常)`,
      line: { color: COLORS[v % COLORS.length], width: 2 },
    })
  }

  // ===== 异常差值 =====
  const diffTraces: any[] = []
  for (let v = 0; v < sample.num_vars; v++) {
    diffTraces.push({
      x: timeSteps,
      y: sample.x_ab.map((row, t) => row[v] - sample.x_n[t][v]),
      mode: 'lines',
      name: varNames[v],
      line: { color: COLORS[v % COLORS.length], width: 1.5 },
    })
  }

  // ===== 单独正常 / 异常 =====
  const buildTraces = (data: number[][]) =>
    data[0].map((_, v) => ({
      x: timeSteps,
      y: data.map((row) => row[v]),
      mode: 'lines',
      name: varNames[v],
      line: { color: COLORS[v % COLORS.length], width: 1.5 },
    }))

  // ===== 异常热力图 =====
  // label shape: T x num_vars，画图时转置 → variables × timesteps
  const labelMatrix = Array.from({ length: sample.num_vars }, (_, v) => sample.label.map((row) => row[v]))

  return (
    <div className="space-y-6">
      <div className="content-card">
        <div className="flex items-center justify-between mb-3">
          <h3 className="section-title m-0">
            <BarChart3 className="h-4 w-4 text-blue-600" />
            测试集样本可视化 - {session.dataset_name.toUpperCase()}
          </h3>
          <span className="text-xs text-slate-500">
            共 {testSize} 个测试样本{sample.from_test_set ? '（来自模型测试集）' : '（原始数据）'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-slate-600">样本索引</span>
          <input
            type="range"
            min={0}
            max={Math.max(0, testSize - 1)}
            value={idx}
            onChange={(e) => setIdx(parseInt(e.target.value, 10))}
            className="flex-1"
          />
          <span className="text-sm font-medium w-10 text-right">{idx}</span>
        </div>
      </div>

      <div className="glass-panel p-4 space-y-3">
        <div className="flex items-center justify-between mb-3">
          <h3 className="section-title mb-0">
            <Search className="h-4 w-4 text-blue-600" />
            异常变化对比
          </h3>
          <button
            className="btn-secondary px-3 py-1.5"
            onClick={() => setShowCompare((v) => !v)}
          >
            {showCompare ? '收起对比图' : '显示对比图'}
          </button>
        </div>

        {showCompare && (
          <div className="space-y-4">
            <Plot
              data={compareTraces}
              layout={{
                title: '正常数据 vs 异常数据叠加对比（实线=异常，虚线=正常）',
                height: 400,
                xaxis: { title: '时间步' },
                yaxis: { title: '变量值' },
                legend: { title: { text: '变量' } },
                margin: { t: 50, b: 50, l: 60, r: 30 },
              }}
              style={{ width: '100%' }}
              useResizeHandler
              config={{ responsive: true, displayModeBar: false }}
            />
            <Plot
              data={diffTraces}
              layout={{
                title: '异常差值曲线（差值越大表示异常越明显）',
                height: 260,
                xaxis: { title: '时间步' },
                yaxis: { title: '差值' },
                margin: { t: 50, b: 50, l: 60, r: 30 },
              }}
              style={{ width: '100%' }}
              useResizeHandler
              config={{ responsive: true, displayModeBar: false }}
            />
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="glass-panel p-4 space-y-3">
          <h3 className="section-title mb-0">正常时间序列</h3>
          <Plot
            data={buildTraces(sample.x_n)}
            layout={{ height: 300, xaxis: { title: '时间步' }, yaxis: { title: '值' }, margin: { t: 30, b: 40, l: 60, r: 30 } }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
        <div className="glass-panel p-4 space-y-3">
          <h3 className="section-title mb-0">异常时间序列</h3>
          <Plot
            data={buildTraces(sample.x_ab)}
            layout={{ height: 300, xaxis: { title: '时间步' }, yaxis: { title: '值' }, margin: { t: 30, b: 40, l: 60, r: 30 } }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      </div>

      <div className="content-card">
        <h3 className="section-title">异常位置热力图（1 = 异常）</h3>
        <MatrixHeatmap
          z={labelMatrix}
          xLabels={timeSteps.map(String)}
          yLabels={varNames}
          xAxisTitle="时间步"
          yAxisTitle="变量"
          height={260}
          domain={[0, 1]}
        />
      </div>

      {trueCausal && (
        <div className="content-card">
          <h3 className="section-title">
            <GitBranch className="h-4 w-4 text-blue-600" />
            真实因果矩阵
          </h3>
          {session.dataset_name === 'lotka_volterra' && (
            <p className="text-xs text-slate-500 mb-2">
              <b>生物解释</b>：前一半变量为<b>猎物 (Prey)</b>，后一半为<b>捕食者 (Predator)</b>，存在明显的捕食者-猎物因果交互。
            </p>
          )}
          {session.dataset_name === 'lorenz96' && (
            <p className="text-xs text-slate-500 mb-2">
              <b>混沌系统</b>：环形因果系统，每个变量受前两个变量影响，同时影响后两个变量。
            </p>
          )}
          <p className="mb-2 text-xs text-slate-500">
            行 = 被影响变量，列 = 影响变量；深色表示存在更强连接。
          </p>
          <MatrixHeatmap
            z={trueCausal}
            xLabels={varNames}
            yLabels={varNames}
            xAxisTitle="影响变量"
            yAxisTitle="被影响变量"
            height={300}
            domain={[0, Math.max(1, ...trueCausal.flat())]}
            showValues={sample.num_vars <= 12}
          />
        </div>
      )}
    </div>
  )
}
