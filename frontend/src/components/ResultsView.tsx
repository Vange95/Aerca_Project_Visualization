import { useMemo } from 'react'
import Plot from './Plot'
import MatrixHeatmap from './MatrixHeatmap'
import { useAppStore } from '../store'
import { getVariableNames } from '../domain/variableNames'
import { GitBranch, Target } from 'lucide-react'

const COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
  '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`
}

export default function ResultsView() {
  const session = useAppStore((s) => s.session)
  const sample = useAppStore((s) => s.currentSample)
  const idx = useAppStore((s) => s.currentSampleIdx)
  const results = useAppStore((s) => s.results)

  const varNames = useMemo(
    () => (session ? getVariableNames(
      session.dataset_name,
      session.num_vars,
      session.fault_id ?? String(session.options_summary?.fault_id ?? ''),
    ) : []),
    [session],
  )

  if (!results || !session) return null

  const rc = results.root_cause
  const cd = results.causal_discovery
  const timeTolerance = rc.time_tolerance ?? 5
  const relaxedAcStarAt = rc.relaxed_ac_star_at ?? rc.ac_star_at
  const relaxedAvgStarAt = rc.relaxed_avg_star_at_500 ?? rc.avg_star_at_500

  // 当前样本根因预测
  const predicted = rc.predicted_root_causes.find((p) => p.sample_idx === idx)

  // 异常区间（从 label 中提取）
  let anomalyShapes: any[] = []
  if (sample) {
    const anomalyMask = sample.label.map((row) => row.some((v) => v > 0))
    let inAnomaly = false
    let start = 0
    for (let t = 0; t < anomalyMask.length; t++) {
      if (anomalyMask[t] && !inAnomaly) {
        start = t
        inAnomaly = true
      } else if (!anomalyMask[t] && inAnomaly) {
        anomalyShapes.push({
          type: 'rect',
          xref: 'x',
          yref: 'paper',
          x0: start,
          x1: t,
          y0: 0,
          y1: 1,
          fillcolor: '#f59e0b',
          opacity: 0.16,
          line: { width: 0 },
        })
        inAnomaly = false
      }
    }
    if (inAnomaly) {
      anomalyShapes.push({
        type: 'rect',
        xref: 'x',
        yref: 'paper',
        x0: start,
        x1: anomalyMask.length,
        y0: 0,
        y1: 1,
        fillcolor: '#f59e0b',
        opacity: 0.16,
        line: { width: 0 },
      })
    }
  }

  // 根因预测高亮（红色虚线）
  if (predicted && sample) {
    anomalyShapes.push({
      type: 'line',
      xref: 'x',
      yref: 'paper',
      x0: predicted.root_cause_time,
      x1: predicted.root_cause_time,
      y0: 0,
      y1: 1,
      line: { color: 'red', width: 3, dash: 'dash' },
    })
  }

  const highlightTraces = sample
    ? sample.x_ab[0].map((_, v) => ({
        x: Array.from({ length: sample.T }, (_, i) => i),
        y: sample.x_ab.map((row) => row[v]),
        mode: 'lines',
        name: varNames[v],
        line: { color: COLORS[v % COLORS.length], width: 1.5 },
      }))
    : []

  return (
      <div className="space-y-6">
      <div className="content-card space-y-4">
        <div>
          <h3 className="section-title mb-1">
            <Target className="h-4 w-4 text-blue-600" />
            根因分析效果
          </h3>
          <p className="text-xs leading-5 text-slate-500">
            主指标关注根因变量是否找对；联合时间指标采用 ±{timeTolerance} tick 容忍，适合阶跃、归零、漂移这类持续故障。
          </p>
        </div>

        <div>
          <p className="mb-2 text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">主指标：根因变量定位</p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="metric-card">
              <span className="metric-label">变量 Top-1</span>
              <span className="metric-value">{pct(rc.ac_at[0] ?? 0)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">变量 Top-5</span>
              <span className="metric-value">{pct(rc.ac_at[2] ?? 0)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">变量 Top-10 平均</span>
              <span className="metric-value">{rc.avg_at_10.toFixed(3)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">预测根因变量</span>
              <span className="metric-value text-xl">
                {predicted ? varNames[predicted.root_cause_var_idx]?.replace(/\s*\((var|metric)_\d+\)\s*$/i, '') : '--'}
              </span>
            </div>
          </div>
        </div>

        <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <p className="text-xs font-semibold uppercase tracking-[0.12em] text-slate-500">辅助指标：变量 + 时间联合定位</p>
            <span className="rounded-full bg-white px-2 py-0.5 text-[11px] font-semibold text-slate-600">时间容忍 ±{timeTolerance} tick</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="metric-card bg-white">
              <span className="metric-label">宽松联合 Top-10</span>
              <span className="metric-value">{pct(relaxedAcStarAt[1] ?? 0)}</span>
            </div>
            <div className="metric-card bg-white">
              <span className="metric-label">宽松联合 Top-100</span>
              <span className="metric-value">{pct(relaxedAcStarAt[2] ?? 0)}</span>
            </div>
            <div className="metric-card bg-white">
              <span className="metric-label">原始严格 Top-10</span>
              <span className="metric-value text-slate-700">{pct(rc.ac_star_at[1] ?? 0)}</span>
            </div>
            <div className="metric-card bg-white">
              <span className="metric-label">宽松联合均值</span>
              <span className="metric-value text-slate-700">{relaxedAvgStarAt.toFixed(3)}</span>
            </div>
          </div>
          <p className="mt-2 text-[11px] leading-5 text-slate-500">
            原始严格指标要求变量和精确时间点完全命中；宽松指标允许预测时间与真实故障段相差 {timeTolerance} tick 以内。
          </p>
        </div>
      </div>

      {sample && predicted && (
        <div className="content-card">
          <h3 className="section-title">异常时序 + 根因高亮（样本 #{idx}）</h3>
          <p className="text-xs text-slate-500 mb-2">
            <span className="inline-block w-3 h-3 bg-orange-300 mr-1 align-middle" />
            真实异常区间 ·
            <span className="inline-block w-3 h-0.5 bg-red-500 mx-1 align-middle border-dashed" />
            模型预测根因：{varNames[predicted.root_cause_var_idx]} @ t={predicted.root_cause_time}
          </p>
          <Plot
            data={highlightTraces as any}
            layout={{
              height: 460,
              xaxis: { title: '时间步' },
              yaxis: { title: '变量值' },
              shapes: anomalyShapes,
              margin: { t: 30, b: 50, l: 60, r: 30 },
              legend: { title: { text: '变量' } },
            }}
            style={{ width: '100%' }}
            useResizeHandler
            config={{ responsive: true, displayModeBar: false }}
          />
        </div>
      )}

      {cd && (
        <div className="content-card space-y-4">
          <h3 className="section-title">
            <GitBranch className="h-4 w-4 text-blue-600" />
            因果发现结果
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="metric-card">
              <span className="metric-label">F1 分数</span>
              <span className="metric-value">{cd.f1_mean.toFixed(4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">AUROC</span>
              <span className="metric-value">{cd.auroc_mean.toFixed(4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">AUPRC</span>
              <span className="metric-value">{cd.auprc_mean.toFixed(4)}</span>
            </div>
            <div className="metric-card">
              <span className="metric-label">Hamming Distance</span>
              <span className="metric-value">{cd.hamming_mean.toFixed(4)}</span>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-5">
            <div>
              <p className="text-sm font-medium text-slate-700 mb-1">真实因果矩阵</p>
              <MatrixHeatmap
                z={cd.true_causal_matrix}
                xLabels={varNames}
                yLabels={varNames}
                xAxisTitle="影响变量"
                yAxisTitle="被影响变量"
                height={420}
                domain={[0, Math.max(1, ...cd.true_causal_matrix.flat())]}
                showValues={session.num_vars <= 12}
              />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-700 mb-1">模型预测因果矩阵</p>
              <MatrixHeatmap
                z={cd.predicted_causal_matrix}
                xLabels={varNames}
                yLabels={varNames}
                xAxisTitle="影响变量"
                yAxisTitle="被影响变量"
                height={420}
                domain={[0, Math.max(1, ...cd.predicted_causal_matrix.flat())]}
                showValues={session.num_vars <= 12}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
