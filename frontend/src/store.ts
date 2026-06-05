import { create } from 'zustand'
import type {
  CreateSessionResponse,
  DatasetInfo,
  ProgressEvent,
  RunResults,
  SampleData,
  SessionInfo,
  FaultScenario,
} from './types'

interface AppState {
  // 数据集列表
  datasets: DatasetInfo[]
  setDatasets: (d: DatasetInfo[]) => void

  // 会话
  session: CreateSessionResponse | null
  sessionInfo: SessionInfo | null
  setSession: (s: CreateSessionResponse | null) => void
  setSessionInfo: (s: SessionInfo | null) => void

  // 当前样本
  currentSample: SampleData | null
  currentSampleIdx: number
  setCurrentSample: (s: SampleData | null) => void
  setCurrentSampleIdx: (i: number) => void

  // 真实因果矩阵
  trueCausalMatrix: number[][] | null
  setTrueCausalMatrix: (m: number[][] | null) => void

  // 运行状态
  runStatus: 'idle' | 'running' | 'done' | 'failed' | 'stopped'
  setRunStatus: (s: 'idle' | 'running' | 'done' | 'failed' | 'stopped') => void
  runError: string | null
  setRunError: (e: string | null) => void

  // 训练进度
  progressLog: ProgressEvent[]
  appendProgress: (e: ProgressEvent) => void
  resetProgress: () => void

  // 模型结果
  results: RunResults | null
  setResults: (r: RunResults | null) => void

  // UI 状态：消息提示
  toast: { kind: 'info' | 'success' | 'error'; text: string } | null
  setToast: (t: AppState['toast']) => void

  // UI：预览所选异常类型
  previewAdtype: string
  setPreviewAdtype: (a: string) => void
  previewFault: FaultScenario | null
  setPreviewFault: (f: FaultScenario | null) => void
}

export const useAppStore = create<AppState>((set) => ({
  datasets: [],
  setDatasets: (d) => set({ datasets: d }),

  session: null,
  sessionInfo: null,
  setSession: (s) => set({ session: s }),
  setSessionInfo: (s) => set({ sessionInfo: s }),

  currentSample: null,
  currentSampleIdx: 0,
  setCurrentSample: (s) => set({ currentSample: s }),
  setCurrentSampleIdx: (i) => set({ currentSampleIdx: i }),

  trueCausalMatrix: null,
  setTrueCausalMatrix: (m) => set({ trueCausalMatrix: m }),

  runStatus: 'idle',
  setRunStatus: (s) => set({ runStatus: s }),
  runError: null,
  setRunError: (e) => set({ runError: e }),

  progressLog: [],
  appendProgress: (e) => set((st) => ({ progressLog: [...st.progressLog, e] })),
  resetProgress: () => set({ progressLog: [] }),

  results: null,
  setResults: (r) => set({ results: r }),

  toast: null,
  setToast: (t) => set({ toast: t }),

  previewAdtype: 'spike',
  setPreviewAdtype: (a) => set({ previewAdtype: a }),
  previewFault: null,
  setPreviewFault: (f) => set({ previewFault: f }),
}))
