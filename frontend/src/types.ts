export interface StreamTick {
  type: 'tick' | 'stopped' | 'hello' | 'ping' | 'error'
  t?: number
  values?: number[]
  scores?: number[]
  detected?: boolean[]
  is_anomaly_step?: boolean
  anomaly_vars?: number[] | null
  anomaly_amp?: number | null
  is_running?: boolean
  message?: string
}

export interface DatasetInfo {
  name: string
  description?: string
  adtypes: string[]
  supports_adtype?: boolean
  has_causal_struct?: boolean
}

export interface CreateSessionResponse {
  session_id: string
  dataset_name: string
  num_vars: number
  has_causal_struct: boolean
  run_status: 'idle' | 'running' | 'done' | 'failed'
}

export interface SessionInfo {
  session_id: string
  dataset_name: string
  run_status: 'idle' | 'running' | 'done' | 'failed'
  n_total_samples?: number
  training_size?: number
  testing_size?: number
  use_slice?: boolean
  num_vars?: number
  T?: number
  seed?: number
  device?: string
}

export interface SampleData {
  T: number
  num_vars: number
  x: number[][]
  x_n: number[][]
  x_ab: number[][]
  label: number[][]
  diff?: number[][]
  from_test_set?: boolean
}

export interface ProgressEvent {
  phase: string
  message?: string
  epoch?: number
  total_epochs?: number
  train_loss?: number
  val_loss?: number
  best_val_loss?: number
  stage?: string
  run_status?: 'idle' | 'running' | 'done' | 'failed'
}

export interface RunResults {
  dataset_name: string
  test_size: number
  causal_discovery: {
    f1: number
    auprc: number
    auroc: number
    hamming: number
    pred_matrix: number[][]
    true_matrix?: number[][]
  }
  root_cause: {
    ac_at: number[] // AC@1,3,5
    avg_topk: number
    predicted_root_causes: {
      sample_idx: number
      root_cause_time: number
      vars: number[]
      scores: number[]
    }[]
  }
}
