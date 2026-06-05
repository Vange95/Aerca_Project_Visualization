export interface StreamTick {
  type: 'tick' | 'stopped' | 'hello' | 'ping' | 'error'
  t?: number
  values?: number[]
  scores?: number[]
  detected?: boolean[]
  inference_ms?: number | null
  detection_source?: 'model' | 'rule' | 'model+rule' | string | null
  rule_hits?: {
    id: string
    name: string
    reason?: string
    affected_vars?: number[]
  }[]
  is_anomaly_step?: boolean
  is_fault_active?: boolean
  fault_start?: boolean
  anomaly_vars?: number[] | null
  anomaly_amp?: number | null
  fault?: FaultScenario | null
  is_running?: boolean
  message?: string
}

export interface FaultScenario {
  id: string
  name: string
  category: string
  pattern: string
  affected_vars: number[]
  duration: number
  root_cause: string
  effect: string
  confidence: number
  fault_id?: string
  fault_name?: string
  fault_category?: string
  fault_pattern?: string
  fault_effect?: string
  fault_root_cause?: string
  fault_confidence?: number
  fault_progress?: number
  amplitude?: number
}

export interface SelectedFaultContext {
  datasetName: string
  faultId: string
}

export interface DatasetInfo {
  name: string
  display_name?: string
  description?: string
  scenario?: string
  signal_profile?: string
  typical_faults?: string[]
  fault_closed_loop?: boolean
  adtypes: string[]
  supports_adtype?: boolean
  has_causal_struct?: boolean
}

export interface CreateSessionResponse {
  session_id: string
  dataset_name: string
  options_summary: Record<string, string | number | boolean | null>
  num_vars: number
  training_size: number
  testing_size: number
  T: number
  has_causal_struct: boolean
  use_slice: boolean
  supports_adtype: boolean
  fault_id?: string | null
  fault_name?: string | null
  fault_category?: string | null
  fault_pattern?: string | null
}

export interface SessionInfo {
  session_id: string
  dataset_name: string
  run_status: 'idle' | 'running' | 'done' | 'failed' | 'stopped'
  n_total_samples?: number
  training_size?: number
  testing_size?: number
  use_slice?: boolean
  num_vars?: number
  T?: number
  seed?: number
  device?: string
  has_causal_struct?: boolean
  supports_adtype?: boolean
  fault_id?: string | null
  fault_name?: string | null
  fault_category?: string | null
  fault_pattern?: string | null
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
  batch?: number
  total_batches?: number
  train_loss?: number
  batch_loss?: number
  val_loss?: number
  best_val_loss?: number
  stage?: string
  run_status?: 'idle' | 'running' | 'done' | 'failed' | 'stopped'
}

export interface RunResults {
  dataset_name?: string
  training_size: number
  use_slice: boolean
  num_vars: number
  test_size: number
  causal_discovery: {
    f1_mean: number
    f1_std: number
    auroc_mean: number
    auroc_std: number
    auprc_mean: number
    auprc_std: number
    hamming_mean: number
    hamming_std: number
    predicted_causal_matrix: number[][]
    true_causal_matrix: number[][]
  } | null
  root_cause: {
    ac_at: number[]
    ac_star_at: number[]
    avg_at_10: number
    avg_star_at_500: number
    time_tolerance?: number
    relaxed_ac_at?: number[]
    relaxed_ac_star_at?: number[]
    relaxed_avg_at_10?: number
    relaxed_avg_star_at_500?: number
    num_vars: number
    predicted_root_causes: {
      sample_idx: number
      root_cause_var_idx: number
      root_cause_time: number
    }[]
  }
}
