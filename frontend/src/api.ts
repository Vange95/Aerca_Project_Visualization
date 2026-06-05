import type { CreateSessionResponse, DatasetInfo, FaultScenario, RunResults, SampleData, SessionInfo } from './types'

const API_BASE = '/api'

async function handleJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || res.statusText)
  }
  return res.json() as Promise<T>
}

export async function listDatasets(): Promise<DatasetInfo[]> {
  const res = await fetch(`${API_BASE}/datasets`)
  const data = await handleJson<{ datasets: DatasetInfo[] }>(res)
  return data.datasets
}

export interface CreateSessionPayload {
  dataset_name: string
  training_size?: number
  testing_size?: number
  T?: number
  seed?: number
  adtype?: string
  fault_id?: string
  preprocessing?: number
  device?: string
}

export async function createSession(payload: CreateSessionPayload): Promise<CreateSessionResponse> {
  const res = await fetch(`${API_BASE}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  return handleJson(res)
}

export async function getSample(sessionId: string, idx: number, source: 'raw' | 'auto' = 'auto'): Promise<SampleData> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/sample/${idx}?source=${source}`)
  return handleJson(res)
}

export async function getCausal(sessionId: string): Promise<number[][]> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/causal`)
  const data = await handleJson<{ matrix: number[][] }>(res)
  return data.matrix
}

export async function runModel(sessionId: string, body: Record<string, any>): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return handleJson(res)
}

export async function stopTraining(sessionId: string): Promise<{ status: string; run_status?: string }> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/run/stop`, { method: 'POST' })
  return handleJson(res)
}

export function openProgressWS(sessionId: string): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  return new WebSocket(`${protocol}://${host}/api/sessions/${sessionId}/ws`)
}

export async function getResults(sessionId: string): Promise<RunResults & SessionInfo> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/results`)
  return handleJson(res)
}

export async function startStream(sessionId: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/stream/start`, { method: 'POST' })
  return handleJson(res)
}

export async function listFaultScenarios(datasetName = 'linear'): Promise<FaultScenario[]> {
  const res = await fetch(`${API_BASE}/stream/fault-scenarios?dataset_name=${encodeURIComponent(datasetName)}`)
  const data = await handleJson<{ faults: FaultScenario[] }>(res)
  return data.faults
}

export async function injectAnomaly(sessionId: string, faultId = 'equipment_spike'): Promise<{ status: string; fault?: FaultScenario }> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/stream/inject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fault_id: faultId }),
  })
  return handleJson(res)
}

export async function stopStream(sessionId: string): Promise<{ status: string }> {
  const res = await fetch(`${API_BASE}/sessions/${sessionId}/stream/stop`, { method: 'POST' })
  return handleJson(res)
}

export function openStreamWS(sessionId: string): WebSocket {
  const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const host = window.location.host
  return new WebSocket(`${protocol}://${host}/api/sessions/${sessionId}/stream/ws`)
}
