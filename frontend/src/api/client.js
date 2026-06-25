const BASE = ''

function parseErrorMessage(text, fallback) {
  if (!text) return fallback
  try {
    const data = JSON.parse(text)
    return data.detail || data.message || text
  } catch (_) {
    return text
  }
}

export async function uploadFiles(priorFiles, onProgress, baseJobId = null) {
  return new Promise((resolve, reject) => {
    const form = new FormData()
    priorFiles.forEach(f => form.append('prior_files', f))
    if (baseJobId) form.append('base_job_id', baseJobId)

    const xhr = new XMLHttpRequest()
    xhr.upload.onprogress = e => {
      if (e.lengthComputable && onProgress) onProgress(Math.round((e.loaded / e.total) * 100))
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText))
      else reject(new Error(parseErrorMessage(xhr.responseText, `업로드 실패 (${xhr.status})`)))
    }
    xhr.onerror = () => reject(new Error('업로드 실패: 백엔드 서버(127.0.0.1:8200)에 연결할 수 없습니다. start.ps1로 백엔드와 프론트를 함께 실행해 주세요.'))
    xhr.open('POST', `${BASE}/analyze/upload`)
    xhr.send(form)
  })
}

export function streamPrepare(jobId, handlers) {
  const es = new EventSource(`${BASE}/analyze/prepare/${jobId}`)
  es.addEventListener('extract_prior', e => handlers.onLog?.(e.data))
  es.addEventListener('extract_prior_done', e => handlers.onLog?.(e.data))
  es.addEventListener('prepare_done', e => {
    handlers.onDone?.(JSON.parse(e.data))
    es.close()
  })
  es.addEventListener('error', e => {
    handlers.onError?.(e.data || '인용발명 준비 중 알 수 없는 오류가 발생했습니다.')
    es.close()
  })
  es.onerror = () => {
    handlers.onError?.('인용발명 준비 실패: 백엔드 서버 연결이 끊겼습니다.')
    es.close()
  }
  return es
}

export async function addManualClaim(jobId, payload) {
  const res = await fetch(`${BASE}/analyze/manual_claim/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function detectCategory(jobId) {
  const res = await fetch(`${BASE}/analyze/detect_category/${jobId}`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export function streamReport(jobId, claimNumber, handlers, useContext = true, force = false) {
  const url = `${BASE}/analyze/report/${jobId}/${claimNumber}?use_context=${useContext}&force=${force}`
  const es = new EventSource(url)
  es.addEventListener('start', e => handlers.onLog?.(e.data))
  es.addEventListener('analyze', e => handlers.onLog?.(e.data))
  es.addEventListener('log', e => handlers.onLog?.(e.data))
  es.addEventListener('generate', e => handlers.onLog?.(e.data))
  es.addEventListener('stream_chunk', e => handlers.onStreamChunk?.(e.data))
  es.addEventListener('phase1_result', e => handlers.onPhase1?.(JSON.parse(e.data)))
  es.addEventListener('done', e => {
    handlers.onDone?.(JSON.parse(e.data))
    es.close()
  })
  es.addEventListener('error', e => {
    handlers.onError?.(e.data || '보고서 생성 오류')
    es.close()
  })
  es.onerror = () => {
    handlers.onError?.('서버 연결 오류')
    es.close()
  }
  return es
}

// 종속항 일괄 보고서 — LLM 1회 호출 (스트리밍 없음)
export async function reportBatchDependent(jobId, claimNumbers, useContext = true, force = false, signal) {
  const res = await fetch(`${BASE}/analyze/report_batch_dependent/${jobId}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ claim_numbers: claimNumbers, use_context: useContext, force }),
    signal,
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getDependentBatchStatus(jobId) {
  const res = await fetch(`${BASE}/analyze/report_batch_dependent_status/${jobId}`)
  if (!res.ok) throw new Error(parseErrorMessage(await res.text(), '종속항 상태 조회 실패'))
  return res.json()
}

// 생성 취소 — 실행 중인 LLM CLI 프로세스 강제 종료
export async function cancelGeneration() {
  const res = await fetch(`${BASE}/analyze/cancel`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// 보고서 Q&A 채팅 — 질문 전용 (보고서 미수정)
export async function chatAboutReport(jobId, claimNumber, messages, reportMd, options = {}) {
  const res = await fetch(`${BASE}/analyze/chat/${jobId}/${claimNumber}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      report_md: reportMd,
      web_search: Boolean(options.webSearch),
    }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getSettings() {
  const res = await fetch(`${BASE}/settings`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function saveSettings(settings) {
  const res = await fetch(`${BASE}/settings`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(settings),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getModels() {
  const res = await fetch(`${BASE}/settings/models`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getEngineStatus() {
  const res = await fetch(`${BASE}/settings/status`)
  if (!res.ok) return { status: 'not_configured', label: '설정 필요' }
  return res.json()
}

export async function checkJobStatus(jobId) {
  const res = await fetch(`${BASE}/analyze/job_status/${jobId}`)
  if (!res.ok) return { exists: false, prior_count: 0 }
  return res.json()
}

export async function getContextInfo(jobId) {
  const res = await fetch(`${BASE}/analyze/context/${jobId}`)
  if (!res.ok) return { context_claims: [], count: 0 }
  return res.json()
}

export async function clearContext(jobId) {
  const res = await fetch(`${BASE}/analyze/context/${jobId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function deleteJob(jobId) {
  const res = await fetch(`${BASE}/analyze/job/${jobId}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function getClaimTree(jobId) {
  const res = await fetch(`${BASE}/analyze/claim_tree/${jobId}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function enhancePurpose(jobId) {
  const res = await fetch(`${BASE}/analyze/enhance_purpose/${jobId}`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function enhanceClaim(jobId, claimNumber) {
  const res = await fetch(`${BASE}/analyze/enhance_claim/${jobId}/${claimNumber}`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// 키워드 추출 (LLM 없음, 즉시)
export async function getKeywords(jobId, claimNumber) {
  const res = await fetch(`${BASE}/analyze/keywords/${jobId}/${claimNumber}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// 보완문서 검색 — 인용발명 미커버 구성요소 조회 (LLM 없음, 즉시)
export async function getGapElements(jobId, claimNumber) {
  const res = await fetch(`${BASE}/analyze/gap_search/${jobId}/${claimNumber}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// 보완문서 웹검색 실행 (LLM 1회, 웹검색 도구 사용 — 수 분 소요 가능)
export async function webSearchGap(jobId, claimNumber) {
  const res = await fetch(`${BASE}/analyze/gap_search/${jobId}/${claimNumber}/web_search`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// 검색 전략 캐시 조회 (LLM 없음, 즉시)
export async function getSearchStrategy(jobId, claimNumber) {
  const res = await fetch(`${BASE}/analyze/search_strategy/${jobId}/${claimNumber}`)
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

// 검색 전략 생성 (LLM 1회 — 구성 분해·키워드 확장·DB별 검색식, 1~3분 소요)
export async function generateSearchStrategy(jobId, claimNumber) {
  const res = await fetch(`${BASE}/analyze/search_strategy/${jobId}/${claimNumber}`, { method: 'POST' })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
