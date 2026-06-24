import { useEffect, useState } from 'react'
import { getModels, getSettings, saveSettings, getEngineStatus } from '../api/client'

const ENGINES = ['claude', 'openai', 'agy']
const ENGINE_LABEL = { claude: 'Claude', openai: 'OpenAI Codex', agy: 'AGY CLI' }

export default function SettingsModal({ onClose }) {
  const [settings, setSettings] = useState(null)
  const [models, setModels] = useState({ claude: [], openai: [], agy: [] })
  const [status, setStatus] = useState({ label: '확인 중...', account_label: '' })
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    Promise.all([getSettings(), getModels(), getEngineStatus()]).then(
      ([s, m, st]) => {
        const validModels = m[s.engine] || []
        const first = validModels[0] || ''
        const sanitized = { ...s }
        if (!['per_doc', 'hybrid'].includes(sanitized.comparison_mode)) {
          sanitized.comparison_mode = 'per_doc'
        }
        for (const key of ['model_parser', 'model_report']) {
          if (!sanitized[key] || (validModels.length > 0 && !validModels.includes(sanitized[key]))) {
            sanitized[key] = s.engine === 'claude'
              ? 'claude-haiku-4-5-20251001'
              : s.engine === 'openai'
                ? 'gpt-5.4-mini'
                : first
          }
        }
        if (!sanitized.model_compare || (validModels.length > 0 && !validModels.includes(sanitized.model_compare))) {
          sanitized.model_compare = s.engine === 'claude'
            ? 'claude-sonnet-4-6'
            : s.engine === 'openai'
              ? 'gpt-5.5'
              : first
        }
        setSettings(sanitized)
        setModels(m)
        setStatus(st)
      }
    )
  }, [])

  function set(key, val) {
    setSettings(prev => ({ ...prev, [key]: val }))
  }

  function handleEngineChange(newEngine) {
    const engineModels = models[newEngine] || []
    const first = engineModels[0] || ''
    const updates = { engine: newEngine }
    if (newEngine === 'claude') {
      updates.model_parser = 'claude-haiku-4-5-20251001'
      updates.model_compare = 'claude-sonnet-4-6'
      updates.model_report = 'claude-haiku-4-5-20251001'
    } else if (newEngine === 'openai') {
      updates.model_parser = 'gpt-5.4-mini'
      updates.model_compare = 'gpt-5.5'
      updates.model_report = 'gpt-5.4-mini'
    } else {
      updates.model_parser = first
      updates.model_compare = first
      updates.model_report = first
    }
    setSettings(prev => ({ ...prev, ...updates }))
  }

  async function handleSave() {
    await saveSettings(settings)
    const st = await getEngineStatus()
    setStatus(st)
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  if (!settings) {
    return (
      <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 text-sm text-gray-500">설정 로드 중...</div>
      </div>
    )
  }

  const currentModels = models[settings.engine] || []
  return (
    <div className="fixed inset-0 bg-black/30 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-white rounded-xl shadow-2xl w-[640px] max-h-[90vh] flex flex-col overflow-hidden"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-5 py-4 border-b shrink-0">
          <h2 className="font-semibold text-gray-800">설정</h2>
          <button className="text-gray-400 hover:text-gray-600 text-lg" onClick={onClose}>x</button>
        </div>

        <div className="flex-1 overflow-y-auto p-5 space-y-5">
          <>
              <div>
                <label className="text-xs font-medium text-gray-600 block mb-2">엔진 선택</label>
                <div className="flex gap-3">
                  {ENGINES.map(e => (
                    <label key={e} className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="radio"
                        name="engine"
                        value={e}
                        checked={settings.engine === e}
                        onChange={() => handleEngineChange(e)}
                        className="accent-blue-500"
                      />
                      <span className="text-sm">{ENGINE_LABEL[e]}</span>
                    </label>
                  ))}
                </div>
                {status.detail && (
                  <div className="mt-2 text-xs text-red-600 leading-relaxed whitespace-pre-wrap">
                    {status.detail}
                  </div>
                )}
              </div>

              <div className="bg-gray-50 rounded px-3 py-2 text-sm">
                <div>
                  상태: <span className={status.status === 'cli_ready' ? 'text-green-600' : 'text-red-500'}>
                    {status.label}
                  </span>
                </div>
                <div className="mt-1 text-xs text-gray-500">
                  계정: <span className="font-medium text-gray-700">
                    {status.account_label || status.account_email || '연결 계정 확인 불가'}
                  </span>
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-gray-600 block mb-1">
                  작업 단계별 모델
                  <span className="text-gray-400 font-normal ml-1">(각 단계에서 사용할 모델)</span>
                </label>
                <div className="space-y-1.5 bg-gray-50 rounded p-2">
                  {[
                    { key: 'model_parser', label: '청구항 파서', desc: '청구항 추출 및 구성요소 분해' },
                    { key: 'model_compare', label: '구성요소 대비', desc: '인용발명 전문 대비 판단' },
                    { key: 'model_report', label: 'Phase 1 생성', desc: '구성요소 분석 보고서 작성' },
                  ].map(({ key, label, desc }) => (
                    <div key={key} className="flex items-center gap-2">
                      <div className="w-28 shrink-0">
                        <div className="text-xs font-medium text-gray-700">{label}</div>
                        <div className="text-[10px] text-gray-400">{desc}</div>
                      </div>
                      <select
                        className="flex-1 border rounded px-2 py-1 text-xs bg-white"
                        value={settings[key] || currentModels[0] || ''}
                        onChange={e => set(key, e.target.value)}
                      >
                        {currentModels.map(m => <option key={m} value={m}>{m}</option>)}
                      </select>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-gray-600 block mb-2">인용발명 비교 방식</label>
                <div className="grid grid-cols-2 gap-2">
                  {[
                    {
                      value: 'per_doc',
                      title: '문헌별 순차 대비',
                      description: '인용발명마다 LLM을 1회씩 호출해 개별적으로 정밀 대비합니다.',
                    },
                    {
                      value: 'hybrid',
                      title: '전체 통합 대비 (LLM 1회)',
                      description: '모든 인용발명을 한 프롬프트에 모아 청구항별로 한 번에 대비합니다.',
                    },
                  ].map(option => {
                    const selected = settings.comparison_mode === option.value
                    return (
                      <label
                        key={option.value}
                        className={`cursor-pointer rounded-lg border p-3 transition ${
                          selected
                            ? 'border-blue-500 bg-blue-50 ring-1 ring-blue-200'
                            : 'border-gray-200 bg-white hover:border-gray-300'
                        }`}
                      >
                        <input
                          type="radio"
                          name="comparison_mode"
                          value={option.value}
                          checked={selected}
                          onChange={() => set('comparison_mode', option.value)}
                          className="sr-only"
                        />
                        <div className={`text-xs font-semibold ${selected ? 'text-blue-700' : 'text-gray-700'}`}>
                          {option.title}
                        </div>
                        <div className="mt-1 text-[10px] leading-relaxed text-gray-500">
                          {option.description}
                        </div>
                      </label>
                    )
                  })}
                </div>
                {settings.comparison_mode === 'hybrid' && (
                  <div className="mt-2 rounded bg-amber-50 px-2.5 py-2 text-[10px] leading-relaxed text-amber-700">
                    입력 한도를 넘는 경우에도 모든 문헌을 포함하되, 각 문헌의 관련 문단을 자동으로 압축해 한 프롬프트로 전송합니다.
                  </div>
                )}
              </div>

              <div className="border rounded-lg overflow-hidden">
                <div className="flex items-center justify-between px-3 py-2 bg-purple-50">
                  <div>
                    <div className="text-xs font-semibold text-purple-800">RAG 단락 필터링</div>
                    <div className="text-[10px] text-purple-600 mt-0.5">
                      BGE-M3 Dense + BM25 Hybrid 검색으로 구성요소별 관련 단락만 골라 LLM 입력 토큰을 줄입니다.
                    </div>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer ml-3 shrink-0">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={!!settings.use_rag_retrieval}
                      onChange={e => set('use_rag_retrieval', e.target.checked)}
                    />
                    <div className="w-9 h-5 bg-gray-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-600" />
                  </label>
                </div>
                {settings.use_rag_retrieval && (
                  <div className="px-3 py-3 bg-white border-t space-y-3">
                    {status.rag && (
                      <div className="rounded border border-slate-200 bg-slate-50 px-2 py-1.5 text-[10px] text-slate-600">
                        실행 상태: dense {status.rag.dense} / Qdrant {status.rag.qdrant} / BM25 {status.rag.bm25} / reranker {status.rag.reranker}
                        {status.rag.fallback_reason && (
                          <div className="mt-1 text-amber-700">폴백 사유: {status.rag.fallback_reason}</div>
                        )}
                      </div>
                    )}
                    <div className="bg-amber-50 border border-amber-100 rounded px-2 py-1.5 text-[10px] text-amber-700">
                      최초 실행 시 BGE-M3 모델 다운로드가 필요할 수 있습니다. 이후에는 캐시에서 로드합니다.
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-1">
                        <label className="text-xs font-medium text-gray-600">최대 선택 단락 수</label>
                        <span className="text-xs font-bold text-purple-700">{settings.rag_top_k ?? 20}개</span>
                      </div>
                      <input
                        type="range"
                        min="10"
                        max="30"
                        step="1"
                        value={settings.rag_top_k ?? 20}
                        onChange={e => set('rag_top_k', parseInt(e.target.value))}
                        className="w-full h-1.5 bg-purple-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                      />
                      <div className="flex justify-between text-[9px] text-gray-400 mt-0.5">
                        <span>10개</span>
                        <span>20개 권장</span>
                        <span>30개</span>
                      </div>
                    </div>
                    <div className="rounded border border-gray-200 p-2.5 space-y-2">
                      <label className="flex items-center justify-between gap-3 cursor-pointer">
                        <span>
                          <span className="block text-xs font-medium text-gray-700">BGE reranker 사용</span>
                          <span className="block text-[10px] text-gray-500">추가 LLM 호출 없이 후보 문단을 로컬에서 재정렬합니다.</span>
                        </span>
                        <input
                          type="checkbox"
                          checked={!!settings.use_reranker}
                          onChange={e => set('use_reranker', e.target.checked)}
                          className="accent-blue-600"
                        />
                      </label>
                      {settings.use_reranker && (
                        <label className="flex items-center justify-between gap-3 text-xs text-gray-600">
                          LLM에 전달할 문단 수
                          <select
                            value={settings.reranker_top_k ?? 10}
                            onChange={e => set('reranker_top_k', parseInt(e.target.value))}
                            className="rounded border bg-white px-2 py-1"
                          >
                            {[5, 8, 10, 12, 15].map(value => <option key={value} value={value}>{value}개</option>)}
                          </select>
                        </label>
                      )}
                    </div>
                  </div>
                )}
              </div>
          </>
        </div>

        <div className="flex items-center justify-between px-5 py-3 border-t bg-gray-50 shrink-0">
          <button
            className="text-xs text-gray-400 hover:text-gray-600"
            onClick={() => {
              if (confirm('모든 설정을 초기화할까요?')) {
                setSettings({
                  engine: 'claude',
                  comparison_mode: 'per_doc',
                  model_parser: '',
                  model_compare: 'claude-sonnet-4-6',
                  model_report: '',
                  use_rag_retrieval: true,
                  rag_top_k: 20,
                  use_reranker: true,
                  reranker_top_k: 10,
                })
              }
            }}
          >
            초기화
          </button>
          <button
            className="bg-blue-600 text-white text-sm px-5 py-1.5 rounded hover:bg-blue-700 transition"
            onClick={handleSave}
          >
            {saved ? '저장됨' : '저장'}
          </button>
        </div>
      </div>
    </div>
  )
}
