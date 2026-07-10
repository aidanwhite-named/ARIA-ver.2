import { useEffect, useState } from 'react'
import { getModels, getSettings, saveSettings, getEngineStatus } from '../api/client'

const ENGINES = ['claude', 'openai', 'agy']
const ENGINE_LABEL = { claude: 'Claude', openai: 'OpenAI Codex', agy: 'AGY CLI' }
const DEFAULT_SETTINGS = {
  engine: 'claude',
  comparison_mode: 'mixed',
  model_parser: 'claude-haiku-4-5-20251001',
  model_compare: 'claude-sonnet-4-6',
  model_report: 'claude-haiku-4-5-20251001',
}
const DEFAULT_MODELS = {
  claude: ['claude-opus-4-7', 'claude-sonnet-4-6', 'claude-haiku-4-5-20251001'],
  openai: ['gpt-5.5', 'gpt-5.4', 'gpt-5.4-mini'],
  agy: ['gemini-3.5-flash', 'gemini-3.1-flash-lite', 'gemini-3.1-pro-preview'],
}



export default function SettingsModal({ onClose }) {
  const [settings, setSettings] = useState(null)
  const [models, setModels] = useState(DEFAULT_MODELS)
  const [status, setStatus] = useState({ label: '확인 중...', account_label: '' })
  const [error, setError] = useState('')
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    Promise.allSettled([getSettings(), getModels(), getEngineStatus()]).then(
      ([settingsResult, modelsResult, statusResult]) => {
        const s = settingsResult.status === 'fulfilled' ? settingsResult.value : DEFAULT_SETTINGS
        const m = modelsResult.status === 'fulfilled' ? modelsResult.value : DEFAULT_MODELS
        const st = statusResult.status === 'fulfilled'
          ? statusResult.value
          : { status: 'server_error', label: '백엔드 연결 실패', account_label: '' }
        const failed = [settingsResult, modelsResult, statusResult].some(result => result.status === 'rejected')
        if (failed) {
          setError('백엔드 서버(127.0.0.1:8200)에 연결할 수 없습니다. start.ps1로 백엔드와 프론트를 함께 실행했는지 확인해 주세요.')
        }
        const validModels = m[s.engine] || []
        const first = validModels[0] || ''
        const sanitized = { ...s }
        if (!['mixed', 'hybrid', 'per_doc'].includes(sanitized.comparison_mode)) {
          sanitized.comparison_mode = 'mixed'
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

  useEffect(() => {
    if (!settings) return
    let cancelled = false
    setStatus(prev => ({ ...prev, label: '확인 중...', detail: '' }))
    const timer = setTimeout(() => {
      getEngineStatus(settings)
        .then(st => {
          if (!cancelled) setStatus(st)
        })
        .catch(() => {
          if (!cancelled) {
            setStatus({ status: 'server_error', label: '백엔드 연결 실패', account_label: '' })
          }
        })
    }, 400)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [settings?.engine, settings?.model_parser])

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
    try {
      await saveSettings(settings)
      const st = await getEngineStatus(settings)
      setStatus(st)
      setError('')
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (_) {
      setError('설정 저장 실패: 백엔드 서버(127.0.0.1:8200)에 연결할 수 없습니다.')
    }
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
              {error && (
                <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-xs leading-relaxed text-red-700">
                  {error}
                </div>
              )}

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
                {status.checked_model && (
                  <div className="mt-1 text-xs text-gray-500">
                    확인 모델: <span className="font-medium text-gray-700">{status.checked_model}</span>
                  </div>
                )}
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
                      value: 'mixed',
                      title: '혼합 모드',
                      description: '기본값. 관련 문단을 먼저 압축한 뒤 전체 문헌을 한 번에 비교합니다.',
                      caution: '빠른 후보 검토와 보고서 품질의 균형',
                    },
                    {
                      value: 'hybrid',
                      title: '정밀 모드',
                      description: '기존 통합 비교 방식. 한도 이내이면 인용발명 전문을 그대로 비교합니다.',
                      caution: '정밀하지만 본문이 길면 시간이 오래 걸림',
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
                        <div className={`text-sm font-semibold ${selected ? 'text-blue-700' : 'text-gray-700'}`}>
                          {option.title}
                        </div>
                        <div className="mt-1 text-xs leading-relaxed text-gray-600">
                          {option.description}
                        </div>
                        <div className="mt-1 text-xs leading-relaxed text-amber-700">
                          {option.caution}
                        </div>
                      </label>
                    )
                  })}
                </div>
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
                  comparison_mode: 'mixed',
                  model_parser: '',
                  model_compare: 'claude-sonnet-4-6',
                  model_report: '',
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
