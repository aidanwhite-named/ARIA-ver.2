import { useState, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getGapElements, webSearchGap, getSearchStrategy, generateSearchStrategy } from '../api/client'

async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

function GapSearchSection({ gap, gapResult, gapSearching, onGenerate, copiedTerm, onCopy }) {
  if (!gap) return null

  if (!gap.analyzed) {
    return (
      <div className="mt-4 border-t pt-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">보완문헌 검색</span>
        </div>
        <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg text-xs text-gray-500 leading-relaxed">
          구성요소 대비 분석이 있어야 미개시 구성요소를 기준으로 보완문헌 검색을 진행할 수 있습니다.
        </div>
      </div>
    )
  }

  if (gap.uncovered.length === 0) {
    return (
      <div className="mt-4 p-3 bg-emerald-50 border border-emerald-100 rounded-lg text-xs text-emerald-700">
        모든 구성요소가 인용발명에서 확인되어 보완문헌 검색이 필요하지 않습니다.
      </div>
    )
  }

  const stars = (importance) => {
    const n = Math.max(1, Math.min(5, parseInt(importance, 10) || 3))
    return '★'.repeat(n)
  }

  return (
    <div className="mt-4 border-t pt-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">보완문헌 검색</span>
        <span className="text-[10px] text-amber-600 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded-full">
          미커버 {gap.uncovered.length}개
        </span>
      </div>

      <div className="mb-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
        <p className="text-xs text-amber-800 font-medium mb-2">
          아래 구성요소는 현재 인용발명에서 직접 확인되지 않아 추가 보완문헌 검토가 필요합니다.
        </p>
        <p className="text-[11px] text-amber-700 mb-2">
          필요하면 이 목록을 기준으로 LLM 보완검색을 실행해 관련 문헌을 더 찾아볼 수 있습니다.
        </p>
        {gap.uncovered.map((item, index) => (
          <div key={index} className="text-xs text-amber-900 mb-1.5 leading-relaxed">
            <span className="font-semibold">({item.label})</span>
            <span className="text-amber-500 ml-1">{stars(item.importance)}</span>
            <span className="ml-1">{item.text.length > 90 ? `${item.text.slice(0, 90)}...` : item.text}</span>
            <span className="text-amber-600 ml-1">최고 판정: {item.best_judgment}</span>
          </div>
        ))}
      </div>

      <button
        onClick={onGenerate}
        disabled={gapSearching}
        className={`w-full flex items-center justify-center gap-1.5 text-xs px-3 py-2 rounded-lg border transition font-medium disabled:opacity-40 ${
          gapResult
            ? 'bg-emerald-50 border-emerald-200 text-emerald-700 hover:bg-emerald-100'
            : 'bg-amber-500 border-amber-500 text-white hover:bg-amber-600'
        }`}
      >
        {gapSearching ? (
          <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
        ) : '🔎'}
        {gapSearching ? '보완검색 실행 중... (수 분 소요될 수 있음)' : gapResult ? '보완문헌 검색 완료 (재실행 가능)' : '보완문헌 검색 실행 (LLM)'}
      </button>

      {gapResult && (
        <div className="mt-3">
          {(gapResult.results || []).map((item, index) => (
            <div key={index} className="border-l-4 border-l-amber-400 pl-3 py-2 bg-white rounded-r-lg border border-l-0 border-gray-100 mb-2">
              <div className="text-sm font-semibold text-gray-900 mb-1.5">
                ({item.label}) {item.feature_ko}
              </div>

              {(item.documents || []).length > 0 ? item.documents.map((doc, docIndex) => (
                <div key={docIndex} className="mb-2 px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg">
                  <div className="flex items-center gap-1.5 flex-wrap mb-0.5">
                    {doc.url ? (
                      <a
                        href={doc.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-xs font-semibold text-blue-700 hover:underline break-all"
                      >
                        {doc.title || doc.url}
                      </a>
                    ) : (
                      <span className="text-xs font-semibold text-gray-800">{doc.title}</span>
                    )}
                    {doc.number && (
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded border font-mono cursor-pointer transition ${
                          copiedTerm === doc.number ? 'bg-green-50 border-green-300 text-green-700' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                        }`}
                        onClick={() => onCopy(doc.number)}
                        title="번호 복사"
                      >
                        {doc.number}
                      </span>
                    )}
                  </div>
                  {doc.summary && <div className="text-xs text-slate-600 leading-relaxed">{doc.summary}</div>}
                  {doc.verification_reason && <div className="mt-1 text-[11px] text-slate-500 leading-relaxed">검증: {doc.verification_reason}</div>}
                  {doc.verification_quote && <div className="mt-1 text-[11px] text-slate-500 leading-relaxed">근거 문구: {doc.verification_quote}</div>}
                </div>
              )) : (
                <div className="text-xs text-gray-400 mb-2">적합한 문헌을 찾지 못했습니다.</div>
              )}

              {(item.queries_used || []).length > 0 && (
                <div className="flex items-start gap-1.5 flex-wrap">
                  <span className="text-[10px] text-gray-400 font-medium pt-0.5 shrink-0">검색어</span>
                  {item.queries_used.map((query, queryIndex) => (
                    <span
                      key={queryIndex}
                      className={`text-[10px] px-1.5 py-0.5 rounded border cursor-pointer transition ${
                        copiedTerm === query ? 'bg-green-50 border-green-300 text-green-700' : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'
                      }`}
                      onClick={() => onCopy(query)}
                      title="복사"
                    >
                      {query}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}

          {gapResult.error && (
            <div className="mt-2 text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded p-2">
              일부 LLM 응답 파싱에 문제가 있어 다시 실행이 필요할 수 있습니다.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function KeywordPanel({ jobId, claimNumber, isVisible }) {
  const [error, setError] = useState('')
  const [copiedTerm, setCopiedTerm] = useState('')
  const [gap, setGap] = useState(null)
  const [gapResult, setGapResult] = useState(null)
  const [gapSearching, setGapSearching] = useState(false)
  const [strategy, setStrategy] = useState(null)
  const [strategyLoading, setStrategyLoading] = useState(false)
  const [strategyCopied, setStrategyCopied] = useState(false)

  const loadPanelData = useCallback(async () => {
    if (!jobId || !claimNumber) return
    setError('')
    setGap(null)
    setGapResult(null)
    try {
      setGap(await getGapElements(jobId, claimNumber))
    } catch (e) {
      setGap(null)
      setError(e.message)
    }
    try {
      const result = await getSearchStrategy(jobId, claimNumber)
      setStrategy(result.exists ? result.strategy_md : null)
    } catch {
      setStrategy(null)
    }
  }, [jobId, claimNumber])

  useEffect(() => {
    if (isVisible && jobId && claimNumber) {
      loadPanelData()
    }
  }, [isVisible, jobId, claimNumber, loadPanelData])

  async function handleGenerateStrategy() {
    if (!jobId || !claimNumber) return
    setStrategyLoading(true)
    setError('')
    try {
      const result = await generateSearchStrategy(jobId, claimNumber)
      setStrategy(result.strategy_md)
    } catch (e) {
      setError(`검색 전략 생성 실패: ${e.message}`)
    } finally {
      setStrategyLoading(false)
    }
  }

  async function handleGapSearch() {
    if (!jobId || !claimNumber) return
    setGapSearching(true)
    setError('')
    try {
      const result = await webSearchGap(jobId, claimNumber)
      setGapResult(result)
    } catch (e) {
      setError(`보완문헌 검색 실패: ${e.message}`)
    } finally {
      setGapSearching(false)
    }
  }

  async function handleCopyStrategy() {
    if (!strategy) return
    await copyToClipboard(strategy)
    setStrategyCopied(true)
    setTimeout(() => setStrategyCopied(false), 2000)
  }

  async function handleCopyTerm(term) {
    await copyToClipboard(term)
    setCopiedTerm(term)
    setTimeout(() => setCopiedTerm(''), 1500)
  }

  if (!jobId || !claimNumber) {
    return (
      <div className="h-full flex items-center justify-center text-gray-400 text-sm flex-col gap-2">
        <span className="text-3xl">🔎</span>
        <p>청구항을 등록하면 검색 전략을 확인할 수 있습니다.</p>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-3 border-b bg-gray-50 shrink-0 flex-wrap">
        <button
          onClick={loadPanelData}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition font-medium text-gray-700"
        >
          ↻
          다시 불러오기
        </button>

        <button
          onClick={handleGenerateStrategy}
          disabled={strategyLoading}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition font-medium disabled:opacity-40 ${
            strategy
              ? 'bg-white border-gray-200 text-gray-700 hover:bg-gray-50'
              : 'bg-indigo-500 border-indigo-500 text-white hover:bg-indigo-600'
          }`}
        >
          {strategyLoading ? <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" /> : '🧠'}
          {strategyLoading ? '검색 전략 생성 중... (1~3분)' : strategy ? '검색 전략 재생성' : '검색 전략 생성 (LLM)'}
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3">
        {error && (
          <div className="mb-3 text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
            {error}
          </div>
        )}

        <GapSearchSection
          gap={gap}
          gapResult={gapResult}
          gapSearching={gapSearching}
          onGenerate={handleGapSearch}
          copiedTerm={copiedTerm}
          onCopy={handleCopyTerm}
        />

        {strategy && (
          <div className="mt-4 border-t pt-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">LLM 검색 전략</span>
              <div className="flex-1" />
              <button
                onClick={handleCopyStrategy}
                className={`flex items-center gap-1 text-[11px] px-2 py-1 rounded-lg border transition font-medium ${
                  strategyCopied ? 'bg-green-50 border-green-300 text-green-700' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'
                }`}
              >
                {strategyCopied ? '✅ 복사됨' : '📋 전체 복사'}
              </button>
            </div>
            <div className="strategy-content px-3 py-2 bg-white border border-gray-200 rounded-lg">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{strategy}</ReactMarkdown>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
