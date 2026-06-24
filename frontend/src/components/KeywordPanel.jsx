/**
 * KeywordPanel.jsx — 특허 검색 키워드 패널
 *
 * 보고서 영역의 "📎 키워드" 탭 콘텐츠.
 * - 기본 키워드: 백엔드 로컬 추출 (LLM 없음, 즉시)
 * - 보완문서 웹검색: 구성대비 후 미커버 구성요소를 LLM이 웹검색으로 직접 탐색
 */
import { useState, useEffect, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getKeywords, getGapElements, webSearchGap, getSearchStrategy, generateSearchStrategy } from '../api/client'

// ── 복사 유틸 ────────────────────────────────────────────────────────────────
async function copyToClipboard(text) {
  try {
    await navigator.clipboard.writeText(text)
    return true
  } catch {
    return false
  }
}

// ── 검색식 생성 (핵심 키워드 → AND 묶음) ────────────────────────────────────
function buildSearchQuery(keywords) {
  const core = keywords.filter(k => k.importance === 'core').map(k => k.term)
  if (core.length === 0) return keywords.slice(0, 5).map(k => k.term).join(' AND ')
  return core.length > 1 ? core.map(t => `"${t}"`).join(' AND ') : `"${core[0]}"`
}

// ── 키워드 태그 ──────────────────────────────────────────────────────────────
function KeywordTag({ term, importance, onRemove, onClick, copied }) {
  const base = importance === 'core'
    ? 'bg-blue-50 text-blue-800 border-blue-200 hover:bg-blue-100'
    : 'bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100'
  return (
    <span
      className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full border text-xs font-medium cursor-pointer transition-all select-none ${base} ${copied ? 'ring-2 ring-green-400' : ''}`}
      title="클릭하여 복사"
      onClick={onClick}
    >
      {term}
      {onRemove && (
        <button
          className="ml-0.5 text-gray-400 hover:text-red-500 leading-none transition"
          onClick={e => { e.stopPropagation(); onRemove() }}
          title="제거"
        >×</button>
      )}
    </span>
  )
}

// ── 보완문서 검색 섹션 ───────────────────────────────────────────────────────
function GapSearchSection({ gap, gapResult, gapSearching, onGenerate, copiedTerm, onCopy }) {
  if (!gap) return null

  // 구성대비 미실행 — 기능 안내만 표시
  if (!gap.analyzed) {
    return (
      <div className="mt-4 border-t pt-4">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">보완문서 웹검색</span>
        </div>
        <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg text-xs text-gray-500 leading-relaxed">
          ℹ️ 구성대비 분석을 먼저 실행하면, 어떤 인용발명도 개시하지 못한 구성요소를 찾아
          해당 구성과 유사한 발명을 웹검색으로 탐색할 수 있습니다.
        </div>
      </div>
    )
  }

  if (gap.uncovered.length === 0) {
    return (
      <div className="mt-4 p-3 bg-emerald-50 border border-emerald-100 rounded-lg text-xs text-emerald-700">
        ✅ 모든 구성요소가 인용발명에서 확인되어 보완문서 검색이 필요하지 않습니다.
      </div>
    )
  }

  const stars = (imp) => {
    const n = Math.max(1, Math.min(5, parseInt(imp, 10) || 3))
    return '★'.repeat(n)
  }

  return (
    <div className="mt-4 border-t pt-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">보완문서 웹검색</span>
        <span className="text-[10px] text-amber-600 bg-amber-50 border border-amber-200 px-1.5 py-0.5 rounded-full">
          미커버 {gap.uncovered.length}개
        </span>
      </div>

      {/* 미커버 구성요소 목록 (LLM 없음, 즉시) */}
      <div className="mb-3 p-3 bg-amber-50 border border-amber-200 rounded-lg">
        <p className="text-xs text-amber-800 font-medium mb-2">
          ⚠️ 어떤 인용발명도 개시하지 못한 구성요소 — 진보성 부정에 보완문헌이 필요합니다.
        </p>
        {gap.uncovered.map((u, i) => (
          <div key={i} className="text-xs text-amber-900 mb-1.5 leading-relaxed">
            <span className="font-semibold">({u.label})</span>
            <span className="text-amber-500 ml-1">{stars(u.importance)}</span>
            <span className="ml-1">{u.text.length > 90 ? u.text.slice(0, 90) + '…' : u.text}</span>
            <span className="text-amber-600 ml-1">— 최고 판정: {u.best_judgment}</span>
          </div>
        ))}
      </div>

      {/* 웹검색 실행 버튼 */}
      <button
        onClick={onGenerate}
        disabled={gapSearching}
        className={`w-full flex items-center justify-center gap-1.5 text-xs px-3 py-2 rounded-lg border transition font-medium disabled:opacity-40
          ${gapResult
            ? 'bg-emerald-50 border-emerald-200 text-emerald-700 hover:bg-emerald-100'
            : 'bg-amber-500 border-amber-500 text-white hover:bg-amber-600'
          }`}
      >
        {gapSearching ? (
          <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
        ) : '🔎'}
        {gapSearching
          ? '웹검색 수행 중... (수 분 소요될 수 있음)'
          : gapResult ? '보완문서 웹검색 완료 (재실행)' : '보완문서 웹검색 실행 (LLM 웹검색)'}
      </button>

      {/* 검색 결과 */}
      {gapResult && (
        <div className="mt-3">
          {(gapResult.results || []).map((t, i) => (
            <div key={i} className="border-l-4 border-l-amber-400 pl-3 py-2 bg-white rounded-r-lg border border-l-0 border-gray-100 mb-2">
              <div className="text-sm font-semibold text-gray-900 mb-1.5">
                ({t.label}) {t.feature_ko}
              </div>

              {/* 후보 문헌 */}
              {(t.documents || []).length > 0 ? t.documents.map((doc, j) => (
                <div key={j} className="mb-2 px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg">
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
                        className={`text-[10px] px-1.5 py-0.5 rounded border font-mono cursor-pointer transition ${copiedTerm === doc.number ? 'bg-green-50 border-green-300 text-green-700' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'}`}
                        onClick={() => onCopy(doc.number)}
                        title="번호 복사"
                      >
                        {doc.number}
                      </span>
                    )}
                    {doc.relevance && (
                      <span className={`text-[10px] px-1.5 py-0.5 rounded-full font-semibold
                        ${doc.relevance === 'high' ? 'bg-red-50 text-red-600 border border-red-200'
                          : doc.relevance === 'medium' ? 'bg-amber-50 text-amber-600 border border-amber-200'
                          : 'bg-gray-50 text-gray-500 border border-gray-200'}`}>
                        {doc.relevance === 'high' ? '관련성 높음' : doc.relevance === 'medium' ? '관련성 중간' : '관련성 낮음'}
                      </span>
                    )}
                  </div>
                  {doc.summary && (
                    <div className="text-xs text-slate-600 leading-relaxed">{doc.summary}</div>
                  )}
                </div>
              )) : (
                <div className="text-xs text-gray-400 mb-2">적합한 문헌을 찾지 못했습니다.</div>
              )}

              {/* 사용한 검색어 */}
              {(t.queries_used || []).length > 0 && (
                <div className="flex items-start gap-1.5 flex-wrap">
                  <span className="text-[10px] text-gray-400 font-medium pt-0.5 shrink-0">검색어</span>
                  {t.queries_used.map((q, j) => (
                    <span
                      key={j}
                      className={`text-[10px] px-1.5 py-0.5 rounded border cursor-pointer transition ${copiedTerm === q ? 'bg-green-50 border-green-300 text-green-700' : 'bg-white border-gray-200 text-gray-500 hover:bg-gray-50'}`}
                      onClick={() => onCopy(q)}
                      title="복사"
                    >
                      {q}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}

          {gapResult.error && (
            <div className="mt-2 text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded p-2">
              ⚠️ LLM 응답 파싱 오류 — 다시 실행해주세요.
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ── 메인 패널 ────────────────────────────────────────────────────────────────
export default function KeywordPanel({ jobId, claimNumber, isVisible }) {
  const [keywords, setKeywords] = useState([])        // 로컬 추출 결과
  const [loading, setLoading] = useState(false)       // 로컬 로딩
  const [error, setError] = useState('')
  const [copiedTerm, setCopiedTerm] = useState('')    // 방금 복사한 용어
  const [queryCopied, setQueryCopied] = useState(false)
  const [removedTerms, setRemovedTerms] = useState(new Set())
  const [gap, setGap] = useState(null)                // 미커버 구성요소 (LLM 없음)
  const [gapResult, setGapResult] = useState(null)    // 보완문서 웹검색 결과 (LLM 1회)
  const [gapSearching, setGapSearching] = useState(false)
  const [strategy, setStrategy] = useState(null)      // LLM 검색 전략 (마크다운)
  const [strategyLoading, setStrategyLoading] = useState(false)
  const [strategyCopied, setStrategyCopied] = useState(false)

  // ── 로컬 키워드 로드 ────────────────────────────────────────────────────
  const loadLocalKeywords = useCallback(async () => {
    if (!jobId || !claimNumber) return
    setLoading(true)
    setError('')
    setRemovedTerms(new Set())
    setGap(null)
    setGapResult(null)
    try {
      const result = await getKeywords(jobId, claimNumber)
      setKeywords(result.keywords || [])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
    // 미커버 구성요소 조회 (LLM 없음) — 실패해도 키워드 기능에 영향 없음
    try {
      setGap(await getGapElements(jobId, claimNumber))
    } catch {
      setGap(null)
    }
    // 캐시된 검색 전략 조회 (LLM 없음) — 청구항 전환 시 해당 청구항 캐시로 교체
    try {
      const s = await getSearchStrategy(jobId, claimNumber)
      setStrategy(s.exists ? s.strategy_md : null)
    } catch {
      setStrategy(null)
    }
  }, [jobId, claimNumber])

  useEffect(() => {
    if (isVisible && jobId && claimNumber) {
      loadLocalKeywords()
    }
  }, [isVisible, jobId, claimNumber, loadLocalKeywords])

  // ── 검색 전략 생성 (LLM 1회) ────────────────────────────────────────────
  async function handleGenerateStrategy() {
    if (!jobId || !claimNumber) return
    setStrategyLoading(true)
    setError('')
    try {
      const s = await generateSearchStrategy(jobId, claimNumber)
      setStrategy(s.strategy_md)
    } catch (e) {
      setError(`검색 전략 생성 실패: ${e.message}`)
    } finally {
      setStrategyLoading(false)
    }
  }

  async function handleCopyStrategy() {
    if (!strategy) return
    await copyToClipboard(strategy)
    setStrategyCopied(true)
    setTimeout(() => setStrategyCopied(false), 2000)
  }

  // ── 보완문서 웹검색 실행 (LLM 1회) ──────────────────────────────────────
  async function handleGapSearch() {
    if (!jobId || !claimNumber) return
    setGapSearching(true)
    setError('')
    try {
      const result = await webSearchGap(jobId, claimNumber)
      setGapResult(result)
    } catch (e) {
      setError(`보완문서 웹검색 실패: ${e.message}`)
    } finally {
      setGapSearching(false)
    }
  }

  // ── 복사 ────────────────────────────────────────────────────────────────
  async function handleCopyTerm(term) {
    await copyToClipboard(term)
    setCopiedTerm(term)
    setTimeout(() => setCopiedTerm(''), 1500)
  }

  async function handleCopyQuery() {
    const visible = keywords.filter(k => !removedTerms.has(k.term))
    const q = buildSearchQuery(visible)
    await copyToClipboard(q)
    setQueryCopied(true)
    setTimeout(() => setQueryCopied(false), 2000)
  }

  // ── 필터된 키워드 ────────────────────────────────────────────────────────
  const visibleKeywords = keywords.filter(k => !removedTerms.has(k.term))
  const coreKws = visibleKeywords.filter(k => k.importance === 'core')
  const secondaryKws = visibleKeywords.filter(k => k.importance === 'secondary')
  const searchQuery = buildSearchQuery(visibleKeywords)

  // ── 상태: 데이터 없음 ────────────────────────────────────────────────────
  if (!jobId || !claimNumber) {
    return (
      <div className="h-full flex items-center justify-center text-gray-400 text-sm flex-col gap-2">
        <span className="text-3xl">🔍</span>
        <p>청구항을 등록하면 검색 키워드를 추출합니다.</p>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col overflow-hidden">
      {/* 툴바 */}
      <div className="flex items-center gap-2 px-4 py-3 border-b bg-gray-50 shrink-0 flex-wrap">
        <button
          onClick={loadLocalKeywords}
          disabled={loading}
          className="flex items-center gap-1.5 text-xs px-3 py-1.5 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition disabled:opacity-40 font-medium text-gray-700"
        >
          {loading ? (
            <span className="w-3 h-3 border border-gray-400 border-t-transparent rounded-full animate-spin" />
          ) : '🔄'}
          다시 추출
        </button>

        {/* 검색 전략 생성 (LLM 1회) */}
        <button
          onClick={handleGenerateStrategy}
          disabled={strategyLoading}
          className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition font-medium disabled:opacity-40
            ${strategy
              ? 'bg-white border-gray-200 text-gray-700 hover:bg-gray-50'
              : 'bg-indigo-500 border-indigo-500 text-white hover:bg-indigo-600'}`}
        >
          {strategyLoading ? (
            <span className="w-3 h-3 border border-current border-t-transparent rounded-full animate-spin" />
          ) : '🧠'}
          {strategyLoading
            ? '검색 전략 생성 중... (1~3분)'
            : strategy ? '검색 전략 재생성' : '검색 전략 생성 (LLM)'}
        </button>

        <div className="flex-1" />

        {/* 검색식 복사 */}
        {searchQuery && (
          <button
            onClick={handleCopyQuery}
            className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border transition font-medium
              ${queryCopied ? 'bg-green-50 border-green-300 text-green-700' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'}`}
          >
            {queryCopied ? '✅ 복사됨' : '📋 검색식 복사'}
          </button>
        )}
      </div>


      {/* 검색식 미리보기 */}
      {searchQuery && (
        <div
          className="mx-4 mt-3 mb-1 px-3 py-2 bg-slate-50 border border-slate-200 rounded-lg cursor-pointer hover:bg-slate-100 transition shrink-0"
          onClick={handleCopyQuery}
          title="클릭하여 복사"
        >
          <div className="text-[10px] text-slate-400 font-medium mb-0.5">검색식</div>
          <div className="text-xs text-slate-700 font-mono break-all leading-relaxed">
            {searchQuery}
          </div>
        </div>
      )}

      {/* 본문 */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {/* 에러 */}
        {error && (
          <div className="mb-3 text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
            {error}
          </div>
        )}

        {/* 미커버 구성이 있으면 보완검색을 가장 먼저 안내한다. */}
        {!loading && (
          <GapSearchSection
            gap={gap}
            gapResult={gapResult}
            gapSearching={gapSearching}
            onGenerate={handleGapSearch}
            copiedTerm={copiedTerm}
            onCopy={handleCopyTerm}
          />
        )}

        {/* 로딩 */}
        {loading && (
          <div className="flex items-center gap-2 text-blue-500 text-sm py-8 justify-center">
            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
            키워드 추출 중...
          </div>
        )}

        {/* 로컬 키워드 태그 */}
        {!loading && keywords.length > 0 ? (
          <div>
            {/* 핵심 키워드 */}
            {coreKws.length > 0 && (
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">핵심 키워드</span>
                  <span className="text-[10px] text-blue-500 bg-blue-50 border border-blue-200 px-1.5 py-0.5 rounded-full">
                    {coreKws.length}개
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {coreKws.map((kw, i) => (
                    <KeywordTag
                      key={i}
                      term={kw.term}
                      importance="core"
                      copied={copiedTerm === kw.term}
                      onClick={() => handleCopyTerm(kw.term)}
                      onRemove={() => setRemovedTerms(prev => new Set([...prev, kw.term]))}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* 부가 키워드 */}
            {secondaryKws.length > 0 && (
              <div className="mb-4">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">부가 키워드</span>
                  <span className="text-[10px] text-gray-400 bg-gray-100 border border-gray-200 px-1.5 py-0.5 rounded-full">
                    {secondaryKws.length}개
                  </span>
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {secondaryKws.map((kw, i) => (
                    <KeywordTag
                      key={i}
                      term={kw.term}
                      importance="secondary"
                      copied={copiedTerm === kw.term}
                      onClick={() => handleCopyTerm(kw.term)}
                      onRemove={() => setRemovedTerms(prev => new Set([...prev, kw.term]))}
                    />
                  ))}
                </div>
              </div>
            )}

          </div>
        ) : !loading && keywords.length === 0 && !error ? (
          <div className="flex flex-col items-center justify-center py-12 text-gray-400">
            <span className="text-4xl mb-3">🔍</span>
            <p className="text-sm">키워드가 없습니다.</p>
            <p className="text-xs mt-1">청구항 구성요소가 추출되었는지 확인해주세요.</p>
          </div>
        ) : null}

        {/* LLM 검색 전략 — 구성 분해·키워드 확장·DB별 검색식 */}
        {!loading && strategy && (
          <div className="mt-4 border-t pt-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-semibold text-gray-500 uppercase tracking-wide">LLM 검색 전략</span>
              <div className="flex-1" />
              <button
                onClick={handleCopyStrategy}
                className={`flex items-center gap-1 text-[11px] px-2 py-1 rounded-lg border transition font-medium
                  ${strategyCopied ? 'bg-green-50 border-green-300 text-green-700' : 'bg-white border-gray-200 text-gray-600 hover:bg-gray-50'}`}
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
