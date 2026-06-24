import { useEffect, useState } from 'react'
import { enhanceClaim, enhancePurpose, getClaimTree } from '../api/client'

// ─── 중요도 별점 ────────────────────────────────────────────────────────────
function StarRating({ importance }) {
  const n = Math.min(5, Math.max(1, parseInt(importance, 10) || 3))
  return (
    <span className="text-xs shrink-0" title={`중요도 ${n}/5`}>
      <span className="text-amber-400">{'★'.repeat(n)}</span>
      <span className="text-gray-200">{'★'.repeat(5 - n)}</span>
    </span>
  )
}

// ─── 구성요소 레이블 배지 ───────────────────────────────────────────────────
function LabelBadge({ label }) {
  if (label === '_') return null
  const isSub = label.includes('-')
  return (
    <span className={`inline-flex items-center justify-center min-w-[2rem] h-5 rounded text-xs font-bold shrink-0 ${
      isSub
        ? 'bg-purple-100 text-purple-700 border border-purple-300'
        : 'bg-blue-100 text-blue-700 border border-blue-300'
    }`}>
      ({label})
    </span>
  )
}

// ─── split_method 배지 ──────────────────────────────────────────────────────
function SplitBadge({ method }) {
  if (method === 'llm') return (
    <span className="text-xs bg-green-100 text-green-700 border border-green-300 rounded px-1.5 py-0.5 ml-1">
      AI 분석
    </span>
  )
  if (method === 'fallback') return (
    <span className="text-xs bg-orange-100 text-orange-700 border border-orange-300 rounded px-1.5 py-0.5 ml-1">
      단순 분할
    </span>
  )
  if (method === 'labeled') return (
    <span className="text-xs bg-gray-100 text-gray-500 border border-gray-200 rounded px-1.5 py-0.5 ml-1">
      라벨 감지
    </span>
  )
  return null
}

// ─── 목적/효과 섹션 ─────────────────────────────────────────────────────────
function PurposeSection({ jobId, purposeEffects, onUpdate }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const { purpose, effects, extracted_by: extractedBy } = purposeEffects || {}

  async function handleEnhance() {
    setLoading(true)
    setError('')
    try {
      const result = await enhancePurpose(jobId)
      onUpdate(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="bg-blue-50 border border-blue-200 rounded-xl p-5 mb-6">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-bold text-blue-900">발명의 목적 및 효과</h2>
        {extractedBy !== 'llm' && (
          <button
            onClick={handleEnhance}
            disabled={loading}
            className="flex items-center gap-1.5 text-xs bg-blue-600 text-white rounded-lg px-3 py-1.5 hover:bg-blue-700 disabled:opacity-50 transition"
          >
            {loading ? (
              <>
                <span className="w-3 h-3 border-2 border-white border-t-transparent rounded-full animate-spin" />
                AI 분석 중…
              </>
            ) : (
              <>✦ AI로 분석하기</>
            )}
          </button>
        )}
        {extractedBy === 'llm' && (
          <span className="text-xs text-green-600 font-medium">✓ AI 분석 완료</span>
        )}
      </div>

      {error && (
        <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded p-2 mb-2">{error}</div>
      )}

      {extractedBy === 'pending' && !loading && (
        <p className="text-sm text-blue-600 italic">
          AI 분석 버튼을 클릭하면 청구항을 바탕으로 목적과 효과를 분석합니다.
        </p>
      )}

      {purpose && (
        <div className="mb-2">
          <span className="text-xs font-semibold text-blue-800 uppercase tracking-wide">목적</span>
          <p className="text-sm text-gray-700 mt-1 leading-relaxed">{purpose}</p>
        </div>
      )}
      {effects && (
        <div>
          <span className="text-xs font-semibold text-blue-800 uppercase tracking-wide">효과</span>
          <p className="text-sm text-gray-700 mt-1 leading-relaxed">{effects}</p>
        </div>
      )}
    </div>
  )
}

// ─── 단일 청구항 카드 ────────────────────────────────────────────────────────
function ClaimCard({ claim, samePairs, jobId, onClaimUpdate, groupColor }) {
  const [expanded, setExpanded] = useState(true)
  const [enhancing, setEnhancing] = useState(false)
  const [error, setError] = useState('')

  const num = claim.claim_number
  const isIndependent = claim.claim_type === 'independent'
  const sameAs = samePairs?.[String(num)]  // 이 청구항이 동일한 원본 청구항 번호
  const isFallback = claim.split_method === 'fallback'

  async function handleEnhance() {
    setEnhancing(true)
    setError('')
    try {
      const result = await enhanceClaim(jobId, num)
      onClaimUpdate(result)
    } catch (e) {
      setError(e.message)
    } finally {
      setEnhancing(false)
    }
  }

  const borderColor = groupColor ? `border-l-4 ${groupColor}` : 'border-l-4 border-transparent'

  return (
    <div className={`bg-white border border-gray-200 rounded-xl shadow-sm mb-3 overflow-hidden ${borderColor}`}>
      {/* 헤더 */}
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-gray-50 transition"
        onClick={() => setExpanded(e => !e)}
      >
        <div className={`flex items-center justify-center w-7 h-7 rounded-full text-xs font-bold shrink-0 ${
          isIndependent ? 'bg-blue-600 text-white' : 'bg-gray-300 text-gray-700'
        }`}>
          {num}
        </div>

        <div className="flex-1 min-w-0 flex items-center gap-2 flex-wrap">
          <span className={`text-sm font-semibold ${isIndependent ? 'text-gray-900' : 'text-gray-600'}`}>
            {isIndependent ? '독립항' : `종속항 (제${claim.parent_claim}항)`}
          </span>
          {sameAs !== undefined && (
            <span className="text-xs bg-purple-100 text-purple-700 border border-purple-300 rounded px-2 py-0.5">
              ↔ 제{sameAs}항과 카테고리 동일
            </span>
          )}
          <SplitBadge method={claim.split_method} />
          {claim.elements?.length > 0 && (
            <span className="text-xs text-gray-400">구성요소 {claim.elements.filter(e => e.label !== '_').length}개</span>
          )}
        </div>

        <span className="text-gray-400 text-xs shrink-0">{expanded ? '▲' : '▼'}</span>
      </div>

      {/* 본문 */}
      {expanded && (
        <div className="px-4 pb-4 space-y-2">
          {/* 어두 */}
          {claim.preamble && (
            <div className="text-xs text-gray-500 italic border-b pb-2 mb-2">
              <span className="font-semibold text-gray-600 not-italic">어두: </span>
              {claim.preamble},
            </div>
          )}

          {/* 구성요소 */}
          {claim.elements?.length > 0 ? (
            <div className="space-y-1.5">
              {claim.elements.map((elem, i) => (
                <div key={`${elem.label}-${i}`} className={`flex items-start gap-2 ${
                  elem.is_sub ? 'pl-5 border-l-2 border-purple-200 ml-3' : ''
                }`}>
                  <LabelBadge label={elem.label} />
                  {elem.label !== '_' && <StarRating importance={elem.importance} />}
                  <span className="text-xs text-gray-700 leading-relaxed flex-1">{elem.text}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-gray-400 italic">구성요소 없음</p>
          )}

          {/* 어미 */}
          {claim.closing && (
            <div className="text-xs text-gray-500 italic border-t pt-2 mt-2">
              <span className="font-semibold text-gray-600 not-italic">어미: </span>
              {claim.closing}
            </div>
          )}

          {/* fallback → AI 개선 버튼 */}
          {isFallback && !enhancing && (
            <div className="pt-2 border-t mt-2">
              <p className="text-xs text-orange-600 mb-2">
                구성요소를 자동으로 분리하지 못했습니다. AI로 개선할 수 있습니다.
              </p>
              <button
                onClick={handleEnhance}
                className="text-xs bg-orange-500 text-white rounded-lg px-3 py-1.5 hover:bg-orange-600 transition"
              >
                ✦ AI로 구성요소 분해
              </button>
            </div>
          )}
          {enhancing && (
            <div className="flex items-center gap-2 text-xs text-orange-600 pt-2">
              <span className="w-3 h-3 border-2 border-orange-500 border-t-transparent rounded-full animate-spin" />
              AI 분석 중…
            </div>
          )}
          {error && (
            <p className="text-xs text-red-500 pt-1">{error}</p>
          )}
        </div>
      )}
    </div>
  )
}

// ─── 청구항 트리 (재귀) ──────────────────────────────────────────────────────
function ClaimTree({ claims, samePairs, jobId, onClaimUpdate }) {
  if (!claims || claims.length === 0) {
    return (
      <div className="text-center text-gray-400 py-12 text-sm">
        청구항이 없습니다.
      </div>
    )
  }

  // 같은 same_pairs 그룹에 동일한 색상 부여
  const GROUP_COLORS = [
    'border-blue-400', 'border-purple-400', 'border-green-400',
    'border-orange-400', 'border-pink-400', 'border-teal-400',
  ]
  const colorMap = {}
  let colorIdx = 0
  const groupRoots = new Set(Object.values(samePairs || {}))
  groupRoots.forEach(rootNum => {
    colorMap[rootNum] = GROUP_COLORS[colorIdx % GROUP_COLORS.length]
    colorIdx++
  })
  Object.entries(samePairs || {}).forEach(([k, v]) => {
    colorMap[parseInt(k)] = colorMap[v]
  })

  const independents = claims.filter(c => c.claim_type === 'independent')
  const dependentsOf = (num) => claims.filter(c => c.parent_claim === num)

  function renderClaim(claim, depth = 0) {
    const num = claim.claim_number
    const children = dependentsOf(num)
    return (
      <div key={num} style={{ paddingLeft: depth > 0 ? 24 : 0 }}>
        {depth > 0 && (
          <div className="border-l-2 border-gray-100 ml-3 pl-3">
            <ClaimCard
              claim={claim}
              samePairs={samePairs}
              jobId={jobId}
              onClaimUpdate={onClaimUpdate}
              groupColor={colorMap[num]}
            />
            {children.map(child => renderClaim(child, depth + 1))}
          </div>
        )}
        {depth === 0 && (
          <>
            <ClaimCard
              claim={claim}
              samePairs={samePairs}
              jobId={jobId}
              onClaimUpdate={onClaimUpdate}
              groupColor={colorMap[num]}
            />
            {children.length > 0 && (
              <div className="ml-4">
                {children.map(child => renderClaim(child, depth + 1))}
              </div>
            )}
          </>
        )}
      </div>
    )
  }

  return (
    <div>
      {independents.map(c => renderClaim(c, 0))}
    </div>
  )
}

// ─── 메인 윈도우 ─────────────────────────────────────────────────────────────
export default function ClaimAnalysisWindow({ jobId, onClose }) {
  const [treeData, setTreeData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')

  useEffect(() => {
    if (!jobId) return
    load()
  }, [jobId])

  async function load() {
    setLoading(true)
    setLoadError('')
    try {
      const data = await getClaimTree(jobId)
      setTreeData(data)
    } catch (e) {
      setLoadError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handlePurposeUpdate(result) {
    setTreeData(prev => ({ ...prev, purpose_effects: result }))
  }

  function handleClaimUpdate(updated) {
    setTreeData(prev => ({
      ...prev,
      claims: prev.claims.map(c =>
        c.claim_number === updated.claim_number ? updated : c
      ),
    }))
  }

  const totalClaims = treeData?.claims?.length ?? 0
  const independentCount = treeData?.claims?.filter(c => c.claim_type === 'independent').length ?? 0

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-gray-50" style={{ fontFamily: 'inherit' }}>

      {/* 헤더 */}
      <header className="flex items-center justify-between px-6 py-3 bg-white border-b shadow-sm shrink-0">
        <div className="flex items-center gap-4">
          <button
            onClick={onClose}
            className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-800 border rounded-lg px-3 py-1.5 hover:bg-gray-50 transition"
          >
            ← 닫기
          </button>
          <h1 className="text-base font-bold text-gray-800">청구항 분석</h1>
          {totalClaims > 0 && (
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <span className="bg-blue-100 text-blue-700 rounded-full px-2 py-0.5 font-medium">
                독립항 {independentCount}개
              </span>
              <span className="bg-gray-100 text-gray-600 rounded-full px-2 py-0.5 font-medium">
                총 {totalClaims}개
              </span>
            </div>
          )}
        </div>
        <button
          onClick={load}
          className="text-xs text-gray-400 hover:text-gray-600 transition"
          title="새로고침"
        >
          ↺ 새로고침
        </button>
      </header>

      {/* 본문 */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-6">

          {loading && (
            <div className="flex items-center justify-center py-20 text-blue-500 gap-3">
              <div className="w-5 h-5 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
              <span className="text-sm">청구항 데이터 불러오는 중…</span>
            </div>
          )}

          {loadError && (
            <div className="text-sm text-red-600 bg-red-50 border border-red-200 rounded-xl p-4 mb-4">
              {loadError}
            </div>
          )}

          {!loading && treeData && (
            <>
              {/* 발명의 목적/효과 */}
              <PurposeSection
                jobId={jobId}
                purposeEffects={treeData.purpose_effects}
                onUpdate={handlePurposeUpdate}
              />

              {/* same_pairs 안내 */}
              {Object.keys(treeData.same_pairs || {}).length > 0 && (
                <div className="bg-purple-50 border border-purple-200 rounded-xl p-4 mb-5 text-sm text-purple-700">
                  <strong>카테고리 동일 청구항:</strong>{' '}
                  {Object.entries(treeData.same_pairs).map(([k, v]) => (
                    <span key={k} className="mr-3">제{k}항 ↔ 제{v}항</span>
                  ))}
                </div>
              )}

              {/* 청구항 트리 */}
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-sm font-bold text-gray-800">청구항 트리</h2>
                <span className="text-xs text-gray-400">{totalClaims}개 청구항</span>
              </div>

              {totalClaims === 0 ? (
                <div className="text-center text-gray-400 py-16 text-sm">
                  <div className="text-3xl mb-3">📋</div>
                  <p>청구항을 먼저 입력하세요.</p>
                </div>
              ) : (
                <ClaimTree
                  claims={treeData.claims}
                  samePairs={treeData.same_pairs}
                  jobId={jobId}
                  onClaimUpdate={handleClaimUpdate}
                />
              )}
            </>
          )}
        </div>
      </div>
    </div>
  )
}
