import { useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import { addManualClaim, streamPrepare, streamReport, reportBatchDependent, getDependentBatchStatus, uploadFiles, getContextInfo, clearContext, checkJobStatus, detectCategory, deleteJob, deleteAllJobs, cancelGeneration, getMissingPriorArt } from './api/client'
import ClaimAnalysisWindow from './components/ClaimAnalysisWindow'
import MissingPriorArtSearch from './components/MissingPriorArtSearch'
import { splitMissingPriorArtMarkdown } from './utils/missingPriorArt'

import FilePanel from './components/FilePanel'
import ProgressPanel from './components/ProgressPanel'
import SettingsModal from './components/SettingsModal'
import ChatPanel from './components/ChatPanel'

function AriaEmblem() {
  return (
    <svg className="aria-emblem" width="42" height="42" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
      <defs>
        <linearGradient id="aria-bg" x1="5" y1="3" x2="43" y2="46" gradientUnits="userSpaceOnUse">
          <stop stopColor="#27235f" />
          <stop offset="0.55" stopColor="#6557e8" />
          <stop offset="1" stopColor="#42b8e8" />
        </linearGradient>
        <linearGradient id="aria-letter" x1="14" y1="38" x2="34" y2="10" gradientUnits="userSpaceOnUse">
          <stop stopColor="#d9f6ff" />
          <stop offset="1" stopColor="#ffffff" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="46" height="46" rx="15" fill="url(#aria-bg)" />
      <rect x="1.5" y="1.5" width="45" height="45" rx="14.5" stroke="white" strokeOpacity="0.18" />
      <path d="M13.5 36 21.5 14.8c.9-2.5 4.5-2.5 5.4 0L35 36" stroke="url(#aria-letter)" strokeWidth="5.5" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M17.5 29h13.2" stroke="#dfff93" strokeWidth="4" strokeLinecap="round" />
      <path d="M37 8.5v7M33.5 12h7" stroke="#ffd1ec" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="40.5" cy="7.5" r="1.4" fill="#dfff93" />
    </svg>
  )
}


// ── Phase 1 유사도 배지 스타일 ───────────────────────────────────────────────
const SIMILARITY_STYLES = {
  '동일':           { badge: 'bg-blue-100 text-blue-800 border border-blue-300' },
  '실질적동일':     { badge: 'bg-green-100 text-green-800 border border-green-300' },
  '일부차이':       { badge: 'bg-orange-100 text-orange-800 border border-orange-300' },
  '일부유사':       { badge: 'bg-amber-100 text-amber-800 border border-amber-300' },
  '차이':           { badge: 'bg-gray-100 text-gray-700 border border-gray-300' },
}

const SIMILARITY_PRESENTATION_BY_JUDGMENT = {
  '동일':       { symbol: '●', row: 'phase-status phase-status-match' },
  '실질적동일': { symbol: '●', row: 'phase-status phase-status-close' },
  '일부차이':   { symbol: '●', row: 'phase-status phase-status-partial' },
  '일부유사':   { symbol: '●', row: 'phase-status phase-status-related' },
  '차이':       { symbol: '●', row: 'phase-status phase-status-diff' },
  '대응없음':   { symbol: '●', row: 'phase-status phase-status-diff' },
  '대응안됨':   { symbol: '●', row: 'phase-status phase-status-diff' },
}

const SIMILARITY_RANGE_BY_JUDGMENT = {
  '동일':       '95~100%',
  '실질적동일': '90~94%',
  '일부차이':   '85~89%',
  '일부유사':   '80~84%',
  '차이':       '1~79%',
  '대응없음':   '0%',
  '대응안됨':   '0%',
}

function similarityRange(pct, judgment = '') {
  if (pct) return String(pct).replace(/\s+/g, '')
  const normalizedJudgment = String(judgment || '').replace(/\s+/g, '')
  return SIMILARITY_RANGE_BY_JUDGMENT[normalizedJudgment] || ''
}

function similarityPresentation(pct, judgment = '') {
  const normalizedJudgment = String(judgment || '').replace(/\s+/g, '')
  if (SIMILARITY_PRESENTATION_BY_JUDGMENT[normalizedJudgment]) {
    return SIMILARITY_PRESENTATION_BY_JUDGMENT[normalizedJudgment]
  }
  const value = Number.parseInt(String(pct || '').replace('%', ''), 10)
  if (value >= 95) return { symbol: '●', row: 'phase-status phase-status-match' }
  if (value >= 90) return { symbol: '●', row: 'phase-status phase-status-close' }
  if (value >= 85) return { symbol: '●', row: 'phase-status phase-status-partial' }
  if (value >= 80) return { symbol: '●', row: 'phase-status phase-status-related' }
  return { symbol: '●', row: 'phase-status phase-status-diff' }
}

function sanitizeReportText(text) {
  return String(text || '')
    // 과거 생성 프롬프트의 단계 안내 문구는 현재 보고서에 표시하지 않는다.
    .replace(/^\s*청구항\s*\d+에\s*대한\s*특허\s*분석\s*Phase\s*1\s*보고서입니다\.?\s*\r?\n?/gim, '')
    // 기존 캐시 보고서도 번역·발췌 라벨 없이 자연스러운 한 줄로 표시한다.
    .replace(/^(\s*인용발명\s*\d+\s*-\s*)번역(?:\s+\d+)?:\s*/gm, '$1')
    .replace(/^(\s*인용발명\s*\d+\s*-\s*)발췌(?:\s+\d+)?:\s*/gm, '$1')
    .replace(/^(\s*인용발명\s*\d+\s*-\s*.*?)\s+발췌(?:\s+\d+)?:\s*/gm, '$1 ')
}

function normalizeCitationQuoteBlocks(text) {
  const fieldBoundary = String.raw`(?=\n-\s*(?:\*\*)?(?:개시 상태|청구항 구성|청구항 추가 구성|기술적 의미|유사도 평가|인용발명 대응 및 판단|판단 이유|판단 근거|차이점|보완 검토|미대응 구성|미대응 구성 및 검색어)(?:\*\*)?\s*:|\n#{1,6}\s|\n---|(?![\s\S]))`
  const blockRe = new RegExp(
    String.raw`(^-\s*(?:\*\*)?\s*인용발명\s*대응\s*원문\s*:\s*(?:\*\*)?\s*\n)([\s\S]*?)${fieldBoundary}`,
    'gm'
  )

  return String(text || '').replace(blockRe, (_, header, rawBody) => {
    const body = rawBody.replace(/^\s*\*\*\s*$/gm, '')
    const sentences = []
    const translationRe = /^\s*-\s*번역\((인용발명\s*\d+)\)\s*:\s*(.+?)\s*$/gm
    let match
    while ((match = translationRe.exec(body)) !== null) {
      const sentence = match[2].trim().replace(/\*+$/g, '').trim()
      const docName = match[1]
      const escapedDocName = docName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
      const excerptRe = new RegExp(
        String.raw`^\s*-\s*발췌\(${escapedDocName}\)\s*:\s*"(.+?)"\s*\((단락|본문)\s+(.+?)\)\s*$`,
        'm'
      )
      const excerpt = body.match(excerptRe)
      const translated = /[.!?]$/.test(sentence) ? sentence : `${sentence}.`
      sentences.push(
        excerpt
          ? `${docName}에는 ${translated} ${excerpt[2]} ${excerpt[3]} "${excerpt[1]}"`
          : `${docName}에는 ${translated}`
      )
    }
    if (sentences.length === 0) {
      const directRe = /^\s*-\s*(인용발명\s*\d+)\s*:\s*"?(.+?)"?(?:\s+\((?:단락|본문).*)?\s*$/gm
      while ((match = directRe.exec(body)) !== null) {
        const sentence = match[2].trim().replace(/\*+$/g, '').trim()
        sentences.push(`${match[1]}에는 ${/[.!?]$/.test(sentence) ? sentence : `${sentence}.`}`)
      }
    }
    if (sentences.length === 0) {
      return header.replace(/\*\*/g, '') + body.trim()
    }
    return `${header.replace(/\*\*/g, '')}  ${sentences.join(' ')}`
  })
}

function normalizeSupplementReviewBlocks(text) {
  const supplementRe = /(^-\s*(?:\*\*)?보완 검토(?:\*\*)?\s*:\s*\n)([\s\S]*?)(?=\n-\s*(?:\*\*)?(?:개시 상태|청구항 구성|청구항 추가 구성|기술적 의미|유사도 평가|인용발명 대응 원문|인용발명 대응 및 판단|판단 이유|판단 근거|차이점|미대응 구성|미대응 구성 및 검색어)(?:\*\*)?\s*:|\n#{1,6}\s|\n---|(?![\s\S]))/gm
  return String(text || '').replace(supplementRe, (_, header, rawBody) => {
    const normalizedLines = rawBody.split('\n').map(line => {
      const legacy = line.match(
        /^\s*-\s*(인용발명\s*\d+)\s*:\s*"(.+?)"\s*\((단락|본문)\s+(.+?)\)\s*;\s*판정\s*[^;]+;\s*잔여 제한\s*:\s*.+$/
      )
      if (!legacy) return line.replace(/^\s*\*\*\s*$/, '')
      return `  ${legacy[1]}에는 ${legacy[3]} ${legacy[4]} "${legacy[2]}"`
    })
    return header.replace(/\*\*/g, '') + normalizedLines.join('\n').trimEnd()
  })
}

const RELATED_A_TAB_KEY = '__relatedA'
const RELATED_A_TAB_LABEL = '관련도 A 인용발명'

function splitRelatedAReport(text) {
  const md = sanitizeReportText(text)
  const match = md.match(/^##\s*관련도\s*A\s*인용발명\s*$/m)
  if (!match) return { reportMd: md, relatedMd: '' }
  const beforeHeading = md.slice(0, match.index)
  const sepIndex = beforeHeading.lastIndexOf('\n---\n')
  const reportMd = (sepIndex >= 0 ? md.slice(0, sepIndex) : beforeHeading).trim()
  const relatedStart = sepIndex >= 0 ? sepIndex + '\n---\n'.length : match.index
  const relatedMd = md.slice(relatedStart).trim()
  return { reportMd, relatedMd }
}

function relatedATabMarkdown(prevMd, claimNumber, relatedMd) {
  const clean = sanitizeReportText(relatedMd).trim()
  if (!clean) return prevMd || ''
  const section = `### 청구항 ${claimNumber}\n\n${clean}`
  const prev = sanitizeReportText(prevMd).trim()
  if (!prev) return section
  if (prev.includes(section)) return prev
  return `${prev}\n\n---\n\n${section}`
}

function addReportEntryWithRelated(reports, claimNumber, reportMd, usedInventions = [], relatedMd = '') {
  const { reportMd: cleanReport, relatedMd: extractedRelated } = splitRelatedAReport(reportMd)
  const next = {
    ...reports,
    [claimNumber]: {
      report_md: cleanReport,
      usedInventions,
    },
  }
  const mergedRelated = relatedATabMarkdown(
    next[RELATED_A_TAB_KEY]?.report_md,
    claimNumber,
    relatedMd || extractedRelated,
  )
  if (mergedRelated) {
    next[RELATED_A_TAB_KEY] = {
      report_md: mergedRelated,
      usedInventions: [],
      isRelatedATab: true,
    }
  }
  return next
}

function reportTabKeys(allReports) {
  const keys = Object.keys(allReports || {})
  const claimKeys = keys
    .filter(key => key !== RELATED_A_TAB_KEY)
    .sort((a, b) => Number(a) - Number(b))
  return allReports?.[RELATED_A_TAB_KEY] ? [...claimKeys, RELATED_A_TAB_KEY] : claimKeys
}

// ── Phase 1 필드 라벨 스타일 ─────────────────────────────────────────────────
const FIELD_LABEL_STYLES = [
  { re: /^(청구항\s*(?:추가\s*)?구성)\s*:/,      cls: 'border-indigo-300 bg-indigo-50/50 text-indigo-950', labelCls: 'text-indigo-700' },
  { re: /^(인용발명\s*대응(?:\s*원문|\s*및\s*판단))\s*:/, cls: 'border-teal-400 bg-teal-50/50 text-teal-950', labelCls: 'text-teal-700' },
  { re: /^(유사도 평가)\s*:/,              cls: 'border-amber-300 bg-amber-50/70 text-amber-950', labelCls: 'text-amber-700' },
  { re: /^(판단 이유)\s*:/,               cls: 'border-violet-300 bg-violet-50/50 text-violet-950', labelCls: 'text-violet-700' },
  { re: /^(판단 근거)\s*:/,               cls: 'border-violet-400 bg-violet-50/50 text-violet-950', labelCls: 'text-violet-700' },
  { re: /^(차이점)\s*:/,                  cls: 'border-rose-300 bg-rose-50/50 text-rose-950', labelCls: 'text-rose-700' },
  { re: /^(보완 검토)\s*:/,               cls: 'border-cyan-300 bg-cyan-50/50 text-cyan-950', labelCls: 'text-cyan-700' },
  { re: /^(유사점 요약)\s*:/,             cls: 'border-green-300 bg-green-50/50 text-green-950', labelCls: 'text-green-700' },
]

function extractText(children) {
  if (typeof children === 'string') return children
  if (Array.isArray(children)) return children.map(c => extractText(c)).join('')
  if (children?.props?.children) return extractText(children.props.children)
  return ''
}

// ── Phase 1 커스텀 렌더러 ────────────────────────────────────────────────────
function Phase1H3({ children }) {
  const text = extractText(children)

  // 청구항 제목 — [추가 구성]보다 약간 크고 굵게
  if (/^\s*(?:청구항|종속항)\s*제?\s*\d+\s*항?\s*$/.test(text)) {
    return (
      <h3 className="mt-1 mb-4 text-base font-bold text-gray-900">
        {children}
      </h3>
    )
  }

  // 종합 분석 요약 섹션
  if (/종합\s*분석\s*요약|종합분석요약/.test(text)) {
    return (
      <h3 className="report-major-heading">
        {children}
      </h3>
    )
  }

  const m = text.match(/^\[(구성요소|추가\s*구성|전제부)(?:\s*\(\s*([A-Z](?:-\d+)?)\s*\))?\]$/)
  if (m) {
    return (
      <h3 className="phase1-component-heading mt-7 mb-3 text-xl font-extrabold text-slate-900">
        {children}
      </h3>
    )
  }

  return <h3 className="text-sm font-bold mt-4 mb-2 text-gray-800">{children}</h3>
}

function Phase1ListItem({ children }) {
  const text = extractText(children)

  // 유사도 라인 감지 — 이모지·기존 바탕색 지시 잔재도 흡수
  const newSimMatch = text.match(
    /^(?:유사도\s*:\s*)?\(([A-Z](?:-\d+)?)\)\s*(동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음|대응안됨|대응 안됨)?(?:\s*(?:\(\s*)?(\d{1,3}(?:\s*~\s*\d{1,3})?%)(?:\s*\))?)?/
  )
  const oldSimMatch = text.match(
    /^유사도\s*:\s*(동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음|대응안됨|대응 안됨)?(?:\s*(?:\(\s*)?(\d{1,3}(?:\s*~\s*\d{1,3})?%)(?:\s*\))?)?/
  )
  if (newSimMatch || oldSimMatch) {
    const elementLabel = newSimMatch ? newSimMatch[1] : ''
    const labelText = newSimMatch ? (newSimMatch[2] || '') : (oldSimMatch?.[1] || '')
    const normalizedLabel = labelText.replace(/\s+/g, '') === '대응없음' ? '차이' : labelText.replace(/\s+/g, '')
    const pct = similarityRange(
      newSimMatch ? (newSimMatch[3] || '') : (oldSimMatch?.[2] || ''),
      normalizedLabel,
    )
    const style = SIMILARITY_STYLES[normalizedLabel] || SIMILARITY_STYLES['차이']
    const presentation = similarityPresentation(pct, normalizedLabel)
    return (
      <li className={`flex items-center gap-2 py-2 px-3 rounded-r my-1.5 list-none -ml-5 ${presentation.row}`}>
        {elementLabel && <span className="text-xs font-semibold text-gray-700 shrink-0">({elementLabel})</span>}
        {normalizedLabel
          ? (
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-semibold ${style.badge}`}>
              {normalizedLabel}
            </span>
          )
          : <span className="text-xs text-gray-400 italic">미입력</span>
        }
        {pct && (
          <span className="inline-flex items-center gap-1 text-xs text-gray-500 font-mono whitespace-nowrap shrink-0">
            <span aria-hidden="true">{presentation.symbol}</span>
            <span className="shrink-0">{pct}</span>
          </span>
        )}
      </li>
    )
  }

  // 필드 라벨 스타일
  for (const { re, cls, labelCls } of FIELD_LABEL_STYLES) {
    const fieldMatch = text.match(re)
    if (fieldMatch) {
      const label = fieldMatch[1].replace(/\s+/g, ' ')
      const body = text.slice(fieldMatch[0].length).trim()
      return (
        <li className={`list-none -ml-5 my-2 rounded-md border-l-4 px-3 py-2 text-sm leading-relaxed ${cls}`}>
          <span className={`block text-xs font-bold ${labelCls}`}>{label}</span>
          {body && <span className="mt-1 block whitespace-pre-line text-slate-800">{body}</span>}
        </li>
      )
    }
  }

  return <li className="text-sm leading-relaxed my-1">{children}</li>
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
}

function renderJudgmentInline(text) {
  const normalizedText = text.replace(/^\(전제부\s+P\)/, '(P)')
  const match = normalizedText.match(
    /^(\([A-Z](?:-\d+)?\)(?:\s*(?:및|,)\s*\([A-Z](?:-\d+)?\))*)\s+(동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음|대응안됨|대응 안됨)(?:\s+(?:\(\s*)?(\d{1,3}(?:\s*~\s*\d{1,3})?%)(?:\s*\))?)?\s*$/
  )
  const fallbackMatch = !match && normalizedText.match(
    /^(?:\(\s*)?(동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음|대응안됨|대응 안됨)(?:\s+(\d{1,3}(?:\s*~\s*\d{1,3})?%))?(?:\s*\))?\s*$/
  )
  if (!match && !fallbackMatch) return null
  const label = match ? match[1] : ''
  const rawJudgment = match ? match[2] : fallbackMatch[1]
  const judgment = rawJudgment.replace(/\s+/g, '')
  const pct = similarityRange(match ? (match[3] || '') : (fallbackMatch[2] || ''), judgment)
  return { label, judgment, pct }
}

function ReportParagraph({ children }) {
  const text = extractText(children)
  const trimmed = text.trim()
  if (/^\[(구성요소|추가\s*구성|전제부)(?:\s*\(\s*[A-Z](?:-\d+)?\s*\))?\]$/.test(trimmed)) {
    return <p className="phase1-component-heading">{children}</p>
  }
  if (/^\[(인용발명\s*\d+\s*단독\(신규성\)|인용발명\s*\d+\s*\+\s*주지관용\(진보성\)|인용발명\s*\d+\s*과\s*인용발명\s*\d+\s*의\s*결합(?:\s*및\s*주지관용)?\(진보성\))\]$/.test(trimmed)) {
    return <p className="mt-2 mb-5 text-xl font-bold tracking-tight text-slate-950">{children}</p>
  }
  if (/^\[(구성대비|종합분석요약|구성요소|종합 판단|유사점|차이점|신규성 검토|결합 검토)\]$/.test(trimmed)) {
    const isMajor = /^\[(구성대비|종합분석요약|종합 판단)\]$/.test(trimmed)
    const isDiff = trimmed === '[차이점]'
    const isSimilar = trimmed === '[유사점]'
    const isLegalReview = /^\[(신규성 검토|결합 검토)\]$/.test(trimmed)
    return (
      <p className={
        isMajor
          ? 'report-major-heading'
          : isDiff
            ? 'report-minor-heading report-minor-diff'
            : isSimilar
              ? 'report-minor-heading report-minor-similar'
              : isLegalReview
                  ? 'report-minor-heading report-minor-legal'
                : 'mt-4 mb-1 text-sm font-semibold text-slate-700'
      }>
        {children}
      </p>
    )
  }
  if (/^\[(차이점)\s*\d+\]$/.test(trimmed)) {
    return <p className="mt-4 mb-1 text-sm font-semibold text-slate-700">{children}</p>
  }
  const judgmentInline = renderJudgmentInline(trimmed)
  if (judgmentInline) {
    const { label, judgment, pct } = judgmentInline
    const presentation = similarityPresentation(pct, judgment)
    return (
      <p className={`flex items-center gap-2 font-semibold text-sm pl-3 py-0.5 rounded-r mt-4 mb-1 overflow-x-auto ${presentation.row}`}>
        {label && <span className="shrink-0">{label}</span>}
        <span className="shrink-0">{judgment}</span>
        {pct && <span className="shrink-0 font-mono">{presentation.symbol} {pct}</span>}
      </p>
    )
  }
  return <p className="my-1 text-sm leading-relaxed">{children}</p>
}

function preprocessReport(md) {
  md = sanitizeReportText(md)
  md = normalizeCitationQuoteBlocks(md)
  md = normalizeSupplementReviewBlocks(md)

  function fieldClass(label) {
    if (label === '청구항 구성' || label === '청구항 추가 구성' || label === '추가 구성' || label.startsWith('청구항')) return 'phase1-field phase1-field-claim'
    if (label === '인용발명 대응 원문' || label === '인용발명 대응 및 판단') return 'phase1-field phase1-field-quote'
    if (label === '유사도 평가') return 'phase1-field phase1-field-similarity'
    if (label === '판단 이유' || label === '판단 근거') return 'phase1-field phase1-field-reason'
    if (label === '차이점') return 'phase1-field phase1-field-diff'
    if (label === '보완 검토') return 'phase1-field phase1-field-supplement'
    if (label === '유사점 요약') return 'phase1-field phase1-field-similar'
    return 'phase1-field'
  }

  function normalizeFieldLabel(rawLabel) {
    return rawLabel.replace(/\*\*/g, '').replace(/\s+/g, ' ').trim()
  }

  function normalizePhase1Fields(text) {
    const fieldRe = /^\s*(?:-\s*)?(?:\*\*)?(청구항\s*(?:추가\s*)?구성|추가\s*구성|기술적\s*의미|인용발명\s*대응(?:\s*원문|\s*및\s*판단)|인용발명 대응 부분 요약|유사도 평가|판단 이유|판단 근거|차이점|보완 검토|미대응\s*구성(?:\s*및\s*검색어)?|유사점 요약)(?:\*\*)?\s*:\s*(.*)$/
    const sectionHeaderRe = /^#{1,6}\s*\[\s*(?:구성요소|추가\s*구성|추가구성|전제부|종속항|청구항\s*\d*(?:\s*추가\s*구성)?)(?:\s*[:\-]?\s*(?:\([^)]*\)|[A-Za-z0-9-]+))?\s*\]\s*$/i
    const componentHeaderRe = /^\s*(?:#{1,6}\s*)?\[\s*(?:구성요소|추가\s*구성|추가구성|전제부|종속항|청구항\s*\d*(?:\s*추가\s*구성)?)(?:\s*[:\-]?\s*(?:\([^)]*\)|[A-Za-z0-9-]+))?\s*\]\s*$/i
    const summaryHeaderRe = /^\s*(?:#{1,6}\s*)?\[\s*(?:종합\s*분석\s*요약|종합\s*판단|유사점|차이점|신규성\s*검토|결합\s*검토|진보성\s*검토)\s*\]\s*$/i
    const lines = text.split('\n')
    const result = []
    let i = 0

    while (i < lines.length) {
      const line = lines[i]
      const trimmedLine = line.trim()
      const isCompHeader = componentHeaderRe.test(trimmedLine)

      if (isCompHeader) {
        result.push(line)
        i += 1
        let fields = {}
        const otherLines = []

        const flushFields = (fieldsDict) => {
          if (!fieldsDict || Object.keys(fieldsDict).length === 0) return
          const blockCards = []
          const claimCard = fieldsDict['청구항 구성'] || fieldsDict['청구항 추가 구성'] || fieldsDict['추가 구성']
          const simCard = fieldsDict['유사도 평가']
          if (claimCard || simCard) {
            const rowCards = []
            if (claimCard) rowCards.push(claimCard)
            if (simCard) rowCards.push(simCard)
            if (rowCards.length === 1) {
              blockCards.push(rowCards[0])
            } else {
              blockCards.push(`<div class="phase1-card-row">${rowCards.join('\n')}</div>`)
            }
          }

          if (fieldsDict['기술적 의미']) {
            blockCards.push(fieldsDict['기술적 의미'])
          }

          const evidenceCard = fieldsDict['인용발명 대응 및 판단'] || fieldsDict['인용발명 대응 원문']
          if (evidenceCard) {
            blockCards.push(evidenceCard)
          }

          const reasonCard = fieldsDict['판단 이유'] || fieldsDict['판단 근거']
          const diffCard = fieldsDict['차이점']
          if (reasonCard || diffCard) {
            const rowCards = []
            if (reasonCard) rowCards.push(reasonCard)
            if (diffCard) rowCards.push(diffCard)
            if (rowCards.length === 1) {
              blockCards.push(rowCards[0])
            } else {
              blockCards.push(`<div class="phase1-card-row">${rowCards.join('\n')}</div>`)
            }
          }

          if (fieldsDict['보완 검토']) {
            blockCards.push(fieldsDict['보완 검토'])
          }

          const missingCard = fieldsDict['미대응 구성 및 검색어'] || fieldsDict['미대응 구성']
          if (missingCard) {
            blockCards.push(missingCard)
          }

          for (const [k, v] of Object.entries(fieldsDict)) {
            if (!['청구항 구성', '청구항 추가 구성', '추가 구성', '기술적 의미', '유사도 평가', '인용발명 대응 및 판단', '인용발명 대응 원문', '판단 이유', '판단 근거', '차이점', '보완 검토', '미대응 구성 및 검색어', '미대응 구성', '인용발명 대응 부분 요약'].includes(k)) {
              blockCards.push(v)
            }
          }


          if (blockCards.length > 0) {
            result.push(`<div class="phase1-component-block">\n${blockCards.join('\n')}\n</div>`)
          }
        }

        while (i < lines.length) {
          const current = lines[i]
          const trimmed = current.trim()
          if (componentHeaderRe.test(trimmed) || summaryHeaderRe.test(trimmed) || /^#{1,6}\s/.test(trimmed)) {
            break
          }

          const match = current.match(fieldRe)
          if (match) {
            const rawLabel = match[1]
            const label = normalizeFieldLabel(rawLabel)

            if ((label === '청구항 구성' || label === '청구항 추가 구성' || label === '추가 구성') && (fields['청구항 구성'] || fields['청구항 추가 구성'] || fields['추가 구성'])) {
              flushFields(fields)
              fields = {}
            }

            if (label === '인용발명 대응 부분 요약') {
              i += 1
              while (i < lines.length) {
                const sub = lines[i].trim()
                if (!sub || fieldRe.test(lines[i]) || componentHeaderRe.test(sub) || /^#{1,6}\s/.test(sub)) break
                i += 1
              }
              continue
            }

            const bodyLines = []
            if (match[2].trim()) bodyLines.push(match[2].trim())
            i += 1

            while (i < lines.length) {
              const cur = lines[i]
              const curTrimmed = cur.trim()
              if (!curTrimmed) {
                if (label === '차이점') {
                  bodyLines.push('')
                  i += 1
                  continue
                }
                i += 1
                break
              }
              if (
                curTrimmed === '---' ||
                fieldRe.test(cur) ||
                /^#{1,6}\s/.test(curTrimmed) ||
                componentHeaderRe.test(curTrimmed) ||
                summaryHeaderRe.test(curTrimmed) ||
                (label !== '차이점' && /^\([A-Z](?:-\d+)?\)\s/.test(curTrimmed)) ||
                /^\s*-?\s*(유사점 요약|차이점)\s*:/.test(curTrimmed)
              ) {
                break
              }
              bodyLines.push(curTrimmed)
              i += 1
            }

            const body = bodyLines.join('\n')
            const fieldHtml = `<div class="${fieldClass(label)}"><div class="phase1-field-label">${escapeHtml(label)}</div>${body ? `<div class="phase1-field-body">${escapeHtml(body).replace(/\n/g, '<br />')}</div>` : ''}</div>`
            fields[label] = fieldHtml
          } else {
            otherLines.push(current)
            i += 1
          }
        }

        flushFields(fields)

        if (otherLines.length > 0) {
          result.push(otherLines.join('\n'))
        }

        continue
      }

      const match = line.match(fieldRe)
      if (!match) {
        result.push(line)
        i += 1
        continue
      }

      const label = normalizeFieldLabel(match[1])
      if (label === '인용발명 대응 부분 요약') {
        i += 1
        while (i < lines.length) {
          const sub = lines[i].trim()
          if (!sub || fieldRe.test(lines[i]) || componentHeaderRe.test(sub) || /^#{1,6}\s/.test(sub)) break
          i += 1
        }
        continue
      }

      const bodyLines = []
      if (match[2].trim()) bodyLines.push(match[2].trim())
      i += 1

      while (i < lines.length) {
        const current = lines[i]
        const trimmed = current.trim()
        if (!trimmed) {
          if (label === '차이점') {
            bodyLines.push('')
            i += 1
            continue
          }
          i += 1
          break
        }
        if (
          trimmed === '---' ||
          fieldRe.test(current) ||
          /^#{1,6}\s/.test(trimmed) ||
          sectionHeaderRe.test(trimmed) ||
          summaryHeaderRe.test(trimmed) ||
          (label !== '차이점' && /^\([A-Z](?:-\d+)?\)\s/.test(trimmed)) ||
          /^\s*-?\s*(유사점 요약|차이점)\s*:/.test(trimmed)
        ) {
          break
        }
        bodyLines.push(trimmed)
        i += 1
      }

      const body = bodyLines.join('\n')
      result.push(
        `<div class="${fieldClass(label)}"><div class="phase1-field-label">${escapeHtml(label)}</div>${body ? `<div class="phase1-field-body">${escapeHtml(body).replace(/\n/g, '<br />')}</div>` : ''}</div>`
      )
    }

    return result.join('\n')
  }

  function normalizeDifferenceEntries(text) {
    const lines = text.split('\n')
    const result = []
    let i = 0

    while (i < lines.length) {
      const line = lines[i]
      if (!/^\([A-Z](?:-\d+)?\)\s/.test(line.trim())) {
        result.push(line)
        i += 1
        continue
      }

      const chunk = [line.trim()]
      i += 1
      while (i < lines.length) {
        const current = lines[i]
        const trimmed = current.trim()
        if (!trimmed) {
          i += 1
          break
        }
        if (
          /^\([A-Z](?:-\d+)?\)\s/.test(trimmed) ||
          /^-?\s*(유사점 요약|차이점)\s*:/.test(trimmed)
        ) {
          break
        }
        chunk.push(trimmed)
        i += 1
      }

      const merged = chunk.join(' ').replace(/\s+/g, ' ').trim()
      const withConclusionBreak = merged.replace(
        /\s+((?:다만|따라서)\s+)/g,
        '\n\n$1'
      )
      result.push(withConclusionBreak)
      if (i < lines.length && lines[i].trim() === '') result.push('')
    }

    return result.join('\n')
  }

  function keepFirstComponentHeader(text) {
    let seenComponentHeader = false
    return text.split('\n').filter(line => {
      if (!/^\s*\[(구성요소)\]\s*$/.test(line)) return true
      if (!seenComponentHeader) {
        seenComponentHeader = true
        return true
      }
      return false
    }).join('\n')
  }

  function mergeClaimIntoJudgmentCards(text) {
    const judgmentRe = String.raw`((?:\([A-Z](?:-\d+)?\)\s*)?(?:동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음|대응안됨|대응 안됨)(?:\s+\d+%)?)`
    const claimRe = String.raw`<div class="phase1-field phase1-field-claim"><div class="phase1-field-label">청구항 구성<\/div>(?:<div class="phase1-field-body">([\s\S]*?)<\/div>)?<\/div>`
    return text.replace(
      new RegExp(`^${judgmentRe}\\s*\\n+${claimRe}`, 'gm'),
      (_, judgment, claimBody = '') => {
        const pct = (judgment.match(/(\d+%)\s*$/) || [])[1] || ''
        const parsedJudgment = renderJudgmentInline(judgment)
        const presentation = similarityPresentation(pct, parsedJudgment?.judgment)
        return (
          `<div class="phase1-judgment-card ${presentation.row}">` +
          `<div class="phase1-judgment-line">${escapeHtml(judgment)}${pct ? ` <span aria-hidden="true">${presentation.symbol}</span>` : ''}</div>` +
          (claimBody ? `<div class="phase1-judgment-claim">${claimBody}</div>` : '') +
          `</div>`
        )
      }
    )
  }

  function normalizeSummaryItems(text) {
    const summaryRegex = /^-\s*(유사점\s*요약|차이점|신규성\s*검토|결합\s*검토|진보성\s*검토)\s*:\s*(.*)$/
    const lines = text.split('\n')
    const result = []
    let i = 0

    while (i < lines.length) {
      const line = lines[i]
      const match = line.match(summaryRegex)
      if (!match) {
        result.push(line)
        i += 1
        continue
      }

      const rawLabel = match[1].replace(/\[|\]/g, '').replace(/\s+/g, ' ')
      const bodyLines = []
      if (match[2].trim()) bodyLines.push(match[2].trim())
      i += 1

      while (i < lines.length) {
        const current = lines[i]
        const trimmed = current.trim()
        if (!trimmed) {
          if (bodyLines.length > 0) {
            i += 1
            break
          }
          i += 1
          continue
        }
        if (
          summaryRegex.test(trimmed) ||
          /^-\s*(?:[^\n:]+)\s*:/.test(trimmed) ||
          /^\[(유사점|차이점|종합분석요약|신규성 검토|결합 검토)\]$/.test(trimmed) ||
          /^#{1,6}\s/.test(trimmed)
        ) {
          break
        }
        bodyLines.push(trimmed)
        i += 1
      }

      const heading =
        rawLabel === '유사점 요약' || rawLabel === '유사점'
          ? '[유사점]'
          : rawLabel === '차이점'
            ? '[차이점]'
            : `[${rawLabel}]`
      result.push(`${heading}\n\n${bodyLines.join('\n').trim()}`)
    }

    return result.join('\n')
  }


  const judgmentPrefix = String.raw`\([A-Z](?:-\d+)?\)(?:\s*(?:및|,)\s*\([A-Z](?:-\d+)?\))*\s+(?:동일|실질적동일|실질적\s+동일|일부차이|일부\s+차이|일부유사|일부\s+유사|차이|대응\s+없음|대응안됨|대응\s+안됨)(?:\s+\d+%)?`
  // CLI 에이전트가 새어 보낸 도구 호출 줄(update_topic(...) 등) 제거 — 캐시·히스토리 구보고서까지 정리
  md = md.replace(/^[ \t]*[a-z][a-z0-9_]*\([a-z_]+\s*=\s*['"].*\)[ \t]*-*[ \t]*$/gm, '')
  md = md.replace(
    /([^\n])\s*(#{3,6}\s*\[(?:구성요소|추가\s*구성|전제부)(?:\s*\([A-Z](?:-\d+)?\))?\])/g,
    '$1\n\n$2'
  )
  md = md.replace(/^(#{3,6})(?=\[(?:구성요소|추가\s*구성))/gm, '$1 ')
  md = md.replace(
    /^#{1,6}\s*(\[(?:구성요소|추가\s*구성|전제부)(?:\s*\([A-Z](?:-\d+)?\))?\])\s*$/gm,
    '$1'
  )
  md = keepFirstComponentHeader(md)
  md = normalizePhase1Fields(md)
  md = mergeClaimIntoJudgmentCards(md)
  md = normalizeSummaryItems(md)
  md = md.replace(
    /([^\n])\s+(-\s*(유사점 요약|차이점)\s*:)/g,
    '$1\n\n$2'
  )
  md = md.replace(
    /^-\s*(유사점 요약|차이점)\s*:\s*(.+)$/gm,
    (_, label, body) => {
      const heading =
        label === '유사점 요약'
          ? '[유사점]'
          : '[차이점]'
      return `${heading}\n\n${body.trim()}`
    }
  )
  md = md.replace(
    /([^\n])\n(-\s*(유사점 요약|차이점)\s*:)/g,
    '$1\n\n$2'
  )
  md = md.replace(
    /([^\n])\s+(대응 이유\s*:)/g,
    '$1\n\n$2'
  )
  md = md.replace(
    /([^\n])\n(\([A-Z](?:-\d+)?\))/g,
    '$1\n\n$2'
  )
  md = md.replace(
    /^(\([A-Z](?:-\d+)?\)(?:\s*(?:및|,)\s*\([A-Z](?:-\d+)?\))*\s+(?:동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음|대응안됨|대응 안됨)(?:\s+\d+%)?)\s+(\S.*)$/gm,
    (_, judgment, body) => `${judgment}\n\n${body}`
  )
  md = normalizeDifferenceEntries(md)
  md = md.replace(
    /\s+((?:다만|따라서)\s+)/g,
    '\n\n$1'
  )
  const lines = md.split('\n')
  const result = []
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const isJudgment = /^\([A-Z](?:-\d+)?\)\s/.test(line)
    if (isJudgment && result.length > 0 && result[result.length - 1] !== '') result.push('')
    result.push(line)
    if (isJudgment && i < lines.length - 1 && lines[i + 1] !== '') result.push('')
  }
  return sanitizeReportText(result.join('\n'))
}

function preprocessPhase1Report(md) {
  function moveSimilarityBeforeClaim(text) {
    const componentHeaderRe = /^\s*(?:#{1,6}\s*)?\[\s*(?:구성요소|추가\s*구성|추가구성|전제부|종속항|청구항\s*\d*(?:\s*추가\s*구성)?)(?:\s*[:\-]?\s*(?:\([^)]*\)|[A-Za-z0-9-]+))?\s*\]\s*$/i
    const sectionHeaderRe = /^\s*#{1,6}\s*\[/
    const similarityRe = /^\s*(?:-\s*)?(?:\*\*)?유사도\s*평가(?:\*\*)?\s*:/
    const claimRe = /^\s*(?:-\s*)?(?:\*\*)?(?:청구항\s*(?:추가\s*)?구성|추가\s*구성)(?:\s*\([^)]*\))?(?:\*\*)?\s*:/
    const legacyClaimRe = /^\s*#{1,6}\s*(?:청구항\s*구성|추가\s*구성)\s*\([A-Z](?:-\d+)?\)/
    const lines = String(text || '').split('\n')
    const output = []
    let component = null

    function flushComponent() {
      if (!component) return
      const similarityIndex = component.findIndex(line => similarityRe.test(line))
      const claimIndex = component.findIndex(line => claimRe.test(line) || legacyClaimRe.test(line))
      if (similarityIndex > claimIndex && claimIndex >= 0) {
        const [similarityLine] = component.splice(similarityIndex, 1)
        component.splice(claimIndex, 0, similarityLine)
      }
      output.push(...component)
      component = null
    }

    for (const line of lines) {
      if (componentHeaderRe.test(line)) {
        flushComponent()
        component = [line]
        continue
      }
      if (component && sectionHeaderRe.test(line)) {
        flushComponent()
        output.push(line)
        continue
      }
      if (component) component.push(line)
      else output.push(line)
    }
    flushComponent()
    return output.join('\n')
  }

  return preprocessReport(
    String(md || '')
      .replace(/\r\n?/g, '\n')
      .replace(/^###\s+claim\s+(\d+)\s*$/gim, '### 청구항 $1')
      .replace(/^\s*(?:#{1,6}\s*)?\[?\s*종합\s*분석\s*요약\s*\]?\s*$/gim, '[종합분석요약]')
  )
}

function extractRejectionBasisHeader(md) {
  const match = String(md || '').match(
    /^\s*\[인용발명\s*\d+\s*과\s*인용발명\s*\d+\s*의\s*결합(?:\s*및\s*주지관용)?\s*\(진보성\)\]\s*$/m
  )
  return match ? match[0].trim() : ''
}

function removeRejectionBasisHeader(md) {
  return String(md || '').replace(
    /^\s*\[인용발명\s*\d+\s*과\s*인용발명\s*\d+\s*의\s*결합(?:\s*및\s*주지관용)?\s*\(진보성\)\]\s*$/m,
    ''
  ).replace(/^\s*\n/, '')
}

function removeQuantitativeAssessment(md) {
  return String(md || '').replace(
    /^\s*\[정량평가\s*-\s*분석 보조지표\]\s*$[\s\S]*?(?=^\s*\[(?!종속항 추가한정 평가\])[^\]]+\]\s*$|(?![\s\S]))/m,
    ''
  ).replace(/^\s*\n/, '')
}

// ── [확장 포인트 1] 청구항 헤더 패턴 ─────────────────────────────────────────
// 새 헤더 형식 발견 시 여기에 추가.
// 조건: 줄 전체가 청구항 번호만으로 구성되어야 함 (번호 뒤에 텍스트 없음)
// 각 패턴은 캡처 그룹 1에 청구항 번호(digits)를 반환해야 함
const CLAIM_HEADER_PATTERNS = [
  /^청구항\s*(\d+)[.．]?\s*$/,          // "청구항 1" / "청구항 1." (가장 일반적)
  /^제\s*(\d+)\s*항[.．]?\s*$/,         // "제1항" / "제 1 항."
]


// Bracketed headers may be followed immediately by the claim body.
// Example: "[CLAIM 2]..." or a bracketed Korean claim header.
const EXPLICIT_CLAIM_HEADER_PATTERNS = [
  /^\u3010\s*\uCCAD\uAD6C\uD56D\s*(\d+)\s*\u3011[.\uFF0E]?\s*(.*)$/,
  /^\[\s*CLAIM\s*(\d+)\s*\][.\uFF0E]?\s*(.*)$/i,
]

// ── [확장 포인트 2] 종속항 참조 마커 ─────────────────────────────────────────
// 이 패턴이 포함된 줄은 헤더로 오인하지 않음 (본문 안에서 선행 청구항 인용)
// 새 종속 표현 발견 시 여기에 추가
const DEPENDENT_REF_PATTERNS = [
  /청구항\s*\d+에\s*있어서/,
  /청구항\s*\d+\s*내지\s*청구항\s*\d+.*있어서/,
  /청구항\s*\d+\s*내지\s*\d+.*있어서/,
  /청구항\s*\d+\s*또는\s*청구항\s*\d+.*있어서/,
  /제\s*\d+\s*항에\s*있어서/,
  /제\s*\d+\s*항\s*내지\s*제\s*\d+\s*항.*있어서/,   // "제1항 내지 제3항 중 어느 한 항에 있어서"
  /제\s*\d+\s*항\s*또는\s*제\s*\d+\s*항.*있어서/,   // "제1항 또는 제2항에 있어서"
]

// 줄이 청구항 헤더인지 판별. 헤더면 { number } 반환, 아니면 null
function matchClaimHeader(line) {
  const t = line.trim()
  // Check explicit headers first so an inline dependency phrase does not hide
  // the real claim boundary.
  for (const pattern of EXPLICIT_CLAIM_HEADER_PATTERNS) {
    const m = t.match(pattern)
    if (m) return { number: parseInt(m[1]), inlineText: (m[2] || '').trim() }
  }

  // 종속항 참조 표현이 포함되면 헤더가 아님
  if (DEPENDENT_REF_PATTERNS.some(p => p.test(t))) return null
  for (const pattern of CLAIM_HEADER_PATTERNS) {
    const m = t.match(pattern)
    if (m) return { number: parseInt(m[1]), inlineText: '' }
  }
  return null
}

// 복수 청구항 텍스트 분리
// ─────────────────────────────────────────────────────────────────────────────
// 분리 규칙:
//   [R1] "청구항 N" 단독 줄 → 새 청구항 시작  (CLAIM_HEADER_PATTERNS)
//   [R2] "청구항 N에 있어서" 등 → 종속항 참조, 분리점 아님  (DEPENDENT_REF_PATTERNS)
//   [R3] 번호 단조 증가 검증으로 오파싱 감지
//   [R4] 마침표/어미는 신뢰하지 않음 (오타 가능성)
//
// 새 예외 추가 방법:
//   - 헤더 형식이 다르면 → CLAIM_HEADER_PATTERNS에 정규식 추가
//   - 종속항 표현이 헤더로 오인되면 → DEPENDENT_REF_PATTERNS에 추가
// ─────────────────────────────────────────────────────────────────────────────
function splitClaims(text) {
  const trimmed = text.trim()
  const lines = trimmed.split('\n')

  // ── 1단계: 헤더 줄 위치 수집 ──────────────────────────────────────────────
  const starts = []
  for (let i = 0; i < lines.length; i++) {
    const h = matchClaimHeader(lines[i])
    if (h) starts.push({ i, number: h.number, inlineText: h.inlineText })
  }



  // ── 2단계: 번호 단조 증가 검증 (오파싱 필터) ──────────────────────────────
  // 번호가 순서대로가 아닌 항목은 헤더가 아닌 것으로 제거
  const validStarts = starts.filter((s, idx) => {
    if (idx === 0) return true
    return s.number > starts[idx - 1].number
  })

  if (validStarts.length >= 1) {
    const result = validStarts.map(({ i, number, inlineText }, idx) => {
      const nextStart = idx + 1 < validStarts.length ? validStarts[idx + 1].i : lines.length
      const bodyLines = lines.slice(i + 1, nextStart)
      if (inlineText) bodyLines.unshift(inlineText)
      const claimText = bodyLines.join('\n').trim()
      return claimText ? { number, text: claimText } : null
    }).filter(Boolean)
    if (result.length >= 1) return result
  }

  // ── 폴백: "N. 텍스트" 인라인 형식 (번호+마침표+공백) ─────────────────────
  const partsA = trimmed.split(/(?=^(?:청구항\s*)?\d+[.．][ \t])/m)
    .map(s => s.trim()).filter(Boolean)
  if (partsA.length > 1) {
    const result = partsA.map(part => {
      const m = part.match(/^(?:청구항\s*)?(\d+)[.．][ \t]*/)
      if (!m) return null
      return { number: parseInt(m[1]), text: part.slice(m[0].length).trim() }
    }).filter(Boolean)
    if (result.length >= 1) return result
  }

  return null  // 단일 청구항
}

// ── 히스토리 로드/저장 헬퍼 ──────────────────────────────────────────────────
function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem('aria_history') || '[]')
      .map(item => ({
        ...item,
        report: sanitizeReportText(item.report || ''),
        relatedInventionsMd: sanitizeReportText(item.relatedInventionsMd || ''),
      }))
  }
  catch { return [] }
}
function saveHistory(list) {
  localStorage.setItem(
    'aria_history',
    JSON.stringify(list.map(item => ({
      ...item,
      report: sanitizeReportText(item.report || ''),
      relatedInventionsMd: sanitizeReportText(item.relatedInventionsMd || ''),
    })))
  )
}

function formatDate(iso) {
  const d = new Date(iso)
  const pad = n => String(n).padStart(2, '0')
  return `${d.getFullYear()}.${pad(d.getMonth()+1)}.${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
}

// ── 히스토리 사이드 패널 ──────────────────────────────────────────────────────
function HistoryPanel({ history, onSelect, onDelete, onClearLocal, onClearAll, onClose }) {
  return (
    <>
      {/* 배경 오버레이 */}
      <div
        className="fixed inset-0 bg-black/30 z-40 transition-opacity"
        onClick={onClose}
      />
      {/* 사이드 드로어 */}
      <div className="fixed top-0 left-0 h-full w-72 bg-white shadow-2xl z-50 flex flex-col">
        {/* 헤더 */}
        <div className="flex items-center justify-between px-4 py-4 border-b bg-gray-50">
          <div className="flex items-center gap-2">
            <span className="text-base font-bold text-gray-800">🕘 히스토리</span>
            <span className="text-xs text-gray-400">{history.length}건</span>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-700 text-xl leading-none"
          >×</button>
        </div>

        {/* 전체 삭제 분할 */}
        <div className="px-4 py-2 border-b flex flex-col gap-1.5">
          {history.length > 0 && (
            <button
              onClick={onClearLocal}
              className="w-full text-xs text-gray-600 hover:text-gray-800 border border-gray-200 rounded-lg py-1.5 hover:bg-gray-50 transition font-medium"
              title="브라우저 히스토리 목록만 지우고 서버 파일은 보존합니다."
            >
              목록만 전체 삭제
            </button>
          )}
          <button
            onClick={onClearAll}
            className="w-full text-xs text-red-500 hover:text-red-700 border border-red-200 rounded-lg py-1.5 hover:bg-red-50 transition font-medium"
            title="히스토리 목록을 비우고 서버의 업로드 파일 및 보고서도 모두 삭제합니다."
          >
            서버 데이터 포함 전체 삭제
          </button>
        </div>

        {/* 목록 */}
        <div className="flex-1 overflow-y-auto py-2">
          {history.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-32 text-gray-400 text-sm gap-2">
              <span className="text-3xl">📋</span>
              <p>보고서 기록이 없습니다</p>
            </div>
          ) : (
            history.map(item => (
              <div
                key={item.id}
                className="group flex items-start gap-2 px-4 py-3 hover:bg-gray-50 cursor-pointer border-b border-gray-50 transition"
                onClick={() => { onSelect(item); onClose() }}
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                    <span className="text-xs font-semibold text-blue-600 bg-blue-50 px-2 py-0.5 rounded-full shrink-0">
                      청구항 {item.claimNumber}
                    </span>
                    {item.missingPriorArt && (
                      <span className="text-[10px] font-medium text-indigo-700 bg-indigo-50 px-1.5 py-0.5 rounded-full shrink-0">
                        선행기술 검색
                      </span>
                    )}
                    {item.usedInventions && item.usedInventions.length > 0 && (
                      <span className="text-[10px] text-gray-500 bg-gray-100 px-1.5 py-0.5 rounded-full shrink-0">
                        {item.usedInventions.map(inv => inv.name).join(' + ')}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-600 leading-relaxed line-clamp-2 mt-1">
                    {item.claimTextPreview}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">{formatDate(item.createdAt)}</p>
                </div>
                <button
                  onClick={e => { e.stopPropagation(); onDelete(item.id) }}
                  className="shrink-0 opacity-0 group-hover:opacity-100 text-gray-400 hover:text-red-500 transition text-base leading-none mt-0.5"
                  title="목록에서 제거 (서버 파일 유지)"
                >×</button>
              </div>
            ))
          )}
        </div>
      </div>
    </>
  )
}

const missingPriorArtMarkdownComponents = {
  a: ({ node: _node, className = '', ...props }) => (
    <a
      {...props}
      target="_blank"
      rel="noreferrer"
      className={`break-all text-indigo-700 underline decoration-indigo-300 underline-offset-2 hover:text-indigo-900 ${className}`}
    />
  ),
}

function MissingPriorArtResultPage({ result, onBack }) {
  const content = splitMissingPriorArtMarkdown(result.result_md || '')
  return (
    <div className="mx-auto max-w-5xl px-6 py-5">
      <div className="mb-5 flex items-center gap-3 border-b border-slate-200 pb-4">
        <button
          type="button"
          onClick={onBack}
          className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50"
        >
          ← 보고서로
        </button>
        <div>
          <h2 className="text-lg font-bold text-slate-900">미커버 구성 선행기술 검색 결과</h2>
          <p className="text-xs text-slate-500">청구항 {result.claim_number} · {result.searched_at}</p>
        </div>
      </div>
      <div className="mb-4 flex flex-wrap items-center gap-2 text-xs">
        <span className="font-semibold text-slate-700">검색 구성</span>
        {(result.target_labels || []).map(label => (
          <span key={label} className="rounded-full bg-indigo-100 px-2 py-0.5 font-medium text-indigo-700">{label}</span>
        ))}
        <span className="rounded-full bg-slate-100 px-2 py-0.5 text-slate-600">
          {(result.search_axes || []).join(' · ')}
        </span>
        {result.expanded && (
          <span className="rounded-full bg-amber-100 px-2 py-0.5 font-medium text-amber-700">후보 부족으로 자동 확장</span>
        )}
      </div>
      {content.candidates.length > 0 ? (
        <div className="space-y-5">
          {content.prefix && (
            <div className="prose prose-sm max-w-none rounded-xl border border-slate-200 bg-white p-5 text-slate-700">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={missingPriorArtMarkdownComponents}>
                {content.prefix}
              </ReactMarkdown>
            </div>
          )}
          <section>
            <h3 className="mb-3 text-base font-bold text-slate-900">후보 문헌</h3>
            <div className="space-y-4">
              {content.candidates.map((candidate, index) => (
                <article
                  key={`${candidate.title}-${index}`}
                  className="rounded-xl border border-indigo-100 bg-white p-5 shadow-sm"
                >
                  <div className="mb-3 flex flex-wrap items-start justify-between gap-2 border-b border-slate-100 pb-3">
                    <h4 className="text-sm font-bold leading-6 text-slate-900">{candidate.title}</h4>
                    {candidate.category && (
                      <span className="rounded-full bg-indigo-50 px-2.5 py-1 text-[11px] font-semibold text-indigo-700">
                        {candidate.category}
                      </span>
                    )}
                  </div>
                  <div className="prose prose-sm max-w-none text-slate-700 prose-li:my-1.5">
                    <ReactMarkdown remarkPlugins={[remarkGfm]} components={missingPriorArtMarkdownComponents}>
                      {candidate.body}
                    </ReactMarkdown>
                  </div>
                </article>
              ))}
            </div>
          </section>
          {content.suffix && (
            <div className="prose prose-sm max-w-none rounded-xl border border-slate-200 bg-white p-5 text-slate-700">
              <ReactMarkdown remarkPlugins={[remarkGfm]} components={missingPriorArtMarkdownComponents}>
                {content.suffix}
              </ReactMarkdown>
            </div>
          )}
        </div>
      ) : (
        <div className="prose prose-sm max-w-none rounded-xl border border-slate-200 bg-white p-5 text-slate-700">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={missingPriorArtMarkdownComponents}>
            {content.prefix}
          </ReactMarkdown>
        </div>
      )}
    </div>
  )
}

function parseRelatedASections(markdown) {
  const lines = sanitizeReportText(markdown).split(/\r?\n/)
  const sections = []
  let current = null
  let currentCard = null

  const startSection = (title = '') => {
    current = { title, intro: [], cards: [] }
    sections.push(current)
    currentCard = null
  }

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed === '---') continue
    if (/^##\s*관련도\s*A\s*인용발명\s*$/.test(trimmed)) continue

    const claimMatch = trimmed.match(/^###\s+청구항\s+(.+)$/)
    if (claimMatch) {
      startSection(`청구항 ${claimMatch[1]}`)
      continue
    }

    const inventionMatch = trimmed.match(/^\*\*(.+?)\*\*\s*(?:\((.*?)\))?\s*$/)
    if (inventionMatch) {
      if (!current) startSection()
      currentCard = {
        name: inventionMatch[1].trim(),
        filename: (inventionMatch[2] || '').trim(),
        body: [],
      }
      current.cards.push(currentCard)
      continue
    }

    if (!current) startSection()
    if (currentCard) currentCard.body.push(line)
    else current.intro.push(line)
  }

  return sections.filter(section => section.title || section.intro.some(Boolean) || section.cards.length > 0)
}

function RelatedAReport({ markdown }) {
  const sections = parseRelatedASections(markdown)
  return (
    <div className="related-a-report space-y-7">
      {sections.map((section, sectionIndex) => (
        <section key={`${section.title}-${sectionIndex}`} className="space-y-4">
          {section.title && (
            <h2 className="mb-0 flex items-center gap-2 border-b border-slate-200 pb-3 text-lg font-bold text-slate-900">
              <span className="h-5 w-1 rounded-full bg-blue-500" />
              {section.title}
            </h2>
          )}
          {section.intro.some(Boolean) && (
            <div className="prose prose-sm max-w-none rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-slate-600">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{section.intro.join('\n')}</ReactMarkdown>
            </div>
          )}
          {section.cards.length > 0 && (
            <div className="grid gap-4 grid-cols-1">
              {section.cards.map((card, cardIndex) => (
                <article
                  key={`${card.name}-${cardIndex}`}
                  className="rounded-xl border border-blue-100 bg-white p-5 shadow-sm ring-1 ring-slate-100"
                >
                  <header className="mb-4 border-b border-slate-100 pb-3">
                    <h3 className="text-base font-bold leading-6 text-slate-900">{card.name}</h3>
                    {card.filename && (
                      <p className="mt-1 break-all text-xs font-medium text-slate-500">{card.filename}</p>
                    )}
                  </header>
                  <div className="prose prose-sm max-w-none text-slate-700 prose-li:my-1.5">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{card.body.join('\n').trim()}</ReactMarkdown>
                  </div>
                </article>
              ))}
            </div>
          )}
        </section>
      ))}
    </div>
  )
}

// ── 메인 앱 ───────────────────────────────────────────────────────────────────
export default function App() {
  const [priorFiles, setPriorFiles] = useState([])
  const [jobId, setJobId] = useState(null)
  const [priorReady, setPriorReady] = useState(false)
  const [loading, setLoading] = useState(false)
  const [generating, setGenerating] = useState(false)
  const cancelRef = useRef(null)  // 생성 취소용: { requested, es, abort, reject }
  const [logs, setLogs] = useState([])
  const [uploadProgress, setUploadProgress] = useState(0)
  const [claimText, setClaimText] = useState('')
  const [claimNumber, setClaimNumber] = useState(1)
  const [claims, setClaims] = useState([])
  const [report, setReport] = useState('')
  const [usedInventions, setUsedInventions] = useState([])  // 실제 사용된 인용발명 목록
  const [allReports, setAllReports] = useState({})  // { claimNum: { report_md, usedInventions } }
  const [activeClaimNumView, setActiveClaimNumView] = useState(null)
  const [error, setError] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [showClaimAnalysis, setShowClaimAnalysis] = useState(false)
  const [missingPriorArt, setMissingPriorArt] = useState(null)
  const [showMissingPriorArt, setShowMissingPriorArt] = useState(false)
  const rejectionBasisHeader = extractRejectionBasisHeader(report)
  const citationBasisLabel = usedInventions[0]?.basis_label
    || rejectionBasisHeader.replace(/^\[|\]$/g, '')
  const reportWithoutRejectionBasisHeader = removeRejectionBasisHeader(report)
  const reportForDisplay = removeQuantitativeAssessment(reportWithoutRejectionBasisHeader)

  // 히스토리
  const [history, setHistory] = useState(loadHistory)
  const [showHistory, setShowHistory] = useState(false)
  const [showChat, setShowChat] = useState(false)

  // 컨텍스트 상태
  const [contextClaims, setContextClaims] = useState([])   // 현재 컨텍스트에 있는 청구항 번호 목록
  const [useCtx, setUseCtx] = useState(true)               // 다음 생성에 컨텍스트 사용 여부

  const addLog = msg => setLogs(prev => [...prev, msg])

  function handlePriorFilesChange(nextFiles) {
    setPriorFiles(nextFiles)
    setJobId(null)
    setPriorReady(false)
    setUploadProgress(0)
    setClaims([])
    setReport('')
    setUsedInventions([])
    setAllReports({})
    setActiveClaimNumView(null)
    setContextClaims([])
    setUseCtx(true)
    setError('')
    setMissingPriorArt(null)
    setShowMissingPriorArt(false)
  }

  // 히스토리 저장
  function addHistoryItem(item) {
    setHistory(prev => {
      const updated = [item, ...prev].slice(0, 50)
      saveHistory(updated)
      return updated
    })
  }

  function handleMissingPriorArtResult(result) {
    setMissingPriorArt(result)
    setShowMissingPriorArt(true)
    setHistory(prev => {
      const claimNo = Number(result.claim_number)
      const updated = prev.map(item =>
        item.jobId === jobId && Number(item.claimNumber) === claimNo
          ? { ...item, missingPriorArt: result }
          : item
      )
      saveHistory(updated)
      return updated
    })
  }
  // 히스토리 개별 항목 삭제 (서버 데이터 포함 삭제)
  async function deleteHistoryItem(id) {
    const target = history.find(h => h.id === id)
    if (target && target.jobId) {
      try {
        await deleteJob(target.jobId)
        addLog(`[히스토리 삭제] 서버의 연관 작업 데이터(${target.jobId})를 삭제했습니다.`)
      } catch (e) {
        console.warn('서버 개별 작업 삭제 실패:', e)
      }
    }
    setHistory(prev => {
      const updated = prev.filter(h => h.id !== id)
      saveHistory(updated)
      return updated
    })
  }

  // 히스토리 로컬 목록만 전체 삭제
  function clearHistoryLocalOnly() {
    setHistory([])
    localStorage.removeItem('aria_history')
    addLog('[히스토리] 브라우저 히스토리 목록을 비웠습니다. (서버 파일은 유지됨)')
  }

  // 히스토리 목록 및 서버 데이터까지 전체 삭제
  async function clearHistoryWithServer() {
    await deleteAllJobs()
    setHistory([])
    localStorage.removeItem('aria_history')
    setJobId(null)
    setPriorReady(false)
    setPriorFiles([])
    setClaims([])
    setClaimText('')
    setClaimNumber(1)
    setReport('')
    setUsedInventions([])
    setAllReports({})
    setActiveClaimNumView(null)
    setContextClaims([])
    setUseCtx(true)
    setError('')
    setMissingPriorArt(null)
    setShowMissingPriorArt(false)
    addLog('[히스토리] 히스토리 목록 및 연관된 서버 데이터를 모두 삭제했습니다.')
  }

  async function handleDeleteCurrentJob() {
    if (!jobId) return
    if (!confirm('현재 작업의 업로드 파일, 보고서, 케이스 데이터를 모두 삭제할까요?')) return
    try {
      await deleteJob(jobId)
      const nextHistory = history.filter(item => item.jobId !== jobId)
      setHistory(nextHistory)
      saveHistory(nextHistory)
      setJobId(null)
      setPriorReady(false)
      setPriorFiles([])
      setClaims([])
      setClaimText('')
      setClaimNumber(1)
      setReport('')
      setUsedInventions([])
      setAllReports({})
      setActiveClaimNumView(null)
      setContextClaims([])
      setUseCtx(true)
      setError('')
      addLog('[현재 작업 삭제] 서버에 저장된 현재 작업 데이터를 삭제했습니다.')
    } catch (e) {
      addLog(`[오류] 현재 작업 삭제 실패: ${e.message}`)
    }
  }

  async function loadHistoryItem(item) {
    const selectedClaimNumber = Number(item.claimNumber)
    const relatedItems = item.jobId
      ? history.filter(h => h.jobId === item.jobId && h.report)
      : [item]
    let reportsByClaim = {}
    for (const h of relatedItems) {
      const claimKey = Number(h.claimNumber)
      if (!Number.isFinite(claimKey) || reportsByClaim[claimKey]) continue
      reportsByClaim = addReportEntryWithRelated(
        reportsByClaim,
        claimKey,
        h.report,
        h.usedInventions || [],
        h.relatedInventionsMd || '',
      )
    }
    if (!reportsByClaim[selectedClaimNumber]) {
      reportsByClaim = addReportEntryWithRelated(
        reportsByClaim,
        selectedClaimNumber,
        item.report,
        item.usedInventions || [],
        item.relatedInventionsMd || '',
      )
    }
    const selectedReport = reportsByClaim[selectedClaimNumber]
    if (!selectedReport?.report_md) {
      addLog(`[히스토리] 청구항 ${selectedClaimNumber}에 저장된 보고서 본문이 없습니다.`)
      return
    }
    setReport(selectedReport.report_md)
    setUsedInventions(selectedReport.usedInventions)
    setAllReports(reportsByClaim)
    setActiveClaimNumView(selectedClaimNumber)
    setClaimNumber(selectedClaimNumber)
    setClaimText(item.claimTextPreview)
    let restoredSearch = item.missingPriorArt || null
    if (!restoredSearch && item.jobId) {
      try {
        restoredSearch = await getMissingPriorArt(item.jobId, selectedClaimNumber)
      } catch (_) {}
    }
    setMissingPriorArt(restoredSearch)
    setShowMissingPriorArt(Boolean(restoredSearch))
    addLog(
      `[히스토리] 청구항 ${Object.keys(reportsByClaim).sort((a, b) => Number(a) - Number(b)).join(', ')} 보고서를 복원했습니다.`
    )

    if (item.jobId) {
      try {
        const status = await checkJobStatus(item.jobId)
        if (status.exists) {
          setJobId(item.jobId)
          setPriorReady(true)
          addLog(`히스토리 복원 — 인용발명 ${status.prior_count}개 서버에서 재사용 가능`)
        } else {
          setJobId(null)
          setPriorReady(false)
          addLog('히스토리 복원 — 서버 파일이 만료됐습니다. 인용발명을 다시 업로드해주세요.')
        }
      } catch (_) {
        setJobId(null)
        setPriorReady(false)
      }
    }
  }

  async function handlePrepare() {
    if (priorFiles.length === 0) return
    setLoading(true)
    setUploadProgress(0)
    setLogs([])
    setClaims([])
    setReport('')
    setAllReports({})
    setActiveClaimNumView(null)
    setError('')
    setPriorReady(false)
    try {
      const { job_id } = await uploadFiles(priorFiles, p => setUploadProgress(p))
      setJobId(job_id)
      addLog('파일 업로드 완료')
      await new Promise((resolve, reject) => {
        streamPrepare(job_id, {
          onLog: msg => addLog(msg),
          onDone: data => {
            addLog(`인용발명 ${data.prior_count || priorFiles.length}개 준비 완료`)
            setPriorReady(true)
            resolve()
          },
          onError: err => reject(new Error(err)),
        })
      })
    } catch (err) {
      setError(err.message)
      addLog(`오류: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  async function handleGenerate() {
    if (!jobId || !claimText.trim()) return
    setGenerating(true)
    cancelRef.current = { requested: false, es: null, abort: null, reject: null }
    setReport('')
    setError('')
    setUsedInventions([])
    setAllReports({})
    setActiveClaimNumView(null)

    try {
      const multiClaims = splitClaims(claimText.trim())
      const toProcess = multiClaims
        ? (() => { addLog(`청구항 ${multiClaims.length}개 감지 → 순차 처리`); return multiClaims })()
        : [{ number: Number(claimNumber) || 1, text: claimText.trim() }]

      const registered = []
      for (const { number, text } of toProcess) {
        addLog(`청구항 ${number} 구성요소 분해 중`)
        const isDependent = DEPENDENT_REF_PATTERNS.some(p => p.test(text))
        const claim = await addManualClaim(jobId, {
          claim_text: text,
          claim_number: number,
          claim_type: isDependent ? 'dependent' : 'independent'
        })
        const resolvedDependent = claim.claim_type === 'dependent'
        registered.push({ claim, text, isDependent: resolvedDependent })
      }

      const independents = registered.filter(r => !r.isDependent)
      const dependents = registered.filter(r => r.isDependent)

      for (const { claim, text } of independents) {
        if (cancelRef.current?.requested) throw new Error('사용자 취소')
        addLog(`청구항 ${claim.claim_number} 분석 중... (독립항)`)
        setActiveClaimNumView(claim.claim_number)
        
        await new Promise((resolve, reject) => {
          cancelRef.current.reject = reject
          const es = streamReport(jobId, claim.claim_number, {
            onLog: msg => addLog(msg),
            onStreamChunk: chunk => setReport(prev => sanitizeReportText(prev + chunk)),
            onPhase1: data => {
              setReport(sanitizeReportText(data.phase1_md || ''))
              setUsedInventions(data.used_inventions || [])
            },
            onDone: async data => {
              const relatedInventionsMd = sanitizeReportText(data.related_inventions_md || '')
              const { reportMd: cleanReport, relatedMd: extractedRelated } = splitRelatedAReport(data.report_md)
              const cleanRelated = relatedInventionsMd || extractedRelated
              setReport(cleanReport)
              setUsedInventions(data.used_inventions || [])
              if (data.timings) {
                const order = ['comparison', 'citation chain', 'quote verification', 'phase1', 'finalize', 'total']
                const summary = order
                  .filter(key => data.timings[key] != null)
                  .map(key => `${key} ${Number(data.timings[key]).toFixed(1)}s`)
                  .join(' | ')
                if (summary) addLog(`[timing] ${summary}`)
              }
              setAllReports(prev => addReportEntryWithRelated(
                prev,
                claim.claim_number,
                cleanReport,
                data.used_inventions || [],
                cleanRelated,
              ))
              addHistoryItem({
                id: Date.now() + claim.claim_number,
                jobId: jobId,
                claimNumber: claim.claim_number,
                claimTextPreview: text.slice(0, 100),
                report: cleanReport,
                relatedInventionsMd: cleanRelated,
                usedInventions: data.used_inventions || [],
                createdAt: new Date().toISOString(),
              })
              try {
                const ctx = await getContextInfo(jobId)
                setContextClaims(ctx.context_claims || [])
                setUseCtx(true)
              } catch (_) {}
              resolve()
            },
            onError: err => reject(new Error(err)),
          }, useCtx)
          cancelRef.current.es = es
        })
      }

      // 종속항은 한 번의 배치 요청으로 일괄 생성한다.
      if (dependents.length > 0) {
        if (cancelRef.current?.requested) throw new Error('사용자 취소')
        addLog(`종속항 ${dependents.length}개 일괄 생성 중… (LLM 1회 호출)`)
        try {
          const ac = new AbortController()
          cancelRef.current.abort = ac
          let statusPollActive = true
          let lastStatusKey = ''
          const statusPoll = (async () => {
            while (statusPollActive && !cancelRef.current?.requested) {
              try {
                const status = await getDependentBatchStatus(jobId)
                const statusKey = [status.state, status.stage, status.message, status.reports_ready].join('::')
                if (status.message && statusKey !== lastStatusKey) {
                  lastStatusKey = statusKey
                  const readySuffix = typeof status.reports_ready === 'number' && status.reports_ready > 0
                    ? ` (완료 ${status.reports_ready}건)`
                    : ''
                  addLog(`[종속항 상태] ${status.message}${readySuffix}`)
                }
                if (status.state === 'completed' || status.state === 'failed') break
              } catch (_) {}
              await new Promise(resolve => setTimeout(resolve, 2000))
            }
          })()
          let reports
          try {
            ;({ reports } = await reportBatchDependent(
              jobId, dependents.map(d => d.claim.claim_number), useCtx, false, ac.signal,
            ))
          } finally {
            statusPollActive = false
            await statusPoll.catch(() => {})
          }
          for (const { claim, text } of dependents) {
            const r = reports[String(claim.claim_number)]
            if (!r) {
              addLog(`경고: 청구항 ${claim.claim_number} 보고서가 누락되었습니다.`)
              continue
            }
            const relatedInventionsMd = sanitizeReportText(r.related_inventions_md || '')
            const { reportMd: cleanReport, relatedMd: extractedRelated } = splitRelatedAReport(r.report_md)
            const cleanRelated = relatedInventionsMd || extractedRelated
            setReport(cleanReport)
            setUsedInventions(r.used_inventions || [])
            setActiveClaimNumView(claim.claim_number)
            setAllReports(prev => addReportEntryWithRelated(
              prev,
              claim.claim_number,
              cleanReport,
              r.used_inventions || [],
              cleanRelated,
            ))
            addHistoryItem({
              id: Date.now() + claim.claim_number,
              jobId: jobId,
              claimNumber: claim.claim_number,
              claimTextPreview: text.slice(0, 100),
              report: cleanReport,
              relatedInventionsMd: cleanRelated,
              usedInventions: r.used_inventions || [],
              createdAt: new Date().toISOString(),
            })
          }
          addLog(`✅ 종속항 ${dependents.length}개 일괄 생성 완료`)
          try {
            const ctx = await getContextInfo(jobId)
            setContextClaims(ctx.context_claims || [])
            setUseCtx(true)
          } catch (_) {}
        } catch (e) {
          if (e.name === 'AbortError' || cancelRef.current?.requested) {
            throw new Error('사용자 취소')
          }
          addLog(`종속항 일괄 생성 실패: ${e.message}`)
          setError(e.message)
        }
      }
    } catch (err) {
      if (err.message === '사용자 취소') {
        addLog('사용자가 보고서 생성을 취소했습니다.')
      } else {
        setError(err.message)
        addLog(`오류: ${err.message}`)
      }
    } finally {
      setGenerating(false)
      cancelRef.current = null
    }
  }

  // 보고서 생성 취소 — 스트림 종료 + 실행 중 LLM CLI 프로세스 강제 종료
  async function handleCancelGenerate() {
    const c = cancelRef.current
    if (!c || c.requested) return
    c.requested = true
    addLog('🛑 생성 취소 요청 — 실행 중인 LLM 프로세스를 종료합니다…')
    try { c.es?.close() } catch (_) {}
    try { c.abort?.abort() } catch (_) {}
    try {
      const { killed } = await cancelGeneration()
      if (killed > 0) addLog(`🛑 LLM 프로세스 ${killed}개 종료됨`)
    } catch (_) {}
    c.reject?.(new Error('사용자 취소'))
  }

  const tabKeys = reportTabKeys(allReports)
  const isRelatedATabActive = activeClaimNumView === RELATED_A_TAB_KEY
  const activeClaimForActions = typeof activeClaimNumView === 'number' ? activeClaimNumView : claimNumber

  return (
    <div className="app-shell min-h-screen flex flex-col">

      {/* 헤더 */}
      <header className="app-header flex items-center justify-between px-7 py-4 shrink-0">
        <div className="flex items-center gap-3">
          <AriaEmblem />
          <div className="flex flex-col justify-center ml-1">
            <div className="flex items-baseline gap-1.5">
              <span className="aria-wordmark">ARIA</span>
            </div>
            <span className="aria-sub">
              <span className="text-sky-500 font-bold text-[0.65rem]">A</span>I{' '}
              <span className="text-sky-500 font-bold text-[0.65rem]">R</span>EPORT{' '}
              <span className="text-sky-500 font-bold text-[0.65rem]">I</span>NTELLIGENCE{' '}
              <span className="text-sky-500 font-bold text-[0.65rem]">A</span>SSISTANT
            </span>
          </div>
        </div>

        <div className="flex-1" />

        <div className="flex items-center gap-2">
          {jobId && (
            <button
              className="ghost-button danger"
              onClick={handleDeleteCurrentJob}
              disabled={generating || loading}
              title="현재 작업의 uploads, reports, cases 저장 데이터를 서버에서 삭제합니다."
            >
              현재 작업 삭제
            </button>
          )}
          <button
            className="ghost-button flex items-center gap-1.5"
            onClick={() => setShowHistory(true)}
          >
            <span>히스토리</span>
            {history.length > 0 && (
              <span className="bg-blue-100 text-blue-600 text-xs font-semibold rounded-full px-1.5 py-0.5 leading-none">
                {history.length}
              </span>
            )}
          </button>
          <button
            className="ghost-button"
            onClick={() => setShowSettings(true)}
          >
            설정
          </button>
        </div>
      </header>

      {/* 본문 */}
      <div className="workspace flex flex-1 overflow-hidden px-5 pb-5 gap-5">

        {/* 좌측 패널 */}
        <aside className="work-rail shrink-0 w-[390px] flex flex-col overflow-y-auto">
          <FilePanel
            priorFiles={priorFiles}
            onPriorFiles={handlePriorFilesChange}
            onStart={handlePrepare}
            loading={loading}
            uploadProgress={uploadProgress}
          />
          <ProgressPanel logs={logs} generating={generating} />

          {/* 청구항 입력 */}
          <section className="rail-section flex flex-col">
            <div className="px-5 pt-5 pb-3 flex items-center justify-between">
              <h2 className="section-eyebrow">02 · 청구항</h2>
              <div className="flex items-center gap-2">
                {claims.length > 0 && (
                  <button
                    className="text-xs text-violet-600 hover:text-violet-800 transition font-semibold"
                    onClick={() => setShowClaimAnalysis(true)}
                  >
                    청구항 분석 보기
                  </button>
                )}
                <span className={`text-xs font-medium ${priorReady ? 'text-green-600' : 'text-gray-400'}`}>
                  {priorReady ? '준비 완료' : '인용발명 먼저'}
                </span>
              </div>
            </div>

            {/* 컨텍스트 상태 바 */}
            {priorReady && (
              <div className={`mx-5 mb-1 px-3 py-2 flex items-center justify-between text-xs rounded-xl
                ${contextClaims.length > 0 && useCtx
                  ? 'bg-blue-50 border-blue-100'
                  : 'bg-gray-50 border-gray-100'}`}
              >
                <div className="flex items-center gap-1.5">
                  {contextClaims.length > 0 && useCtx ? (
                    <>
                      <span className="text-blue-500">🔗</span>
                      <span className="text-blue-700 font-medium">
                        컨텍스트 포함:
                      </span>
                      <span className="text-blue-600">
                        청구항 {contextClaims.map(item => item?.claim_number ?? item).join(', ')}
                      </span>
                    </>
                  ) : (
                    <>
                      <span className="text-gray-400">○</span>
                      <span className="text-gray-500">
                        {useCtx ? '컨텍스트 없음 (첫 번째 청구항)' : '컨텍스트 클리어됨'}
                      </span>
                    </>
                  )}
                </div>
                {contextClaims.length > 0 && (
                  <button
                    className="text-[11px] text-red-400 hover:text-red-600 transition"
                    title="이전 분석 컨텍스트를 초기화합니다. 다음 생성부터 이전 청구항 맥락 없이 독립 분석됩니다."
                    onClick={async () => {
                      if (!jobId) return
                      try {
                        await clearContext(jobId)
                        setContextClaims([])
                        setUseCtx(false)
                        addLog('[컨텍스트 클리어] 이전 청구항 분석 맥락이 초기화되었습니다.')
                      } catch (e) {
                        addLog(`[오류] 컨텍스트 클리어 실패: ${e.message}`)
                      }
                    }}
                  >
                    컨텍스트 클리어
                  </button>
                )}
              </div>
            )}

            <div className="px-5 pb-5 flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-500 shrink-0">청구항 번호</label>
                <input
                  type="number"
                  min="1"
                  className="soft-input px-3 py-2 text-sm w-24"
                  value={claimNumber}
                  onChange={e => setClaimNumber(e.target.value)}
                />
              </div>

              <textarea
                className="soft-input p-3 text-sm leading-relaxed resize-none h-40"
                placeholder="청구항 전문을 붙여넣으세요."
                value={claimText}
                onChange={e => setClaimText(e.target.value)}
              />

              {generating ? (
                <div className="flex gap-2">
                  <button
                    className="primary-button flex-1 opacity-60 cursor-not-allowed"
                    disabled
                  >
                    보고서 작성 중…
                  </button>
                  <button
                    className="danger-button px-4"
                    onClick={handleCancelGenerate}
                  >
                    취소
                  </button>
                </div>
              ) : (
                <button
                  className="primary-button w-full disabled:opacity-40 disabled:cursor-not-allowed"
                  disabled={!priorReady || !claimText.trim()}
                  onClick={handleGenerate}
                >
                  구성대비 보고서 생성
                </button>
              )}

              {error && (
                <div className="text-xs text-red-600 bg-red-50 border border-red-200 rounded-lg p-3">
                  {error}
                </div>
              )}
            </div>
          </section>
        </aside>

        {/* 우측: 보고서 */}
        <main className="flex-1 overflow-hidden">
          <div className="report-canvas flex flex-col h-full">
            {/* 청구항 전환 탭 — 복수 청구항 생성 시만 표시 */}
            {tabKeys.length > 1 && (
              <div className="px-7 pt-5 flex items-center gap-2">
                <span className="text-xs text-slate-500 font-medium shrink-0">청구항</span>
                {tabKeys.map(num => (
                  <button
                    key={num}
                    onClick={() => {
                      const r = allReports[num]
                      setReport(sanitizeReportText(r.report_md))
                      setUsedInventions(r.usedInventions)
                      setActiveClaimNumView(num === RELATED_A_TAB_KEY ? RELATED_A_TAB_KEY : Number(num))
                      if (num === RELATED_A_TAB_KEY) setShowMissingPriorArt(false)
                    }}
                    className={[
                      'text-xs px-3 py-1.5 rounded-full transition-colors font-semibold',
                      activeClaimNumView === (num === RELATED_A_TAB_KEY ? RELATED_A_TAB_KEY : Number(num))
                        ? 'bg-slate-950 text-white'
                        : 'text-slate-500 hover:bg-slate-100',
                    ].join(' ')}
                  >
                    {num === RELATED_A_TAB_KEY ? RELATED_A_TAB_LABEL : `청구항 ${num}`}
                  </button>
                ))}
              </div>
            )}
            <div className="report-scroll flex-1 overflow-y-auto relative">
              {generating && !report ? (
                <div className="flex items-center gap-3 text-blue-500 text-sm">
                  <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  인용발명과 청구항을 순차 대비하고 있습니다.
                </div>
              ) : !isRelatedATabActive
                && showMissingPriorArt
                && missingPriorArt
                && Number(missingPriorArt.claim_number) === Number(activeClaimForActions) ? (
                <MissingPriorArtResultPage
                  result={missingPriorArt}
                  onBack={() => setShowMissingPriorArt(false)}
                />
              ) : report ? (
                <article className="report-document">
                  <header className="report-document-header">
                    <div>
                      <span className="report-overline">COMPARISON REPORT</span>
                      <h1>{isRelatedATabActive ? RELATED_A_TAB_LABEL : `청구항 ${activeClaimForActions}`}</h1>
                    </div>
                    <span className="report-edition">ARIA</span>
                  </header>
                  {!isRelatedATabActive && usedInventions.length > 0 && (
                    <section className="report-citations">
                      <p className="report-citations-label">인용발명</p>
                      <div className="flex min-w-0 flex-1 flex-wrap gap-1.5">
                        {citationBasisLabel && (
                          <span className="citation-basis">
                            {citationBasisLabel}
                          </span>
                        )}
                        {usedInventions.map((inv, i) => (
                          <span
                            key={i}
                            className="citation-item"
                          >
                            <span className="font-bold text-slate-800 shrink-0">{inv.name}</span>
                            <span className="text-slate-300">:</span>
                            <span className="min-w-0 truncate font-medium text-slate-600">{inv.filename}</span>
                          </span>
                        ))}
                      </div>
                    </section>
                  )}
                  {!isRelatedATabActive && (
                    <MissingPriorArtSearch
                      jobId={jobId}
                      claimNumber={activeClaimForActions}
                      savedResult={
                        Number(missingPriorArt?.claim_number) === Number(activeClaimForActions)
                          ? missingPriorArt
                          : null
                      }
                      onResult={handleMissingPriorArtResult}
                    />
                  )}
                  {isRelatedATabActive ? (
                    <RelatedAReport markdown={reportForDisplay} />
                  ) : (
                    <div className="report-content phase1-report-content prose max-w-none">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={[rehypeRaw]}
                        components={{ p: ReportParagraph, h3: Phase1H3, li: Phase1ListItem }}
                      >
                        {preprocessPhase1Report(reportForDisplay)}
                      </ReactMarkdown>
                    </div>
                  )}
                </article>
              ) : (
                <div className="empty-report h-full flex flex-col justify-end">
                  <span className="empty-kicker">Patent intelligence, distilled.</span>
                  <h2>복잡한 대비를<br />명료한 판단으로.</h2>
                  <p>인용발명과 청구항을 준비하면<br />분석 결과가 이곳에 정리됩니다.</p>
                </div>
              )}
            </div>
          </div>
        </main>

      </div>

      <footer className="shrink-0 px-7 pb-3 flex justify-end">
        <span className="text-[9px] tracking-[0.12em] text-slate-300">All rights reserved by AIdan.</span>
      </footer>

      {/* 모달/패널 */}
      {showHistory && (
        <HistoryPanel
          history={history}
          onSelect={loadHistoryItem}
          onDelete={deleteHistoryItem}
          onClearLocal={clearHistoryLocalOnly}
          onClearAll={clearHistoryWithServer}
          onClose={() => setShowHistory(false)}
        />
      )}
      {showSettings && <SettingsModal onClose={() => setShowSettings(false)} />}
      <ChatPanel
        open={showChat}
        onOpen={() => setShowChat(true)}
        onClose={() => setShowChat(false)}
        jobId={jobId}
        claimNumber={activeClaimForActions}
        reportMd={report}
      />
      {showClaimAnalysis && jobId && (
        <ClaimAnalysisWindow
          jobId={jobId}
          onClose={() => setShowClaimAnalysis(false)}
        />
      )}
    </div>
  )
}
