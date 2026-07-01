import { useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import { addManualClaim, streamPrepare, streamReport, reportBatchDependent, getDependentBatchStatus, uploadFiles, getContextInfo, clearContext, checkJobStatus, detectCategory, deleteJob, deleteAllJobs, cancelGeneration } from './api/client'
import ClaimAnalysisWindow from './components/ClaimAnalysisWindow'

import FilePanel from './components/FilePanel'
import ProgressPanel from './components/ProgressPanel'
import SettingsModal from './components/SettingsModal'
import KeywordPanel from './components/KeywordPanel'
import ChatPanel from './components/ChatPanel'

function AriaEmblem() {
  return (
    <svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
      <defs>
        {/* Soft Dreamy Background */}
        <linearGradient id="dream-bg" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#fdf4ff" />
          <stop offset="100%" stopColor="#eff6ff" />
        </linearGradient>
        
        {/* Ethereal Glows */}
        <radialGradient id="glow-cyan" cx="0.3" cy="0.3" r="0.6">
          <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="glow-magenta" cx="0.7" cy="0.7" r="0.6">
          <stop offset="0%" stopColor="#d946ef" stopOpacity="0.8" />
          <stop offset="100%" stopColor="#d946ef" stopOpacity="0" />
        </radialGradient>
        <radialGradient id="glow-violet" cx="0.5" cy="0.8" r="0.7">
          <stop offset="0%" stopColor="#8b5cf6" stopOpacity="0.7" />
          <stop offset="100%" stopColor="#8b5cf6" stopOpacity="0" />
        </radialGradient>

        <filter id="dream-blur" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="6" />
        </filter>
        
        <filter id="glass-shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="6" floodColor="#8b5cf6" floodOpacity="0.25" />
        </filter>
      </defs>

      <rect width="48" height="48" rx="14" fill="url(#dream-bg)" />
      
      <g filter="url(#dream-blur)">
        <circle cx="16" cy="16" r="14" fill="url(#glow-cyan)" />
        <circle cx="34" cy="32" r="16" fill="url(#glow-magenta)" />
        <circle cx="24" cy="40" r="18" fill="url(#glow-violet)" />
      </g>
      
      <rect x="0.5" y="0.5" width="47" height="47" rx="13.5" fill="rgba(255, 255, 255, 0.4)" stroke="rgba(255,255,255,0.8)" strokeWidth="1" />
      
      <g filter="url(#glass-shadow)">
        <path d="M24 10L12 36H16.5L24 18L31.5 36H36L24 10Z" fill="#ffffff" />
        <path d="M24 23L25.5 27.5L30 29L25.5 30.5L24 35L22.5 30.5L18 29L22.5 27.5L24 23Z" fill="#ffffff" />
      </g>
    </svg>
  )
}


// ── Phase 2 판정 라벨 색상 ────────────────────────────────────────────────────
const JUDGMENT_COLORS = {
  '동일':       'text-green-700 border-green-500 bg-green-50',
  '실질적동일': 'text-blue-700 border-blue-400 bg-blue-50',
  '일부차이':   'text-orange-700 border-orange-400 bg-orange-50',
  '일부유사':   'text-amber-700 border-amber-400 bg-amber-50',
  '차이':       'text-gray-500 border-gray-300 bg-gray-50',
}

// ── Phase 1 유사도 배지 스타일 ───────────────────────────────────────────────
const SIMILARITY_STYLES = {
  '동일':           { badge: 'bg-green-100 text-green-800 border border-green-300' },
  '실질적동일':     { badge: 'bg-orange-100 text-orange-800 border border-orange-300' },
  '일부차이':       { badge: 'bg-amber-100 text-amber-700 border border-amber-300' },
  '일부유사':       { badge: 'bg-green-100 text-green-800 border border-green-300' },
  '차이':           { badge: 'bg-gray-100 text-gray-700 border border-gray-300' },
}

function normalizeReportStatusIcons(text) {
  return String(text || '')
    .replace(/[🔵🟠🟢🟡⚪🔴\uFFFD�]+/g, '')
    .replace(/(^|\r?\n)\s*[�\uFFFD]+\s*(?=최종 보고서)/g, '$1')
    .replace(/[ \t]{2,}/g, ' ')
    .replace(/[ \t]+(\r?\n)/g, '$1')
    .replace(/\uFFFD/g, '')
}

function sanitizeReportText(text) {
  return normalizeReportStatusIcons(text)
}

// ── Phase 1 필드 라벨 스타일 ─────────────────────────────────────────────────
const FIELD_LABEL_STYLES = [
  { re: /^(청구항 구성)\s*:/,              cls: 'border-indigo-300 bg-indigo-50/50 text-indigo-950', labelCls: 'text-indigo-700' },
  { re: /^(인용발명 대응 부분 요약)\s*:/,  cls: 'border-sky-300 bg-sky-50/50 text-sky-950', labelCls: 'text-sky-700' },
  { re: /^(인용발명\s*대응 원문)\s*:/,     cls: 'border-teal-400 bg-teal-50/50 text-teal-950', labelCls: 'text-teal-700' },
  { re: /^(판단 이유)\s*:/,               cls: 'border-violet-300 bg-violet-50/50 text-violet-950', labelCls: 'text-violet-700' },
  { re: /^(판단 근거)\s*:/,               cls: 'border-violet-400 bg-violet-50/50 text-violet-950', labelCls: 'text-violet-700' },
  { re: /^(차이점)\s*:/,                  cls: 'border-rose-300 bg-rose-50/50 text-rose-950', labelCls: 'text-rose-700' },
  { re: /^(유사점 요약)\s*:/,             cls: 'border-green-300 bg-green-50/50 text-green-950', labelCls: 'text-green-700' },
]

function extractText(children) {
  if (typeof children === 'string') return children
  if (Array.isArray(children)) return children.map(c => extractText(c)).join('')
  if (children?.props?.children) return extractText(children.props.children)
  return ''
}

// ── 유사도 단계별 행 색상 (배경 + 왼쪽 테두리) ──────────────────────────────
const SIMILARITY_ROW_COLORS = {
  '동일':           'bg-green-50  border-l-4 border-green-500',
  '실질적동일':     'bg-orange-50 border-l-4 border-orange-500',
  '일부차이':       'bg-amber-50  border-l-4 border-amber-400',
  '일부유사':       'bg-green-50  border-l-4 border-green-500',
  '차이':           'bg-red-50    border-l-4 border-red-400',
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
  if (/종합 분석 요약/.test(text)) {
    return (
      <h3 className="flex items-center gap-2 mt-8 mb-3 px-3 py-2 bg-slate-100 border border-slate-200 rounded-lg text-sm font-bold text-slate-700">
        {children}
      </h3>
    )
  }

  const m = text.match(/^\[(구성요소|추가\s*구성)(?:\s*\(\s*([A-J](?:-\d+)?)\s*\))?\]$/)
  if (m) {
    return (
      <h3 className="phase1-component-heading mt-7 mb-2 text-sm font-semibold text-gray-800">
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
    /^(?:유사도\s*:\s*)?\(([A-J](?:-\d+)?)\)\s*(동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음)?\s*(\d+%)?/
  )
  const oldSimMatch = text.match(
    /^유사도\s*:\s*(동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음)?\s*(\d+%)?/
  )
  if (newSimMatch || oldSimMatch) {
    const elementLabel = newSimMatch ? newSimMatch[1] : ''
    const labelText = newSimMatch ? (newSimMatch[2] || '') : (oldSimMatch?.[1] || '')
    const normalizedLabel = labelText.replace(/\s+/g, '') === '대응없음' ? '차이' : labelText.replace(/\s+/g, '')
    const pct = newSimMatch ? (newSimMatch[3] || '') : (oldSimMatch?.[2] || '')
    const style = SIMILARITY_STYLES[normalizedLabel] || SIMILARITY_STYLES['차이']
    const rowColor = SIMILARITY_ROW_COLORS[normalizedLabel] || SIMILARITY_ROW_COLORS['차이']
    return (
      <li className={`flex items-center gap-2 py-2 px-3 rounded-r my-1.5 list-none -ml-5 ${rowColor}`}>
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
  const match = text.match(
    /^(\([A-J](?:-\d+)?\)(?:\s*(?:및|,)\s*\([A-J](?:-\d+)?\))*)\s+(동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음)(?:\s+(\d+%))?\s*$/
  )
  const fallbackMatch = !match && text.match(
    /^(동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음)(?:\s+(\d+%))?\s*$/
  )
  if (!match && !fallbackMatch) return null
  const label = match ? match[1] : ''
  const rawJudgment = match ? match[2] : fallbackMatch[1]
  const pct = match ? (match[3] || '') : (fallbackMatch[2] || '')
  const judgment = rawJudgment.replace(/\s+/g, '')
  return { label, judgment, pct }
}

function getJudgmentTone(judgment) {
  const normalized = String(judgment || '').replace(/\s+/g, '')
  if (normalized === '동일') return 'border-green-300 bg-green-50/60'
  if (normalized === '실질적동일') return 'border-blue-300 bg-blue-50/60'
  if (normalized === '일부차이') return 'border-orange-300 bg-orange-50/60'
  if (normalized === '일부유사') return 'border-amber-300 bg-amber-50/60'
  if (normalized === '차이' || normalized === '대응없음') return 'border-gray-300 bg-gray-50'
  return 'border-slate-200 bg-white'
}

function splitPhase2ComponentBlocks(body) {
  const lines = String(body || '').split('\n')
  const blocks = []
  let current = null
  let lead = []

  const flushCurrent = () => {
    if (current) {
      current.body = current.lines.join('\n').trim()
      blocks.push(current)
      current = null
    }
  }

  let fallbackLabelCode = 'A'.charCodeAt(0)
  for (const line of lines) {
    const judgment = renderJudgmentInline(line.trim())
    if (judgment) {
      flushCurrent()
      let blockLine = line
      if (!judgment.label) {
        judgment.label = `(${String.fromCharCode(fallbackLabelCode)})`
        fallbackLabelCode += 1
        blockLine = `${judgment.label} ${line.trim()}`
      }
      current = { judgment, lines: [blockLine] }
      continue
    }
    if (current) current.lines.push(line)
    else lead.push(line)
  }
  flushCurrent()

  return {
    lead: lead.join('\n').trim(),
    blocks: blocks.filter(block => block.body),
  }
}

function ReportParagraph({ children }) {
  const text = extractText(children)
  const trimmed = text.trim()
  if (/^\[(구성요소|추가\s*구성)(?:\s*\(\s*[A-J](?:-\d+)?\s*\))?\]$/.test(trimmed)) {
    return <p className="phase1-component-heading">{children}</p>
  }
  if (/^\[(인용발명 단독\(신규성\)|인용발명 1 \+ 주지관용\(진보성\)|인용발명 1과 2의 결합\(진보성\)|인용발명 1과 2의 결합 및 주지관용\(진보성\))\]$/.test(trimmed)) {
    return <p className="mt-2 mb-4 text-lg font-bold tracking-tight text-slate-900">{children}</p>
  }
  if (/^\[(구성대비|구성요소|종합 판단|유사점|차이점|결론)\]$/.test(trimmed)) {
    const isMajor = /^\[(구성대비|종합 판단)\]$/.test(trimmed)
    const isDiff = trimmed === '[차이점]'
    const isSimilar = trimmed === '[유사점]'
    const isConclusion = trimmed === '[결론]'
    return (
      <p className={
        isMajor
          ? 'mt-6 mb-2 px-4 py-3 rounded-xl bg-slate-100 border border-slate-200 text-base font-bold text-slate-800'
          : isDiff
            ? 'mt-5 mb-2 px-3 py-2 rounded-lg bg-rose-50 border border-rose-200 text-sm font-bold text-rose-800'
            : isSimilar
              ? 'mt-5 mb-2 px-3 py-2 rounded-lg bg-emerald-50 border border-emerald-200 text-sm font-bold text-emerald-800'
              : isConclusion
                ? 'mt-5 mb-2 px-3 py-2 rounded-lg bg-amber-50 border border-amber-200 text-sm font-bold text-amber-800'
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
    const color = JUDGMENT_COLORS[judgment] || 'text-gray-500 border-gray-300 bg-gray-50'
    return (
      <p className={`flex items-center gap-2 font-semibold text-sm border-l-4 pl-3 py-0.5 rounded-r mt-4 mb-1 overflow-x-auto ${color}`}>
        {label && <span className="shrink-0">{label}</span>}
        <span className="shrink-0">{judgment}</span>
        {pct && <span className="shrink-0 font-mono">{pct}</span>}
      </p>
    )
  }
  return <p className="my-1 text-sm leading-relaxed">{children}</p>
}

function Phase2Markdown({ body }) {
  return (
    <div className="report-content prose max-w-none">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw]}
        components={{ p: ReportParagraph, h3: Phase1H3, li: Phase1ListItem }}
      >
        {preprocessReport(body)}
      </ReactMarkdown>
    </div>
  )
}

function Phase2CompositionBody({ body }) {
  const { lead, blocks } = splitPhase2ComponentBlocks(body)
  if (blocks.length === 0) return <Phase2Markdown body={body} />

  return (
    <div>
      {lead && <Phase2Markdown body={lead} />}
      <div className="space-y-3">
        {blocks.map((block, index) => (
          <section
            key={`${block.judgment.label}-${index}`}
            className={`rounded-xl border px-4 py-3 shadow-sm ${getJudgmentTone(block.judgment.judgment)}`}
          >
            <Phase2Markdown body={block.body} />
          </section>
        ))}
      </div>
    </div>
  )
}

function Phase2SectionCard({ title, body }) {
  const isComposition = title === '[구성대비]'
  return (
    <section className="mb-5 rounded-2xl border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-200 px-5 py-4">
        <h3 className="text-base font-bold tracking-tight text-slate-800">{title}</h3>
      </div>
      <div className="px-5 py-4">
        {isComposition ? <Phase2CompositionBody body={body} /> : <Phase2Markdown body={body} />}
      </div>
    </section>
  )
}

function preprocessReport(md) {
  md = sanitizeReportText(md)

  function fieldClass(label) {
    if (label === '청구항 구성') return 'phase1-field phase1-field-claim'
    if (label === '인용발명 대응 원문') return 'phase1-field phase1-field-quote'
    if (label === '인용발명 대응 부분 요약') return 'phase1-field phase1-field-summary'
    if (label === '판단 이유' || label === '판단 근거') return 'phase1-field phase1-field-reason'
    if (label === '차이점') return 'phase1-field phase1-field-diff'
    if (label === '유사점 요약') return 'phase1-field phase1-field-similar'
    return 'phase1-field'
  }

  function normalizeFieldLabel(rawLabel) {
    return rawLabel.replace(/\*\*/g, '').replace(/\s+/g, ' ').trim()
  }

  function normalizePhase1Fields(text) {
    const fieldRe = /^-\s*(?:\*\*)?(청구항 구성|인용발명\s*대응 원문|인용발명 대응 부분 요약|판단 이유|판단 근거|차이점|유사점 요약)(?:\*\*)?\s*:\s*(.*)$/
    const sectionHeaderRe = /^#{1,6}\s*\[(구성요소|추가\s*구성)(?:\s*\([A-J](?:-\d+)?\))?\]\s*$/
    const lines = text.split('\n')
    const result = []
    let i = 0

    while (i < lines.length) {
      const line = lines[i]
      const match = line.match(fieldRe)
      if (!match) {
        result.push(line)
        i += 1
        continue
      }

      const label = normalizeFieldLabel(match[1])
      const bodyLines = []
      if (match[2].trim()) bodyLines.push(match[2].trim())
      i += 1

      while (i < lines.length) {
        const current = lines[i]
        const trimmed = current.trim()
        if (!trimmed) {
          i += 1
          break
        }
        if (
          fieldRe.test(current) ||
          /^#{1,6}\s/.test(trimmed) ||
          sectionHeaderRe.test(trimmed) ||
          /^\([A-J](?:-\d+)?\)\s/.test(trimmed) ||
          /^-\s*(유사점 요약|차이점|결론)\s*:/.test(trimmed)
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
      if (!/^\([A-J](?:-\d+)?\)\s/.test(line.trim())) {
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
          /^\([A-J](?:-\d+)?\)\s/.test(trimmed) ||
          /^\[결론\]$/.test(trimmed) ||
          /^-?\s*(유사점 요약|차이점|결론)\s*:/.test(trimmed)
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

  const judgmentPrefix = String.raw`\([A-J](?:-\d+)?\)(?:\s*(?:및|,)\s*\([A-J](?:-\d+)?\))*\s+(?:동일|실질적동일|실질적\s+동일|일부차이|일부\s+차이|일부유사|일부\s+유사|차이|대응\s+없음)(?:\s+\d+%)?`
  // CLI 에이전트가 새어 보낸 도구 호출 줄(update_topic(...) 등) 제거 — 캐시·히스토리 구보고서까지 정리
  md = md.replace(/^[ \t]*[a-z][a-z0-9_]*\([a-z_]+\s*=\s*['"].*\)[ \t]*-*[ \t]*$/gm, '')
  md = md.replace(
    /([^\n])\s*(#{3,6}\s*\[(?:구성요소|추가\s*구성)(?:\s*\([A-J](?:-\d+)?\))?\])/g,
    '$1\n\n$2'
  )
  md = md.replace(/^(#{3,6})(?=\[(?:구성요소|추가\s*구성))/gm, '$1 ')
  md = md.replace(
    /^#{1,6}\s*(\[(?:구성요소|추가\s*구성)(?:\s*\([A-J](?:-\d+)?\))?\])\s*$/gm,
    '$1'
  )
  md = keepFirstComponentHeader(md)
  md = normalizePhase1Fields(md)
  md = md.replace(
    /([^\n])\s+(-\s*(유사점 요약|차이점|결론)\s*:)/g,
    '$1\n\n$2'
  )
  md = md.replace(
    /^-\s*(유사점 요약|차이점|결론)\s*:\s*(.+)$/gm,
    (_, label, body) => {
      const heading =
        label === '유사점 요약'
          ? '[유사점]'
          : label === '차이점'
            ? '[차이점]'
            : '[결론]'
      return `${heading}\n\n${body.trim()}`
    }
  )
  md = md.replace(
    /([^\n])\n(-\s*(유사점 요약|차이점|결론)\s*:)/g,
    '$1\n\n$2'
  )
  md = md.replace(
    /([^\n])\s+(대응 이유\s*:)/g,
    '$1\n\n$2'
  )
  md = md.replace(
    /([^\n])\n(\([A-J](?:-\d+)?\))/g,
    '$1\n\n$2'
  )
  md = md.replace(
    /^(\([A-J](?:-\d+)?\)(?:\s*(?:및|,)\s*\([A-J](?:-\d+)?\))*\s+(?:동일|실질적동일|실질적 동일|일부차이|일부 차이|일부유사|일부 유사|차이|대응 없음)(?:\s+\d+%)?)\s+(\S.*)$/gm,
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
    const isJudgment = /^\([A-J](?:-\d+)?\)\s/.test(line)
    if (isJudgment && result.length > 0 && result[result.length - 1] !== '') result.push('')
    result.push(line)
    if (isJudgment && i < lines.length - 1 && lines[i + 1] !== '') result.push('')
  }
  return sanitizeReportText(result.join('\n'))
}

function preprocessPhase1Report(md) {
  const hiddenHeadingRe = /^\s*(?:#{1,6}\s*)?\[(?:구성요소|추가\s*구성)(?:\s*\([A-J](?:-\d+)?\))?\]\s*$/gm
  return preprocessReport(String(md || '').replace(hiddenHeadingRe, ''))
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

// Phase 1 / Phase 2 분리
function splitPhases(md) {
  if (!md) return { phase1: '', phase2: '' }
  // "# [Phase 2]" 또는 "# [Phase 2]" 로 시작하는 줄을 기준으로 분리
  const idx = md.search(/^#\s*\[Phase\s*2\]/m)
  if (idx === -1) return { phase1: md, phase2: '' }
  return {
    phase1: md.slice(0, idx).trimEnd(),
    phase2: md.slice(idx).trimStart(),
  }
}

function splitPhase2Sections(md) {
  if (!md) return null
  const trimmed = md.trim()
  const compositionMarker = '[구성대비]'
  const judgmentMarker = '[종합 판단]'
  const rejectedMarker = '## 관련도 A 인용발명'
  const rejectedIdx = trimmed.indexOf(rejectedMarker)
  const phase2Only = rejectedIdx === -1 ? trimmed : trimmed.slice(0, rejectedIdx).trimEnd()
  const compositionIdx = phase2Only.indexOf(compositionMarker)
  const judgmentIdx = phase2Only.indexOf(judgmentMarker)
  if (compositionIdx === -1 || judgmentIdx === -1 || judgmentIdx <= compositionIdx) return null

  const header = phase2Only.slice(0, compositionIdx).trim()
  const headerLines = header.split('\n').map(line => line.trim()).filter(Boolean)
  const phaseTitleLine = headerLines.find(line => /^#\s*\[Phase\s*2\]/i.test(line)) || ''
  const inventionHeaderLine = headerLines.find(line => /^\[.+\]$/.test(line) && !/^\[Phase\s*2\]$/i.test(line)) || ''
  const phaseTitle = phaseTitleLine.replace(/^#\s*\[Phase\s*2\]\s*/i, '').trim()
  const inventionHeader = inventionHeaderLine.replace(/^\[|\]$/g, '').trim()

  return {
    header,
    phaseTitle,
    inventionHeader,
    compositionBody: phase2Only.slice(compositionIdx + compositionMarker.length, judgmentIdx).trim(),
    judgmentBody: phase2Only.slice(judgmentIdx + judgmentMarker.length).trim(),
    rejectedBody: rejectedIdx === -1 ? '' : trimmed.slice(rejectedIdx + rejectedMarker.length).trim(),
  }
}

// ── 히스토리 로드/저장 헬퍼 ──────────────────────────────────────────────────
function loadHistory() {
  try {
    return JSON.parse(localStorage.getItem('aria_history') || '[]')
      .map(item => ({ ...item, report: sanitizeReportText(item.report || '') }))
  }
  catch { return [] }
}
function saveHistory(list) {
  localStorage.setItem(
    'aria_history',
    JSON.stringify(list.map(item => ({ ...item, report: sanitizeReportText(item.report || '') })))
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
  const [reportTab, setReportTab] = useState('phase1')  // 'phase1' | 'phase2' | 'keywords'
  const [usedInventions, setUsedInventions] = useState([])  // 실제 사용된 인용발명 목록
  const [allReports, setAllReports] = useState({})  // { claimNum: { report_md, usedInventions } }
  const [activeClaimNumView, setActiveClaimNumView] = useState(null)
  const [error, setError] = useState('')
  const [showSettings, setShowSettings] = useState(false)
  const [showClaimAnalysis, setShowClaimAnalysis] = useState(false)

  // 키워드 상태
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
    setReportTab('phase1')
    setUsedInventions([])
    setAllReports({})
    setActiveClaimNumView(null)
    setContextClaims([])
    setUseCtx(true)
    setError('')
  }

  // 히스토리 저장
  function addHistoryItem(item) {
    setHistory(prev => {
      const updated = [item, ...prev].slice(0, 50)
      saveHistory(updated)
      return updated
    })
  }
  // 히스토리 개별 항목 삭제 (목록에서만 제거)
  function deleteHistoryItem(id) {
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
    setReportTab('phase1')
    setUsedInventions([])
    setAllReports({})
    setActiveClaimNumView(null)
    setContextClaims([])
    setUseCtx(true)
    setError('')
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
      setReportTab('phase1')
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
    const cleanReport = sanitizeReportText(item.report)
    setReport(cleanReport)
    setReportTab('phase1')
    setUsedInventions(item.usedInventions || [])
    setAllReports({ [item.claimNumber]: { report_md: cleanReport, usedInventions: item.usedInventions || [] } })
    setActiveClaimNumView(item.claimNumber)
    setClaimNumber(item.claimNumber)
    setClaimText(item.claimTextPreview)

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
    setReportTab('phase1')
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
              const cleanReport = sanitizeReportText(data.report_md)
              setReport(cleanReport)
              setUsedInventions(data.used_inventions || [])
              if (data.timings) {
                const order = ['comparison', 'citation chain', 'quote verification', 'phase1', 'phase2', 'finalize', 'total']
                const summary = order
                  .filter(key => data.timings[key] != null)
                  .map(key => `${key} ${Number(data.timings[key]).toFixed(1)}s`)
                  .join(' | ')
                if (summary) addLog(`[timing] ${summary}`)
              }
              setAllReports(prev => ({
                ...prev,
                [claim.claim_number]: {
                  report_md: cleanReport,
                  usedInventions: data.used_inventions || []
                }
              }))
              addHistoryItem({
                id: Date.now() + claim.claim_number,
                jobId: jobId,
                claimNumber: claim.claim_number,
                claimTextPreview: text.slice(0, 100),
                report: cleanReport,
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
            const cleanReport = sanitizeReportText(r.report_md)
            setReport(cleanReport)
            setReportTab('phase1')
            setUsedInventions(r.used_inventions || [])
            setActiveClaimNumView(claim.claim_number)
            setAllReports(prev => ({
              ...prev,
              [claim.claim_number]: {
                report_md: cleanReport,
                usedInventions: r.used_inventions || [],
              },
            }))
            addHistoryItem({
              id: Date.now() + claim.claim_number,
              jobId: jobId,
              claimNumber: claim.claim_number,
              claimTextPreview: text.slice(0, 100),
              report: cleanReport,
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

  return (
    <div className="min-h-screen flex flex-col bg-gray-100">

      {/* 헤더 */}
      <header className="flex items-center justify-between px-6 py-3 bg-white border-b shadow-sm shrink-0">
        <div className="flex items-center gap-3">
          <AriaEmblem />
          <div className="flex flex-col justify-center ml-1">
            <div className="flex items-baseline gap-1.5">
              <span className="font-extrabold text-[1.4rem] text-slate-800 tracking-tight leading-none">ARIA</span>
              <span className="font-semibold text-[0.85rem] text-slate-500 tracking-wide">ver.2</span>
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
              className="text-sm text-red-500 hover:text-red-700 border border-red-200 rounded-lg px-3 py-1.5 hover:bg-red-50 transition disabled:opacity-40 disabled:cursor-not-allowed"
              onClick={handleDeleteCurrentJob}
              disabled={generating || loading}
              title="현재 작업의 uploads, reports, cases 저장 데이터를 서버에서 삭제합니다."
            >
              현재 작업 삭제
            </button>
          )}
          <button
            className="text-sm text-gray-500 hover:text-gray-800 border rounded-lg px-3 py-1.5 hover:bg-gray-50 transition flex items-center gap-1.5"
            onClick={() => setShowHistory(true)}
          >
            <span>🕘</span>
            <span>히스토리</span>
            {history.length > 0 && (
              <span className="bg-blue-100 text-blue-600 text-xs font-semibold rounded-full px-1.5 py-0.5 leading-none">
                {history.length}
              </span>
            )}
          </button>
          <button
            className="text-sm text-gray-500 hover:text-gray-800 border rounded-lg px-3 py-1.5 hover:bg-gray-50 transition"
            onClick={() => setShowSettings(true)}
          >
            설정
          </button>
        </div>
      </header>

      {/* 본문 */}
      <div className="flex flex-1 overflow-hidden p-4 gap-4">

        {/* 좌측 패널 */}
        <div className="shrink-0 w-[500px] flex flex-col gap-4 overflow-y-auto">
          <FilePanel
            priorFiles={priorFiles}
            onPriorFiles={handlePriorFilesChange}
            onStart={handlePrepare}
            loading={loading}
            uploadProgress={uploadProgress}
          />
          <ProgressPanel logs={logs} generating={generating} />

          {/* 청구항 입력 */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col">
            <div className="px-4 py-3 border-b flex items-center justify-between">
              <h2 className="text-sm font-semibold text-gray-800">청구항 입력</h2>
              <div className="flex items-center gap-2">
                {claims.length > 0 && (
                  <button
                    className="text-xs bg-indigo-50 text-indigo-600 border border-indigo-200 rounded-lg px-3 py-1 hover:bg-indigo-100 transition font-medium"
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
              <div className={`px-4 py-2 flex items-center justify-between text-xs border-b
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
                    className="text-xs text-red-400 hover:text-red-600 border border-red-200 hover:border-red-400 rounded px-2 py-0.5 hover:bg-red-50 transition"
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

            <div className="p-4 flex flex-col gap-3">
              <div className="flex items-center gap-3">
                <label className="text-xs text-gray-500 shrink-0">청구항 번호</label>
                <input
                  type="number"
                  min="1"
                  className="border rounded-lg px-3 py-2 text-sm w-24 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  value={claimNumber}
                  onChange={e => setClaimNumber(e.target.value)}
                />
              </div>

              <textarea
                className="border rounded-lg p-3 text-sm leading-relaxed resize-none h-48 focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="청구항 전문을 붙여넣으세요."
                value={claimText}
                onChange={e => setClaimText(e.target.value)}
              />

              {generating ? (
                <div className="flex gap-2">
                  <button
                    className="flex-1 py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg opacity-60 cursor-not-allowed"
                    disabled
                  >
                    보고서 작성 중…
                  </button>
                  <button
                    className="px-4 py-2.5 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 transition"
                    onClick={handleCancelGenerate}
                  >
                    취소
                  </button>
                </div>
              ) : (
                <button
                  className="w-full py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
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
          </div>
        </div>

        {/* 우측: 보고서 */}
        <main className="flex-1 overflow-hidden">
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col h-full">
            {/* 청구항 전환 탭 — 복수 청구항 생성 시만 표시 */}
            {Object.keys(allReports).length > 1 && (
              <div className="px-5 py-2 flex items-center gap-2 border-b bg-slate-50">
                <span className="text-xs text-slate-500 font-medium shrink-0">청구항</span>
                {Object.keys(allReports).sort((a, b) => Number(a) - Number(b)).map(num => (
                  <button
                    key={num}
                    onClick={() => {
                      const r = allReports[num]
                      setReport(sanitizeReportText(r.report_md))
                      setUsedInventions(r.usedInventions)
                      setReportTab('phase1')
                      setActiveClaimNumView(Number(num))
                    }}
                    className={[
                      'text-xs px-3 py-1 rounded-full border transition-colors font-medium',
                      activeClaimNumView === Number(num)
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'text-slate-600 border-slate-300 hover:bg-slate-100',
                    ].join(' ')}
                  >
                    청구항 {num}
                  </button>
                ))}
              </div>
            )}
            {/* 헤더 + 탭 */}
            <div className="px-5 py-0 border-b flex items-center justify-between">
              <div className="flex items-center gap-0">
                {[
                  { key: 'phase1', label: '📝 Phase 1 — 구성요소 대비' },
                  { key: 'phase2', label: '📑 Phase 2 — 최종 보고서' },
                  { key: 'keywords', label: '🔍 키워드·보완검색' },
                ].map(({ key, label }) => {
                  const { phase2 } = splitPhases(report)
                  // Phase 2 탭: 생성 중이거나 이미 있으면 활성화
                  const phase2Loading = generating && report && !phase2
                  const disabled =
                    (key === 'phase2' && !phase2 && !phase2Loading) ||
                    (key === 'keywords' && !jobId)
                  return (
                    <button
                      key={key}
                      onClick={() => !disabled && setReportTab(key)}
                      className={[
                        'px-4 py-3 text-xs font-medium border-b-2 transition-colors',
                        reportTab === key
                          ? key === 'keywords'
                            ? 'border-violet-500 text-violet-700'
                            : 'border-blue-600 text-blue-700'
                          : disabled
                            ? 'border-transparent text-gray-300 cursor-default'
                            : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300',
                      ].join(' ')}
                    >
                      {label}
                      {key === 'phase2' && phase2Loading && (
                        <span className="ml-1 inline-block w-3 h-3 border border-blue-400 border-t-transparent rounded-full animate-spin align-middle" />
                      )}
                    </button>
                  )
                })}
              </div>
              {report && reportTab !== 'keywords' && (
                <button
                  onClick={() => setShowChat(v => !v)}
                  className={[
                    'text-xs font-medium rounded-lg px-3 py-1.5 border transition-colors',
                    showChat
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'text-blue-700 border-blue-200 hover:bg-blue-50',
                  ].join(' ')}
                >
                  💬 보고서에 질문
                </button>
              )}
            </div>
            {reportTab === 'keywords' ? (
              <div className="flex-1 overflow-hidden relative">
                <KeywordPanel
                  jobId={jobId}
                  claimNumber={activeClaimNumView || claimNumber}
                  isVisible={reportTab === 'keywords'}
                />
              </div>
            ) : (
            <div className="flex-1 overflow-y-auto px-8 py-6 relative">
              {generating && !report ? (
                <div className="flex items-center gap-3 text-blue-500 text-sm">
                  <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                  인용발명과 청구항을 순차 대비하고 있습니다.
                </div>
              ) : report ? (() => {
                const { phase1, phase2 } = splitPhases(report)
                // Phase 2 생성 중일 때 탭 전환하면 로딩 표시
                const isPhase2 = reportTab === 'phase2'
                const content = isPhase2 ? phase2 : phase1
                const phase2Sections = isPhase2 ? splitPhase2Sections(content) : null
                // Phase 2 탭인데 아직 내용 없으면 생성 중 메시지
                if (isPhase2 && !phase2) {
                  return (
                    <div className="flex items-center gap-3 text-blue-500 text-sm">
                      <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin" />
                      Phase 2 최종 보고서 작성 중... Phase 1 탭에서 먼저 확인하실 수 있습니다.
                    </div>
                  )
                }
                return (
                  <>
                    {isPhase2 && phase2Sections ? (
                      <>
                        {usedInventions.length > 0 && (
                          <section className="mb-5 rounded-2xl border border-slate-200 bg-slate-50 shadow-sm">
                            <div className="border-b border-slate-200 px-5 py-4">
                              {phase2Sections.phaseTitle && (
                                <p className="text-lg font-bold tracking-tight text-slate-900">{phase2Sections.phaseTitle}</p>
                              )}
                              {phase2Sections.inventionHeader && (
                                <p className="mt-1 text-sm font-semibold text-slate-600">{phase2Sections.inventionHeader}</p>
                              )}
                            </div>
                            <div className="flex flex-wrap gap-2 px-5 py-4">
                              {usedInventions.map((inv, i) => (
                                <span
                                  key={i}
                                  className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm"
                                >
                                  <span className="text-base font-bold text-slate-800">{inv.name}</span>
                                  <span className="text-slate-300">:</span>
                                  <span className="font-medium text-slate-600">{inv.filename}</span>
                                </span>
                              ))}
                            </div>
                          </section>
                        )}
                        <Phase2SectionCard title="[구성대비]" body={phase2Sections.compositionBody} />
                        <Phase2SectionCard title="[종합 판단]" body={phase2Sections.judgmentBody} />
                        {phase2Sections.rejectedBody && (
                          <Phase2SectionCard title="관련도 A 인용발명" body={phase2Sections.rejectedBody} />
                        )}
                      </>
                    ) : (
                      <>
                        {/* 구성대비에 사용된 인용발명 표시 */}
                        {usedInventions.length > 0 && (
                          <section className="mb-5 rounded-2xl border border-slate-200 bg-slate-50 shadow-sm">
                            <div className="border-b border-slate-200 px-5 py-4">
                              <p className="text-lg font-bold tracking-tight text-slate-900">구성요소 대비</p>
                            </div>
                            <div className="flex flex-wrap gap-2 px-5 py-4">
                              {usedInventions.map((inv, i) => (
                                <span
                                  key={i}
                                  className="inline-flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm shadow-sm"
                                >
                                  <span className="text-base font-bold text-slate-800">{inv.name}</span>
                                  <span className="text-slate-300">:</span>
                                  <span className="font-medium text-slate-600">{inv.filename}</span>
                                </span>
                              ))}
                            </div>
                          </section>
                        )}
                        <div className="report-content phase1-report-content prose max-w-none">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            rehypePlugins={[rehypeRaw]}
                            components={{ p: ReportParagraph, h3: Phase1H3, li: Phase1ListItem }}
                          >
                            {preprocessPhase1Report(content)}
                          </ReactMarkdown>
                        </div>
                      </>
                    )}
                  </>
                )
              })() : (
                <div className="h-full flex items-center justify-center text-gray-400 text-sm">
                  청구항을 입력하고 보고서를 생성하세요.
                </div>
              )}
            </div>
            )}
          </div>
        </main>

      </div>

      <footer className="shrink-0 border-t bg-white px-6 py-2 flex justify-end">
        <span className="text-xs text-gray-400">All rights reserved AIdan</span>
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
        onClose={() => setShowChat(false)}
        jobId={jobId}
        claimNumber={activeClaimNumView || claimNumber}
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
