import { useState, useEffect, useRef } from 'react'
import { chatAboutReport } from '../api/client'

// Phase 1 분석 결과에 대한 Q&A 채팅 (질문 전용 — 보고서 수정 없음).
// 우측 슬라이드 드로어. 청구항이 바뀌면 대화가 초기화된다.
export default function ChatPanel({ open, onClose, jobId, claimNumber, reportMd }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [webSearch, setWebSearch] = useState(false)
  const scrollRef = useRef(null)
  const helpExamples = [
    '구성 (C)가 왜 대응 없음인지 설명해줘',
    '인용발명 2보다 인용발명 1을 우선 본 이유를 설명해줘',
  ]
  const searchHint = '미대응 구성 검색은 키워드 탭의 보완문서 웹검색 버튼이 더 빠르고 안정적입니다.'

  // 청구항 전환 시 대화 초기화
  useEffect(() => {
    setMessages([])
    setError('')
  }, [claimNumber])

  // 새 메시지 시 자동 스크롤
  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages, loading])

  async function send() {
    const text = input.trim()
    if (!text || loading) return
    setError('')
    const next = [...messages, { role: 'user', content: text }]
    setMessages(next)
    setInput('')
    setLoading(true)
    try {
      const { answer } = await chatAboutReport(jobId, claimNumber, next, reportMd, { webSearch })
      setMessages([...next, { role: 'assistant', content: answer }])
    } catch (e) {
      setError(e.message)
      // 실패 시 보낸 질문은 유지 (재시도 가능)
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  if (!open) return null

  return (
    <div className="fixed inset-y-0 right-0 z-40 w-[400px] bg-white border-l border-gray-200 shadow-2xl flex flex-col">
      {/* 헤더 */}
      <div className="px-4 py-3 border-b flex items-center justify-between bg-slate-50 shrink-0">
        <div className="flex flex-col">
          <span className="text-sm font-semibold text-slate-800">💬 보고서에 질문</span>
          <span className="text-[11px] text-slate-400">
            청구항 {claimNumber} Phase 1 분석 · 보고서는 수정되지 않습니다
          </span>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-slate-700 text-lg leading-none px-1"
          aria-label="닫기"
        >
          ✕
        </button>
      </div>

      {/* 메시지 영역 */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 py-4 space-y-3">
        <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2 text-[11px] text-slate-500 leading-relaxed">
          보고서 설명과 근거 확인용 채팅입니다. {searchHint}
        </div>
        {messages.length === 0 && !loading && (
          <div className="text-xs text-slate-400 leading-relaxed mt-4">
            이 청구항의 Phase 1 분석에 대해 궁금한 점을 물어보세요.<br />
            예: "구성 (C)는 왜 대응 없음으로 봤나요?", "인용발명 2가 더 적합하지 않나요?"
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}
          >
            <div
              className={[
                'max-w-[85%] rounded-2xl px-3.5 py-2 text-sm whitespace-pre-wrap break-words',
                m.role === 'user'
                  ? 'bg-blue-600 text-white rounded-br-sm'
                  : 'bg-slate-100 text-slate-800 rounded-bl-sm',
              ].join(' ')}
            >
              {m.content}
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-slate-100 text-slate-500 rounded-2xl rounded-bl-sm px-3.5 py-2 text-sm flex items-center gap-2">
              <div className="w-3 h-3 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
              생각 중...
            </div>
          </div>
        )}
        {error && (
          <div className="text-xs text-red-500 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
            오류: {error}
          </div>
        )}
      </div>

      {/* 입력 영역 */}
      <div className="border-t px-3 py-3 shrink-0">
        <label className="mb-2 flex items-center gap-2 text-xs text-slate-600 select-none">
          <input
            type="checkbox"
            checked={webSearch}
            onChange={e => setWebSearch(e.target.checked)}
            disabled={loading}
            className="h-3.5 w-3.5 rounded border-slate-300 text-blue-600 focus:ring-blue-500"
          />
          <span>웹검색 포함</span>
          <span className="text-slate-400">저장된 보완문서 결과는 항상 참조</span>
        </label>
        <div className="flex items-end gap-2">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown}
            rows={2}
            placeholder="질문을 입력하세요 (Enter 전송, Shift+Enter 줄바꿈)"
            className="flex-1 resize-none text-sm border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:border-blue-400"
            disabled={loading}
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="shrink-0 bg-blue-600 text-white text-sm font-medium rounded-lg px-4 py-2 disabled:opacity-40 hover:bg-blue-700 transition-colors"
          >
            전송
          </button>
        </div>
      </div>
    </div>
  )
}
