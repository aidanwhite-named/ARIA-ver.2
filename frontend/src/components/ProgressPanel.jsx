import { useEffect, useState, useRef } from 'react'

export default function ProgressPanel({ logs, generating }) {
  const bottomRef = useRef()
  const [elapsed, setElapsed] = useState(0)
  const timerRef = useRef(null)
  const startTimeRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  useEffect(() => {
    if (generating) {
      startTimeRef.current = Date.now()
      setElapsed(0)
      timerRef.current = setInterval(() => {
        if (startTimeRef.current) {
          setElapsed(((Date.now() - startTimeRef.current) / 1000).toFixed(1))
        }
      }, 100)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
        timerRef.current = null
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }, [generating])

  if (logs.length === 0) return null

  return (
    <div className="w-full shrink-0 border border-gray-200 rounded bg-gray-900 text-gray-100 text-xs font-mono p-3 overflow-y-auto max-h-64 relative">
      <div className="flex items-center justify-between mb-2 select-none">
        <p className="text-gray-500">진행 로그</p>
        {(generating || elapsed > 0) && (
          <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] font-bold ${
            generating 
              ? 'bg-blue-900/50 text-blue-300 border border-blue-700/50 animate-pulse' 
              : 'bg-gray-800 text-gray-400 border border-gray-700'
          }`}>
            {generating && <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-ping" />}
            <span>경과 시간: {elapsed}초</span>
          </div>
        )}
      </div>
      {logs.map((log, i) => (
        <p key={i} className="leading-5">
          <span className="text-gray-500 select-none">▶ </span>{log}
        </p>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
