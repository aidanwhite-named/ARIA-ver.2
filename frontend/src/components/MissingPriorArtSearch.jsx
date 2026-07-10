import { useEffect, useState } from 'react'
import { searchMissingPriorArt } from '../api/client'

export default function MissingPriorArtSearch({ jobId, claimNumber, savedResult, onResult }) {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [additionalQuery, setAdditionalQuery] = useState('')

  useEffect(() => {
    setLoading(false)
    setError('')
    setAdditionalQuery('')
  }, [jobId, claimNumber])

  async function handleSearch() {
    setLoading(true)
    setError('')
    try {
      const data = await searchMissingPriorArt(jobId, claimNumber, { additionalQuery })
      onResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <section className="mb-4 rounded-xl border border-indigo-200 bg-indigo-50/60 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="min-w-0 flex-1">
          <p className="text-sm font-bold text-indigo-950">미커버 구성 선행기술 검색</p>
          <p className="mt-0.5 text-xs text-indigo-700">
            정량평가에서 대응이 부족한 구성만 추출해 공개 문헌을 웹에서 검색합니다.
          </p>
        </div>
        <button
          type="button"
          onClick={handleSearch}
          disabled={loading || !jobId || !claimNumber}
          className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-2 text-xs font-semibold text-white transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? (
            <>
              <span className="h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent" />
              검색 중…
            </>
          ) : (
            <>⌕ 부족한 구성 검색</>
          )}
        </button>
        {savedResult && (
          <button
            type="button"
            onClick={() => onResult(savedResult)}
            className="rounded-lg border border-indigo-300 bg-white px-3 py-2 text-xs font-medium text-indigo-700 hover:bg-indigo-50"
          >
            저장된 결과 보기
          </button>
        )}
      </div>

      <div className="mt-2">
        <input
          value={additionalQuery}
          onChange={event => setAdditionalQuery(event.target.value)}
          placeholder="선택사항: 검색 국가, 기준일, 기술분야 등 추가 조건"
          className="w-full rounded-lg border border-indigo-200 bg-white px-3 py-2 text-xs text-slate-700 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
        />
      </div>

      {error && (
        <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-xs text-red-700">
          {error}
        </div>
      )}
    </section>
  )
}
