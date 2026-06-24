import { useRef } from 'react'

export default function FilePanel({ priorFiles, onPriorFiles, onStart, loading, uploadProgress }) {
  const priorRef = useRef()
  const canStart = priorFiles.length > 0 && priorFiles.length <= 7 && !loading

  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm flex flex-col gap-3 p-4 w-full">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-800">인용발명 PDF</h2>
        <span className="text-xs text-gray-400">{priorFiles.length} / 7</span>
      </div>

      {/* 드롭존 */}
      <div
        className="border-2 border-dashed border-gray-200 rounded-lg bg-gray-50 cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition min-h-[120px] p-3 flex flex-col justify-start"
        onClick={() => {
          if (!loading) priorRef.current.click()
        }}
      >
        {priorFiles.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full gap-2 py-6 text-center">
            <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center text-gray-400 text-lg">+</div>
            <span className="text-xs text-gray-400">클릭하여 PDF 선택</span>
            <span className="text-[10px] text-gray-300">최대 7개</span>
          </div>
        ) : (
          <ul className="space-y-1.5 w-full">
            {priorFiles.map((f, i) => (
              <li key={`${f.name}-${i}`} className="flex items-center gap-2 group">
                <span className="text-xs font-semibold text-blue-500 w-5 shrink-0 text-center">{i + 1}</span>
                <span className="text-xs text-gray-700 truncate flex-1">{f.name}</span>
                <button
                  className="text-gray-300 hover:text-red-400 text-base px-1 opacity-0 group-hover:opacity-100 transition shrink-0"
                  disabled={loading}
                  onClick={e => {
                    e.stopPropagation()
                    if (loading) return
                    onPriorFiles(priorFiles.filter((_, j) => j !== i))
                  }}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <input
        ref={priorRef}
        type="file"
        accept=".pdf"
        multiple
        className="hidden"
        onChange={e => {
          if (loading) {
            e.target.value = ''
            return
          }
          const next = [...priorFiles, ...Array.from(e.target.files)].slice(0, 7)
          onPriorFiles(next)
          e.target.value = ''
        }}
      />

      {priorFiles.length > 7 && (
        <p className="text-xs text-red-500">최대 7개까지 가능합니다.</p>
      )}

      {loading && uploadProgress < 100 && (
        <div>
          <div className="flex justify-between text-xs text-gray-400 mb-1">
            <span>업로드 중</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="w-full bg-gray-100 rounded-full h-1.5">
            <div
              className="bg-blue-500 h-1.5 rounded-full transition-all duration-200"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      <button
        className="w-full py-2.5 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
        disabled={!canStart}
        onClick={onStart}
      >
        {loading
          ? uploadProgress < 100
            ? `업로드 ${uploadProgress}%`
            : '인용발명 읽는 중…'
          : '인용발명 준비'}
      </button>
    </div>
  )
}
