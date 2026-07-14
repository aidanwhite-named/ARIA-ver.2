export function normalizeMissingPriorArtMarkdown(markdown = '') {
  return markdown
    .replace(/\*\*식별자\*\*\s*:/g, '**문헌번호(이름)**:')
    .replace(
      /(\*\*직접 링크\*\*\s*:\s*)\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g,
      '$1<$3>',
    )
}

function normalizeCandidateDocumentName(title, body) {
  const documentLine = /^(\s*[-*]\s*\*\*문헌번호\(이름\)\*\*\s*:\s*)(.+)$/m
  const match = body.match(documentLine)
  if (!match) return body

  const patentNumber = /\b(?:US|WO|EP|KR|JP|CN)\s*[-/]?\s*\d{4,}[A-Z]\d?\b/i
  if (patentNumber.test(match[2])) return body

  const paperTitle = title
    .replace(/^후보\s*\d+\s*:\s*/i, '')
    .replace(/^[①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮]\s*/, '')
    .trim()
  if (!paperTitle) return body
  return body.replace(documentLine, `$1${paperTitle}`)
}

export function splitMissingPriorArtMarkdown(markdown = '') {
  const normalized = normalizeMissingPriorArtMarkdown(markdown)
  const lines = normalized.split(/\r?\n/)
  const sectionStart = lines.findIndex(line => /^##\s+후보 문헌\s*$/.test(line.trim()))
  if (sectionStart < 0) {
    return { prefix: normalized, candidates: [], suffix: '' }
  }

  let sectionEnd = lines.findIndex(
    (line, index) => index > sectionStart && /^##\s+/.test(line.trim()),
  )
  if (sectionEnd < 0) sectionEnd = lines.length

  const candidates = []
  let category = ''
  let current = null
  const flush = () => {
    if (!current) return
    current.body = normalizeCandidateDocumentName(
      current.title,
      current.lines.join('\n').trim(),
    )
    delete current.lines
    candidates.push(current)
    current = null
  }

  for (const line of lines.slice(sectionStart + 1, sectionEnd)) {
    const heading = line.trim().match(/^(#{3,4})\s+(.+)$/)
    if (heading) {
      const [, level, title] = heading
      const isLegacyCategory = level === '###' && (
        /^(?:\d+\.)?\s*(?:직접 대응|부분 대응|대응 불충분)\s*후보/.test(title)
        || /후보\s*\([^)]*(?:직접|부분|불충분)[^)]*\)/.test(title)
      )
      if (isLegacyCategory) {
        flush()
        category = title.replace(/^\d+\.\s*/, '')
      } else {
        flush()
        current = { title, category, lines: [] }
      }
      continue
    }
    if (current) current.lines.push(line)
  }
  flush()

  return {
    prefix: lines.slice(0, sectionStart).join('\n').trim(),
    candidates,
    suffix: lines.slice(sectionEnd).join('\n').trim(),
  }
}
