import { useState } from 'react'
import { useChatStore } from '../../store/chatStore'
import { Citation } from '../../types'

function DocIcon() {
  return (
    <svg className="w-3.5 h-3.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  )
}

function relevanceLabel(score: number): { label: string; color: string } {
  if (score >= 0.6)  return { label: 'High relevance',   color: 'rgb(var(--accent-green))' }
  if (score >= 0.35) return { label: 'Medium relevance', color: 'rgb(var(--brand))' }
  return                     { label: 'Low relevance',   color: 'rgb(var(--accent-amber))' }
}

function CitationCard({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false)
  const barWidth = Math.max(Math.round(citation.relevance_score * 100), 10)
  const { label, color } = relevanceLabel(citation.relevance_score)
  const hasMore = citation.excerpt.length > 160
  // Break the preview on a word boundary so it never cuts mid-word.
  const preview = hasMore
    ? citation.excerpt.slice(0, 160).replace(/\s+\S*$/, '')
    : citation.excerpt

  return (
    <div
      className="rounded-xl border overflow-hidden transition-all duration-200"
      style={{
        backgroundColor: 'rgb(var(--bg-secondary))',
        borderColor: 'rgb(var(--border))',
      }}
    >
      {/* Header */}
      <div className="flex items-start gap-2.5 p-3">
        <div
          className="w-6 h-6 rounded-md flex items-center justify-center flex-shrink-0 mt-0.5"
          style={{ backgroundColor: 'rgb(var(--brand) / 0.12)', color: 'rgb(var(--brand))' }}
        >
          <DocIcon />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className="text-xs font-bold w-4 h-4 rounded-full flex items-center justify-center flex-shrink-0"
              style={{ backgroundColor: 'rgb(var(--brand))', color: 'white', fontSize: '10px' }}
            >
              {citation.id}
            </span>
            <span className="text-xs font-semibold text-primary truncate flex-1">{citation.source}</span>
            {citation.page && (
              <span
                className="text-xs px-1.5 py-0.5 rounded font-medium flex-shrink-0"
                style={{ backgroundColor: 'rgb(var(--bg-tertiary))', color: 'rgb(var(--text-muted))' }}
              >
                p.{citation.page}
              </span>
            )}
          </div>

          {/* Relevance bar */}
          <div className="flex items-center gap-2 mt-2">
            <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ backgroundColor: 'rgb(var(--border))' }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${barWidth}%`, backgroundColor: color, transition: 'width 0.4s ease' }}
              />
            </div>
            <span className="text-xs font-medium flex-shrink-0" style={{ color }}>
              {label}
            </span>
          </div>
        </div>
      </div>

      {/* Excerpt */}
      <div className="px-3 pb-3">
        <div
          className="rounded-lg p-2.5 text-xs leading-relaxed"
          style={{
            backgroundColor: 'rgb(var(--bg-tertiary))',
            color: 'rgb(var(--text-secondary))',
            borderLeft: '2px solid rgb(var(--brand) / 0.3)',
          }}
        >
          <p>
            {expanded ? citation.excerpt : preview}
            {!expanded && hasMore && <span style={{ color: 'rgb(var(--text-muted))' }}>…</span>}
          </p>
          {hasMore && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mt-1.5 text-xs font-medium transition-colors"
              style={{ color: 'rgb(var(--brand))' }}
            >
              {expanded ? 'Show less' : 'Show more'}
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

export function CitationPanel() {
  const { messages, citationPanelOpen, setCitationPanelOpen } = useChatStore()

  // Show citations from the most recent completed assistant message.
  // Citations are globally unique per-chunk — same chunk always gets the same ID.
  const latestMessage = [...messages]
    .reverse()
    .find((m) => m.role === 'assistant' && m.status === 'complete' && !m.hideRagUI)
  const allCitations: Citation[] = latestMessage?.citations ?? []

  if (!citationPanelOpen) return null

  return (
    <aside
      className="w-72 border-l flex flex-col flex-shrink-0"
      style={{
        backgroundColor: 'rgb(var(--bg-primary))',
        borderColor: 'rgb(var(--border))',
        animation: 'slideInRight 0.2s ease',
      }}
    >
      <div
        className="h-14 flex items-center justify-between px-4 border-b flex-shrink-0"
        style={{ borderColor: 'rgb(var(--border))' }}
      >
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2} style={{ color: 'rgb(var(--text-muted))' }}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
          </svg>
          <span className="text-sm font-semibold text-primary">All Sources</span>
          {allCitations.length > 0 && (
            <span
              className="text-xs px-1.5 py-0.5 rounded-full font-semibold"
              style={{ backgroundColor: 'rgb(var(--brand) / 0.12)', color: 'rgb(var(--brand))' }}
            >
              {allCitations.length}
            </span>
          )}
        </div>
        <button
          onClick={() => setCitationPanelOpen(false)}
          className="btn-ghost p-1.5 rounded-md"
        >
          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {allCitations.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-40 text-center gap-3">
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center"
              style={{ backgroundColor: 'rgb(var(--bg-tertiary))' }}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} style={{ color: 'rgb(var(--text-muted))' }}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>
            <p className="text-xs text-muted">Citations appear here as you ask questions about your documents</p>
          </div>
        ) : (
          allCitations.map((c) => (
            <CitationCard key={`${c.id}-${c.source}`} citation={c} />
          ))
        )}
      </div>
    </aside>
  )
}
