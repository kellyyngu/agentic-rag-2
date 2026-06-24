import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Message, Citation } from '../../types'
import { AgentTrace } from './AgentTrace'

interface Props {
  message: Message
  onFollowUp: (q: string) => void
}

function relevanceLabel(score: number): { label: string; color: string } {
  if (score >= 0.6)  return { label: 'High',   color: 'rgb(var(--accent-green))' }
  if (score >= 0.35) return { label: 'Medium', color: 'rgb(var(--brand))' }
  return                     { label: 'Low',   color: 'rgb(var(--accent-amber))' }
}

function SourceCard({ citation }: { citation: Citation }) {
  const [expanded, setExpanded] = useState(false)
  const barWidth = Math.max(Math.round(citation.relevance_score * 100), 10)
  const { label, color } = relevanceLabel(citation.relevance_score)
  const preview = citation.excerpt.slice(0, 180)
  const hasMore = citation.excerpt.length > 180

  return (
    <div
      className="rounded-xl border overflow-hidden transition-all duration-200"
      style={{ backgroundColor: 'rgb(var(--bg-secondary))', borderColor: 'rgb(var(--border))' }}
    >
      {/* Card header */}
      <div className="flex items-center gap-2.5 px-3 py-2.5">
        <span
          className="w-5 h-5 rounded-full flex items-center justify-center flex-shrink-0 font-bold"
          style={{ backgroundColor: 'rgb(var(--brand))', color: 'white', fontSize: '10px' }}
        >
          {citation.id}
        </span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-semibold text-primary truncate">{citation.source}</span>
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
          <div className="flex items-center gap-1.5 mt-1">
            <div className="flex-1 h-0.5 rounded-full overflow-hidden" style={{ backgroundColor: 'rgb(var(--border))' }}>
              <div
                className="h-full rounded-full"
                style={{ width: `${barWidth}%`, backgroundColor: color }}
              />
            </div>
            <span className="text-xs flex-shrink-0" style={{ color }}>{label}</span>
          </div>
        </div>
      </div>

      {/* Excerpt */}
      <div className="px-3 pb-3">
        <div
          className="rounded-lg px-3 py-2 text-xs leading-relaxed"
          style={{
            backgroundColor: 'rgb(var(--bg-tertiary))',
            color: 'rgb(var(--text-secondary))',
            borderLeft: '2px solid rgb(var(--brand) / 0.35)',
          }}
        >
          <p>
            {expanded ? citation.excerpt : preview}
            {!expanded && hasMore && <span style={{ color: 'rgb(var(--text-muted))' }}>…</span>}
          </p>
          {hasMore && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mt-1.5 text-xs font-semibold transition-colors"
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

function InlineSources({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false)
  if (!citations.length) return null

  return (
    <div className="mt-4">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all duration-150"
        style={{
          backgroundColor: open ? 'rgb(var(--brand) / 0.1)' : 'rgb(var(--bg-tertiary))',
          color: open ? 'rgb(var(--brand))' : 'rgb(var(--text-secondary))',
          border: '1px solid',
          borderColor: open ? 'rgb(var(--brand) / 0.3)' : 'rgb(var(--border))',
        }}
      >
        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
        </svg>
        Sources ({citations.length})
        <svg
          className="w-3 h-3 transition-transform duration-200"
          style={{ transform: open ? 'rotate(180deg)' : 'rotate(0deg)' }}
          fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {open && (
        <div className="mt-2 space-y-2" style={{ animation: 'fadeIn 0.15s ease' }}>
          {citations.map((c) => (
            <SourceCard key={`${c.id}-${c.source}`} citation={c} />
          ))}
        </div>
      )}
    </div>
  )
}

function FollowUpSuggestions({ questions, onSelect }: { questions: string[]; onSelect: (q: string) => void }) {
  if (!questions.length) return null
  return (
    <div className="mt-4">
      <p className="text-xs font-semibold text-muted mb-2 uppercase tracking-wider">Continue exploring</p>
      <div className="flex flex-col gap-1.5">
        {questions.map((q) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-left text-sm transition-all duration-150"
            style={{
              backgroundColor: 'rgb(var(--bg-tertiary))',
              color: 'rgb(var(--text-secondary))',
              border: '1px solid rgb(var(--border))',
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'rgb(var(--brand))'
              e.currentTarget.style.color = 'rgb(var(--text-primary))'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = 'rgb(var(--border))'
              e.currentTarget.style.color = 'rgb(var(--text-secondary))'
            }}
          >
            <svg className="w-3.5 h-3.5 flex-shrink-0 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
            </svg>
            {q}
          </button>
        ))}
      </div>
    </div>
  )
}

export function MessageBubble({ message, onFollowUp }: Props) {
  const isUser = message.role === 'user'
  const isStreaming = message.status === 'streaming'

  if (isUser) {
    return (
      <div className="flex justify-end animate-slide-up">
        <div
          className="max-w-[85%] px-4 py-3 rounded-2xl rounded-tr-sm text-sm leading-relaxed"
          style={{ backgroundColor: 'rgb(var(--brand))', color: 'white' }}
        >
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="flex gap-3 animate-slide-up">
      {/* Avatar */}
      <div
        className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5"
        style={{ background: 'linear-gradient(135deg, rgb(var(--brand)) 0%, rgb(56 189 248 / 0.6) 100%)' }}
      >
        <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
        </svg>
      </div>

      <div className="flex-1 min-w-0">
        <AgentTrace
          plan={message.plan}
          agentActions={message.agentActions}
          sourcesCount={message.sources_count ?? message.citations.length}
          confidence={message.confidence}
          hideRagUI={message.hideRagUI}
          status={message.status}
        />

        {message.status === 'error' ? (
          <div
            className="px-4 py-3 rounded-xl text-sm"
            style={{ backgroundColor: 'rgb(var(--accent-red) / 0.1)', color: 'rgb(var(--accent-red))' }}
          >
            ⚠ {message.content}
          </div>
        ) : (
          <div className={`markdown-body text-sm leading-relaxed ${isStreaming && !message.content ? 'animate-pulse' : ''}`}>
            {!message.content && isStreaming ? (
              <div className="flex gap-1 py-2">
                <span className="w-1.5 h-1.5 rounded-full animate-pulse-dot" style={{ backgroundColor: 'rgb(var(--brand))' }} />
                <span className="w-1.5 h-1.5 rounded-full animate-pulse-dot [animation-delay:0.2s]" style={{ backgroundColor: 'rgb(var(--brand))' }} />
                <span className="w-1.5 h-1.5 rounded-full animate-pulse-dot [animation-delay:0.4s]" style={{ backgroundColor: 'rgb(var(--brand))' }} />
              </div>
            ) : (
              <>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {message.content + (isStreaming ? '▋' : '')}
                </ReactMarkdown>
                {!isStreaming && !message.hideRagUI && (
                  <>
                    <InlineSources citations={message.citations} />
                    <FollowUpSuggestions questions={message.follow_up_questions} onSelect={onFollowUp} />
                  </>
                )}
              </>
            )}
          </div>
        )}

        {message.status === 'complete' && message.latency_s && (
          <p className="text-xs text-muted mt-2">{message.latency_s.toFixed(1)}s</p>
        )}
      </div>
    </div>
  )
}
