interface Props {
  onPrompt: (q: string) => void
}

const SUGGESTIONS = [
  { icon: '📄', label: 'Summarize the document', prompt: 'Summarize the main points of the uploaded document' },
  { icon: '🔍', label: 'Key findings', prompt: 'What are the key findings or conclusions?' },
  { icon: '❓', label: 'Ask anything', prompt: 'What topics does this document cover?' },
  { icon: '📊', label: 'Compare concepts', prompt: 'Compare and contrast the main concepts discussed' },
]

export function EmptyState({ onPrompt }: Props) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 text-center animate-fade-in">
      {/* Animated logo */}
      <div className="relative mb-6">
        <div
          className="w-16 h-16 rounded-2xl flex items-center justify-center"
          style={{ background: 'linear-gradient(135deg, rgb(var(--brand)) 0%, rgb(99 179 237) 100%)' }}
        >
          <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
          </svg>
        </div>
        <div
          className="absolute -bottom-1 -right-1 w-5 h-5 rounded-full border-2 flex items-center justify-center text-xs"
          style={{
            backgroundColor: 'rgb(var(--surface))',
            borderColor: 'rgb(var(--surface))',
            background: 'rgb(var(--accent-green))',
            color: 'white',
          }}
        >
          ✓
        </div>
      </div>

      <h2 className="text-xl font-semibold text-primary mb-2">Agentic RAG v2</h2>
      <p className="text-secondary text-sm max-w-sm leading-relaxed mb-1">
        Powered by LangGraph + hybrid retrieval (BM25 + vector) with CrossEncoder reranking.
      </p>
      <p className="text-muted text-xs mb-8">Upload a document above, then ask anything.</p>

      {/* Feature pills */}
      <div className="flex flex-wrap gap-2 justify-center mb-8 max-w-sm">
        {['Query planning', 'Sub-question decomposition', 'Hybrid BM25 + vector', 'CrossEncoder reranking', 'Grounded citations', 'Reflection loop'].map((f) => (
          <span
            key={f}
            className="text-xs px-2.5 py-1 rounded-full font-medium"
            style={{ backgroundColor: 'rgb(var(--bg-tertiary))', color: 'rgb(var(--text-secondary))' }}
          >
            {f}
          </span>
        ))}
      </div>

      {/* Starter prompts */}
      <p className="text-xs text-muted mb-3 font-medium uppercase tracking-wider">Try asking</p>
      <div className="grid grid-cols-2 gap-2 w-full max-w-md">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.prompt}
            onClick={() => onPrompt(s.prompt)}
            className="flex items-center gap-2 p-3 rounded-xl text-left transition-all duration-150"
            style={{
              backgroundColor: 'rgb(var(--bg-tertiary))',
              border: '1px solid rgb(var(--border))',
            }}
            onMouseEnter={(e) => {
              ;(e.currentTarget as HTMLElement).style.borderColor = 'rgb(var(--brand))'
            }}
            onMouseLeave={(e) => {
              ;(e.currentTarget as HTMLElement).style.borderColor = 'rgb(var(--border))'
            }}
          >
            <span className="text-base">{s.icon}</span>
            <span className="text-xs text-secondary font-medium leading-snug">{s.label}</span>
          </button>
        ))}
      </div>
    </div>
  )
}
