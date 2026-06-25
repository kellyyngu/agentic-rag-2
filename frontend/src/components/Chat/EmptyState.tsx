export function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full px-6 text-center animate-fade-in">
      {/* Icon */}
      <div className="mb-6">
        <div
          className="w-12 h-12 rounded-xl flex items-center justify-center"
          style={{ backgroundColor: 'rgb(var(--bg-tertiary))', border: '1px solid rgb(var(--border))' }}
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} style={{ color: 'rgb(var(--brand))' }}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
          </svg>
        </div>
      </div>

      <h2 className="text-lg font-semibold text-primary mb-2">Agentic RAG</h2>
      <p className="text-secondary text-sm max-w-xs leading-relaxed mb-1">
        Hybrid retrieval with grounded citations and agentic reasoning.
      </p>
      <p className="text-muted text-xs mb-8">Upload a document above, then ask anything.</p>

      {/* Feature pills */}
      <div className="flex flex-wrap gap-2 justify-center max-w-sm">
        {['Hybrid BM25 + vector', 'CrossEncoder reranking', 'Grounded citations', 'Reflection loop', 'Web search fallback'].map((f) => (
          <span
            key={f}
            className="text-xs px-2.5 py-1 rounded-full font-medium"
            style={{ backgroundColor: 'rgb(var(--bg-tertiary))', color: 'rgb(var(--text-secondary))' }}
          >
            {f}
          </span>
        ))}
      </div>
    </div>
  )
}
