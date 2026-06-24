import { useState } from 'react'
import { ThemeToggle } from '../UI/ThemeToggle'
import { UploadModal } from '../UI/UploadModal'
import { useChatStore } from '../../store/chatStore'

export function Header() {
  const [showUpload, setShowUpload] = useState(false)
  const { docCount, clearMessages, citationPanelOpen, setCitationPanelOpen } = useChatStore()

  return (
    <>
      <header
        className="h-14 flex items-center justify-between px-4 border-b"
        style={{
          backgroundColor: 'rgb(var(--surface))',
          borderColor: 'rgb(var(--border))',
        }}
      >
        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: 'linear-gradient(135deg, rgb(var(--brand)), rgb(56 189 248 / 0.6))' }}
          >
            <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09z" />
            </svg>
          </div>
          <div>
            <span className="font-semibold text-sm text-primary">Agentic RAG</span>
            <span className="text-xs text-muted ml-1.5 hidden sm:inline">v2</span>
          </div>
        </div>

        {/* Center: doc count */}
        {docCount > 0 && (
          <div
            className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium"
            style={{ backgroundColor: 'rgb(var(--bg-tertiary))', color: 'rgb(var(--text-secondary))' }}
          >
            <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: 'rgb(var(--accent-green))' }} />
            {docCount.toLocaleString()} chunks indexed
          </div>
        )}

        {/* Actions */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowUpload(true)}
            className="btn-ghost flex items-center gap-1.5 text-xs font-medium"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
            <span className="hidden sm:inline">Upload</span>
          </button>

          <button
            onClick={() => setCitationPanelOpen(!citationPanelOpen)}
            className="btn-ghost flex items-center gap-1.5 text-xs font-medium"
            title="Toggle citations panel"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
            </svg>
            <span className="hidden sm:inline">Sources</span>
          </button>

          <button
            onClick={clearMessages}
            className="btn-ghost text-xs"
            title="Clear conversation"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
            </svg>
          </button>

          <div className="w-px h-4 mx-1" style={{ backgroundColor: 'rgb(var(--border))' }} />
          <ThemeToggle />
        </div>
      </header>

      {showUpload && <UploadModal onClose={() => setShowUpload(false)} />}
    </>
  )
}
