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
            style={{ backgroundColor: 'rgb(var(--bg-tertiary))', border: '1px solid rgb(var(--border))' }}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.75} style={{ color: 'rgb(var(--brand))' }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
            </svg>
          </div>
          <span className="font-semibold text-sm text-primary">Agentic RAG</span>
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
