import { useState, useRef, DragEvent } from 'react'
import { useChatStore } from '../../store/chatStore'

const API_BASE = '/api'

interface Props {
  onClose: () => void
}

export function UploadModal({ onClose }: Props) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const setDocCount = useChatStore((s) => s.setDocCount)

  const upload = async (file: File) => {
    setUploading(true)
    setError('')
    setMessage('')
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_BASE}/documents/upload`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed')
      }
      const data = await res.json()
      setMessage(`✓ Indexed ${data.chunks_added} chunks from "${data.source}"`)
      // Refresh doc count
      const statsRes = await fetch(`${API_BASE}/documents/stats`)
      if (statsRes.ok) {
        const stats = await statsRes.json()
        setDocCount(stats.vector_count)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed')
    } finally {
      setUploading(false)
    }
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) upload(file)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in">
      <div className="card w-full max-w-md mx-4 p-6 animate-slide-up">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-primary text-lg">Upload Document</h2>
          <button onClick={onClose} className="btn-ghost p-1.5 rounded-md">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
          className={`
            border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200
            ${dragging ? 'border-brand-400 bg-brand-50 dark:bg-brand-900/20' : 'border-default hover:border-brand-300'}
          `}
          style={{ borderColor: dragging ? 'rgb(var(--brand))' : undefined }}
        >
          <svg className="w-8 h-8 mx-auto mb-3 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
          </svg>
          <p className="text-secondary text-sm font-medium">Drop a file here or click to browse</p>
          <p className="text-muted text-xs mt-1">PDF, TXT, MD, DOCX supported</p>
          <input
            ref={inputRef}
            type="file"
            accept=".pdf,.txt,.md,.docx"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
          />
        </div>

        {uploading && (
          <div className="mt-4 flex items-center gap-3">
            <div className="w-4 h-4 border-2 border-brand-400 border-t-transparent rounded-full animate-spin" />
            <span className="text-secondary text-sm">Processing and indexing...</span>
          </div>
        )}

        {message && (
          <p className="mt-4 text-sm font-medium" style={{ color: 'rgb(var(--accent-green))' }}>{message}</p>
        )}
        {error && (
          <p className="mt-4 text-sm" style={{ color: 'rgb(var(--accent-red))' }}>{error}</p>
        )}

        <button onClick={onClose} className="btn-primary w-full mt-5">
          Done
        </button>
      </div>
    </div>
  )
}
