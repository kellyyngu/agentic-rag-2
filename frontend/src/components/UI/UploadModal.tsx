import { useState, useRef, DragEvent, useEffect } from 'react'
import { useChatStore } from '../../store/chatStore'

const API_BASE = '/api'

interface DocInfo {
  source: string
  chunk_count: number
  page_count: number
}

function FileIcon() {
  return (
    <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
    </svg>
  )
}

function ext(name: string) {
  return name.split('.').pop()?.toUpperCase() ?? 'FILE'
}

function extColor(name: string) {
  const e = name.split('.').pop()?.toLowerCase()
  if (e === 'pdf') return '#ef4444'
  if (e === 'docx') return '#3b82f6'
  if (e === 'md') return '#8b5cf6'
  return '#6b7280'
}

interface Props {
  onClose: () => void
}

export function UploadModal({ onClose }: Props) {
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')
  const [docs, setDocs] = useState<DocInfo[]>([])
  const [loadingDocs, setLoadingDocs] = useState(true)
  const inputRef = useRef<HTMLInputElement>(null)
  const setDocCount = useChatStore((s) => s.setDocCount)

  const fetchDocs = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents/list`)
      if (res.ok) {
        const data = await res.json()
        setDocs(data.documents ?? [])
      }
    } catch {
      // silently ignore
    } finally {
      setLoadingDocs(false)
    }
  }

  useEffect(() => { fetchDocs() }, [])

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
      setMessage(`Indexed ${data.chunks_added} chunks from "${data.source}"`)
      const statsRes = await fetch(`${API_BASE}/documents/stats`)
      if (statsRes.ok) {
        const stats = await statsRes.json()
        setDocCount(stats.vector_count)
      }
      await fetchDocs()
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
      <div className="card w-full max-w-lg mx-4 p-0 overflow-hidden animate-slide-up" style={{ maxHeight: '85vh', display: 'flex', flexDirection: 'column' }}>

        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b" style={{ borderColor: 'rgb(var(--border))' }}>
          <h2 className="font-semibold text-primary text-base">Document Library</h2>
          <button onClick={onClose} className="btn-ghost p-1.5 rounded-md">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="overflow-y-auto flex-1 p-5 space-y-5">
          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            onClick={() => inputRef.current?.click()}
            className="border-2 border-dashed rounded-xl p-6 text-center cursor-pointer transition-all duration-200"
            style={{
              borderColor: dragging ? 'rgb(var(--brand))' : 'rgb(var(--border))',
              backgroundColor: dragging ? 'rgb(var(--brand) / 0.05)' : 'transparent',
            }}
          >
            <div
              className="w-10 h-10 rounded-xl flex items-center justify-center mx-auto mb-3"
              style={{ backgroundColor: 'rgb(var(--brand) / 0.1)' }}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5} style={{ color: 'rgb(var(--brand))' }}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
              </svg>
            </div>
            <p className="text-sm font-medium text-primary">Drop a file or click to browse</p>
            <p className="text-xs text-muted mt-1">PDF, TXT, MD, DOCX</p>
            <input
              ref={inputRef}
              type="file"
              accept=".pdf,.txt,.md,.docx"
              className="hidden"
              onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
            />
          </div>

          {/* Status messages */}
          {uploading && (
            <div className="flex items-center gap-3">
              <div className="w-4 h-4 border-2 border-t-transparent rounded-full animate-spin" style={{ borderColor: 'rgb(var(--brand))' }} />
              <span className="text-sm text-secondary">Processing and indexing…</span>
            </div>
          )}
          {message && (
            <p className="text-sm font-medium" style={{ color: 'rgb(var(--accent-green))' }}>✓ {message}</p>
          )}
          {error && (
            <p className="text-sm" style={{ color: 'rgb(var(--accent-red))' }}>⚠ {error}</p>
          )}

          {/* Document list */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-muted">Indexed documents</p>
              {docs.length > 0 && (
                <span
                  className="text-xs px-2 py-0.5 rounded-full font-semibold"
                  style={{ backgroundColor: 'rgb(var(--brand) / 0.1)', color: 'rgb(var(--brand))' }}
                >
                  {docs.length}
                </span>
              )}
            </div>

            {loadingDocs ? (
              <div className="space-y-2">
                {[1, 2].map((i) => (
                  <div key={i} className="h-14 rounded-xl animate-pulse" style={{ backgroundColor: 'rgb(var(--bg-tertiary))' }} />
                ))}
              </div>
            ) : docs.length === 0 ? (
              <div
                className="rounded-xl p-6 text-center"
                style={{ backgroundColor: 'rgb(var(--bg-tertiary))' }}
              >
                <FileIcon />
                <p className="text-xs text-muted mt-2">No documents indexed yet</p>
              </div>
            ) : (
              <div className="space-y-2">
                {docs.map((doc) => (
                  <div
                    key={doc.source}
                    className="flex items-center gap-3 px-3 py-2.5 rounded-xl"
                    style={{ backgroundColor: 'rgb(var(--bg-tertiary))' }}
                  >
                    {/* Extension badge */}
                    <div
                      className="w-9 h-9 rounded-lg flex items-center justify-center flex-shrink-0 text-white font-bold"
                      style={{ backgroundColor: extColor(doc.source), fontSize: '9px' }}
                    >
                      {ext(doc.source)}
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-primary truncate">{doc.source}</p>
                      <p className="text-xs text-muted mt-0.5">
                        {doc.chunk_count} chunks
                        {doc.page_count > 0 && ` · ${doc.page_count} pages`}
                      </p>
                    </div>

                    {/* Status dot */}
                    <div
                      className="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: 'rgb(var(--accent-green))' }}
                      title="Indexed"
                    />
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="px-5 py-4 border-t" style={{ borderColor: 'rgb(var(--border))' }}>
          <button onClick={onClose} className="btn-primary w-full">Done</button>
        </div>
      </div>
    </div>
  )
}
