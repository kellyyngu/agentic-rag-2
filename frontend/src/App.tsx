import { useEffect } from 'react'
import { Header } from './components/Layout/Header'
import { ChatWindow } from './components/Chat/ChatWindow'
import { CitationPanel } from './components/Citations/CitationPanel'
import { useChatStore } from './store/chatStore'
import { useTheme } from './hooks/useTheme'

const API_BASE = import.meta.env.VITE_API_URL || '/api'

export default function App() {
  useTheme()
  const setDocCount = useChatStore((s) => s.setDocCount)

  useEffect(() => {
    fetch(`${API_BASE}/documents/stats`)
      .then((r) => r.json())
      .then((d) => setDocCount(d.vector_count ?? 0))
      .catch(() => {})
  }, [setDocCount])

  return (
    <div
      className="flex flex-col h-screen overflow-hidden"
      style={{ backgroundColor: 'rgb(var(--bg-primary))' }}
    >
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <ChatWindow />
        <CitationPanel />
      </div>
    </div>
  )
}
