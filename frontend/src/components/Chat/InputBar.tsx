import { useState, useRef, KeyboardEvent } from 'react'
import { useChatStore } from '../../store/chatStore'
import { useChat } from '../../hooks/useChat'

export function InputBar() {
  const [text, setText] = useState('')
  const { isStreaming } = useChatStore()
  const { sendMessage, stop, regenerate } = useChat()
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const submit = async () => {
    const q = text.trim()
    if (!q || isStreaming) return
    setText('')
    if (textareaRef.current) textareaRef.current.style.height = 'auto'
    await sendMessage(q)
  }

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const autoResize = () => {
    const el = textareaRef.current
    if (el) {
      el.style.height = 'auto'
      el.style.height = Math.min(el.scrollHeight, 160) + 'px'
    }
  }

  return (
    <div
      className="border-t p-4"
      style={{
        backgroundColor: 'rgb(var(--surface))',
        borderColor: 'rgb(var(--border))',
      }}
    >
      <div
        className="flex items-end gap-2 rounded-2xl border px-4 py-3 transition-all duration-150 focus-within:ring-1"
        style={{
          backgroundColor: 'rgb(var(--bg-tertiary))',
          borderColor: 'rgb(var(--border))',
          '--tw-ring-color': 'rgb(var(--brand))',
        } as React.CSSProperties}
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => { setText(e.target.value); autoResize() }}
          onKeyDown={onKeyDown}
          placeholder="Ask anything about your documents…"
          rows={1}
          disabled={isStreaming}
          className="flex-1 bg-transparent text-sm text-primary placeholder:text-muted resize-none outline-none leading-relaxed"
          style={{ maxHeight: '160px' }}
        />

        <div className="flex items-center gap-1.5 flex-shrink-0">
          {/* Regenerate */}
          <button
            onClick={regenerate}
            disabled={isStreaming}
            title="Regenerate last response"
            className="p-1.5 rounded-lg transition-all duration-150"
            style={{ color: 'rgb(var(--text-muted))' }}
            onMouseEnter={(e) => (e.currentTarget.style.color = 'rgb(var(--text-secondary))')}
            onMouseLeave={(e) => (e.currentTarget.style.color = 'rgb(var(--text-muted))')}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M16.023 9.348h4.992v-.001M2.985 19.644v-4.992m0 0h4.992m-4.993 0l3.181 3.183a8.25 8.25 0 0013.803-3.7M4.031 9.865a8.25 8.25 0 0113.803-3.7l3.181 3.182m0-4.991v4.99" />
            </svg>
          </button>

          {/* Send / Stop */}
          {isStreaming ? (
            <button
              onClick={stop}
              className="w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-150"
              style={{ backgroundColor: 'rgb(var(--accent-red))' }}
              title="Stop generation"
            >
              <svg className="w-3.5 h-3.5 text-white" fill="currentColor" viewBox="0 0 24 24">
                <rect x="6" y="6" width="12" height="12" rx="1" />
              </svg>
            </button>
          ) : (
            <button
              onClick={submit}
              disabled={!text.trim()}
              className="w-8 h-8 rounded-xl flex items-center justify-center transition-all duration-150 disabled:opacity-40"
              style={{ backgroundColor: text.trim() ? 'rgb(var(--brand))' : 'rgb(var(--bg-hover))' }}
              title="Send (Enter)"
            >
              <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
              </svg>
            </button>
          )}
        </div>
      </div>

      <p className="text-xs text-muted text-center mt-2">
        Press <kbd className="px-1 py-0.5 rounded text-xs font-mono" style={{ backgroundColor: 'rgb(var(--bg-tertiary))' }}>Enter</kbd> to send, <kbd className="px-1 py-0.5 rounded text-xs font-mono" style={{ backgroundColor: 'rgb(var(--bg-tertiary))' }}>Shift+Enter</kbd> for new line
      </p>
    </div>
  )
}
