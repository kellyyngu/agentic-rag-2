import { useRef, useCallback } from 'react'
import { useChatStore } from '../store/chatStore'
import { Citation, AgentPlan } from '../types'

const API_BASE = '/api'

function parseSSEBlock(block: string): { event: string; data: Record<string, unknown> } | null {
  const lines = block.trim().split('\n')
  let event = ''
  let dataStr = ''

  for (const line of lines) {
    if (line.startsWith('event: ')) event = line.slice(7).trim()
    else if (line.startsWith('data: ')) dataStr = line.slice(6).trim()
  }

  if (!event || !dataStr) return null
  try {
    return { event, data: JSON.parse(dataStr) }
  } catch {
    return null
  }
}

export function useChat() {
  const store = useChatStore()
  const abortRef = useRef<AbortController | null>(null)

  const sendMessage = useCallback(async (query: string) => {
    if (store.isStreaming) return

    store.addUserMessage(query)
    const assistantId = store.startAssistantMessage()
    store.setStreaming(true)

    abortRef.current = new AbortController()

    const history = store.messages
      .filter((m) => m.status === 'complete')
      .slice(-6)
      .map((m) => ({ role: m.role, content: m.content }))

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, conversation_history: history }),
        signal: abortRef.current.signal,
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`)
      if (!res.body) throw new Error('No response body')

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Split on double-newline (SSE event separator)
        const parts = buffer.split('\n\n')
        buffer = parts.pop() ?? ''  // last incomplete chunk stays in buffer

        for (const part of parts) {
          const parsed = parseSSEBlock(part)
          if (!parsed) continue
          const { event, data } = parsed

          switch (event) {
            case 'conversational':
              store.setConversational(assistantId)
              break
            case 'plan':
              store.setPlan(assistantId, data as unknown as AgentPlan)
              break
            case 'token':
              store.appendToken(assistantId, (data.text as string) || '')
              break
            case 'citations':
              store.setCitations(assistantId, (data.citations as Citation[]) || [])
              break
            case 'follow_ups':
              store.setFollowUps(assistantId, (data.questions as string[]) || [])
              break
            case 'reflection':
              store.setConfidence(assistantId, (data.confidence as number) || 0)
              break
            case 'done':
              store.finalizeMessage(assistantId, data.latency_s as number)
              break
            case 'error':
              store.setError(assistantId, (data.message as string) || 'Unknown error')
              break
          }
        }
      }

      // Ensure message is finalized even if no 'done' event
      store.finalizeMessage(assistantId)
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') {
        store.finalizeMessage(assistantId)
      } else {
        store.setError(assistantId, err instanceof Error ? err.message : 'Request failed')
      }
    } finally {
      store.setStreaming(false)
    }
  }, [store])

  const stop = useCallback(() => {
    abortRef.current?.abort()
    store.setStreaming(false)
  }, [store])

  const regenerate = useCallback(async () => {
    const query = store.regenerateLast()
    if (query) await sendMessage(query)
  }, [store, sendMessage])

  return { sendMessage, stop, regenerate }
}
