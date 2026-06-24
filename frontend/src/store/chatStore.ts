import { create } from 'zustand'
import { Message, Citation, AgentPlan, AgentAction, Theme } from '../types'

interface ChatState {
  messages: Message[]
  isStreaming: boolean
  activeCitation: Citation | null
  citationPanelOpen: boolean
  theme: Theme
  docCount: number

  addUserMessage: (content: string) => string
  startAssistantMessage: () => string
  appendToken: (id: string, token: string) => void
  setPlan: (id: string, plan: AgentPlan) => void
  setCitations: (id: string, citations: Citation[]) => void
  setFollowUps: (id: string, questions: string[]) => void
  setConfidence: (id: string, confidence: number) => void
  setSourcesCount: (id: string, count: number) => void
  setHideRagUI: (id: string) => void
  setAnswer: (id: string, text: string) => void
  addAgentAction: (id: string, action: AgentAction) => void
  updateLastAgentObservation: (id: string, tool: string, observation: string) => void
  finalizeMessage: (id: string, latency?: number) => void
  setError: (id: string, msg: string) => void
  setStreaming: (v: boolean) => void
  setActiveCitation: (c: Citation | null) => void
  setCitationPanelOpen: (v: boolean) => void
  setTheme: (t: Theme) => void
  setDocCount: (n: number) => void
  clearMessages: () => void
  regenerateLast: () => string | null
}

const uuid = () => Math.random().toString(36).slice(2, 10)

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  isStreaming: false,
  activeCitation: null,
  citationPanelOpen: false,
  theme: (localStorage.getItem('theme') as Theme) || 'system',
  docCount: 0,

  addUserMessage: (content) => {
    const id = uuid()
    set((s) => ({
      messages: [
        ...s.messages,
        { id, role: 'user', content, citations: [], follow_up_questions: [], status: 'complete', created_at: Date.now() },
      ],
    }))
    return id
  },

  startAssistantMessage: () => {
    const id = uuid()
    set((s) => ({
      messages: [
        ...s.messages,
        { id, role: 'assistant', content: '', citations: [], follow_up_questions: [], status: 'streaming', created_at: Date.now() },
      ],
    }))
    return id
  },

  appendToken: (id, token) => {
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, content: m.content + token } : m
      ),
    }))
  },

  setPlan: (id, plan) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, plan } : m)),
    }))
  },

  setCitations: (id, citations) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, citations } : m)),
    }))
  },

  setFollowUps: (id, follow_up_questions) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, follow_up_questions } : m)),
    }))
  },

  setConfidence: (id, confidence) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, confidence } : m)),
    }))
  },

  finalizeMessage: (id, latency) => {
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, status: 'complete', latency_s: latency } : m
      ),
    }))
  },

  setError: (id, msg) => {
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id ? { ...m, status: 'error', content: msg } : m
      ),
    }))
  },

  setSourcesCount: (id, count) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, sources_count: count } : m)),
    }))
  },

  setHideRagUI: (id) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, hideRagUI: true } : m)),
    }))
  },

  setAnswer: (id, text) => {
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, content: text } : m)),
    }))
  },

  addAgentAction: (id, action) => {
    set((s) => ({
      messages: s.messages.map((m) =>
        m.id === id
          ? { ...m, agentActions: [...(m.agentActions ?? []), action] }
          : m
      ),
    }))
  },

  updateLastAgentObservation: (id, tool, observation) => {
    set((s) => ({
      messages: s.messages.map((m) => {
        if (m.id !== id || !m.agentActions?.length) return m
        // Find the most recent action for this tool without an observation yet
        const actions = [...m.agentActions]
        const idx = actions.map((a, i) => ({ a, i }))
          .reverse()
          .find(({ a }) => a.tool === tool && !a.observation)?.i
        if (idx === undefined) return m
        actions[idx] = { ...actions[idx], observation }
        return { ...m, agentActions: actions }
      }),
    }))
  },

  setStreaming: (v) => set({ isStreaming: v }),
  setActiveCitation: (c) => set({ activeCitation: c, citationPanelOpen: c !== null }),
  setCitationPanelOpen: (v) => set({ citationPanelOpen: v }),
  setDocCount: (n) => set({ docCount: n }),

  setTheme: (t) => {
    localStorage.setItem('theme', t)
    set({ theme: t })
  },

  clearMessages: () => set({ messages: [] }),

  regenerateLast: () => {
    const msgs = get().messages
    const last = msgs.findLast((m: Message) => m.role === 'user')
    if (!last) return null
    set((s) => ({ messages: s.messages.filter((m: Message) => m.role === 'user' || m.created_at < last.created_at) }))
    return last.content
  },
}))
