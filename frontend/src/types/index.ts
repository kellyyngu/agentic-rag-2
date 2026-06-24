export interface Citation {
  id: string
  source: string
  page?: number
  excerpt: string
  relevance_score: number
}

export interface AgentPlan {
  sub_questions: string[]
  strategy: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  follow_up_questions: string[]
  plan?: AgentPlan
  confidence?: number
  isConversational?: boolean
  status: 'streaming' | 'complete' | 'error'
  created_at: number
  sources_count?: number
  latency_s?: number
}

export interface StreamEvent {
  event: 'plan' | 'chunks' | 'web_search' | 'token' | 'citations' | 'follow_ups' | 'reflection' | 'done' | 'error'
  data: Record<string, unknown>
}

export type Theme = 'light' | 'dark' | 'system'
