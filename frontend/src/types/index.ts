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

export interface AgentAction {
  tool: string
  args: Record<string, unknown>
  iteration: number
  observation?: string  // filled in when agent_observation SSE arrives
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations: Citation[]
  follow_up_questions: string[]
  plan?: AgentPlan
  agentActions?: AgentAction[]
  confidence?: number
  hideRagUI?: boolean
  status: 'streaming' | 'complete' | 'error'
  created_at: number
  sources_count?: number
  latency_s?: number
}

export interface StreamEvent {
  event: 'plan' | 'chunks' | 'web_search' | 'token' | 'citations' | 'follow_ups' | 'reflection' | 'done' | 'error' | 'agent_action' | 'agent_observation'
  data: Record<string, unknown>
}

export type Theme = 'light' | 'dark' | 'system'
