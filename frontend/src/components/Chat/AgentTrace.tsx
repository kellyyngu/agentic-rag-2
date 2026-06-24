import { AgentPlan } from '../../types'

interface Props {
  plan?: AgentPlan
  sourcesCount?: number
  confidence?: number
  isConversational?: boolean
  status: 'streaming' | 'complete' | 'error'
}

function Step({ icon, label, active, done }: { icon: React.ReactNode; label: string; active?: boolean; done?: boolean }) {
  return (
    <div
      className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs transition-all duration-300 ${
        done ? 'opacity-100' : active ? 'opacity-100' : 'opacity-40'
      }`}
      style={{
        backgroundColor: done
          ? 'rgb(var(--accent-green) / 0.15)'
          : active
          ? 'rgb(var(--brand) / 0.15)'
          : 'rgb(var(--bg-tertiary))',
        color: done
          ? 'rgb(var(--accent-green))'
          : active
          ? 'rgb(var(--brand))'
          : 'rgb(var(--text-muted))',
      }}
    >
      {active && !done ? (
        <span className="inline-block w-1.5 h-1.5 rounded-full animate-pulse-dot" style={{ backgroundColor: 'rgb(var(--brand))' }} />
      ) : done ? (
        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
        </svg>
      ) : (
        icon
      )}
      <span className="font-medium">{label}</span>
    </div>
  )
}

export function AgentTrace({ plan, sourcesCount, confidence, isConversational, status }: Props) {
  // Conversational turns: no pipeline UI
  if (isConversational) return null

  const hasPlan = Boolean(plan)
  const hasSources = (sourcesCount ?? 0) > 0
  const isGenerating = status === 'streaming'
  const isDone = status === 'complete'

  return (
    <div className="flex flex-wrap items-center gap-1.5 mb-2">
      <Step
        icon={<span>🗺</span>}
        label={plan ? `Plan: ${plan.sub_questions.length} sub-qs` : 'Planning'}
        done={hasPlan}
        active={!hasPlan && isGenerating}
      />
      <svg className="w-3 h-3 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
      </svg>
      <Step
        icon={<span>🔍</span>}
        label={hasSources ? `${sourcesCount} chunks` : 'Retrieving'}
        done={hasSources}
        active={hasPlan && !hasSources}
      />
      <svg className="w-3 h-3 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
      </svg>
      <Step
        icon={<span>✍️</span>}
        label="Generating"
        done={isDone}
        active={hasSources && isGenerating}
      />
      {confidence !== undefined && confidence > 0 && (
        <>
          <svg className="w-3 h-3 text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
          </svg>
          <div
            className="flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium"
            style={{
              backgroundColor: 'rgb(var(--accent-green) / 0.15)',
              color: 'rgb(var(--accent-green))',
            }}
          >
            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            {Math.round(confidence * 100)}% confidence
          </div>
        </>
      )}
    </div>
  )
}
