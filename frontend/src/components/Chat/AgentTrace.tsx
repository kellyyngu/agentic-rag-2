import { AgentPlan, AgentAction } from '../../types'

interface Props {
  plan?: AgentPlan
  agentActions?: AgentAction[]
  sourcesCount?: number
  confidence?: number
  hideRagUI?: boolean
  status: 'streaming' | 'complete' | 'error'
}

function Step({
  icon,
  label,
  active,
  done,
}: {
  icon: React.ReactNode
  label: string
  active?: boolean
  done?: boolean
}) {
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
        <span
          className="inline-block w-1.5 h-1.5 rounded-full animate-pulse-dot"
          style={{ backgroundColor: 'rgb(var(--brand))' }}
        />
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

/** Single tool call pill shown during / after the orchestrator loop. */
function ToolCallPill({ action, isLast, isStreaming }: { action: AgentAction; isLast: boolean; isStreaming: boolean }) {
  const isActive = isLast && isStreaming && !action.observation
  const isDone   = Boolean(action.observation)

  const toolLabel =
    action.tool === 'retrieve_documents'
      ? `retrieve("${String(action.args.query ?? '').slice(0, 28)}${String(action.args.query ?? '').length > 28 ? '…' : ''}")`
      : action.tool === 'web_search'
      ? `web("${String(action.args.query ?? '').slice(0, 28)}${String(action.args.query ?? '').length > 28 ? '…' : ''}")`
      : action.tool

  // Derive quality badge from observation string
  const obs = action.observation ?? ''
  const isGood  = obs.includes('GOOD')
  const isWeak  = obs.includes('WEAK')
  const isNoNew = obs.includes('Duplicate') || obs.includes('already retrieved')

  return (
    <div
      className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs transition-all duration-300"
      style={{
        backgroundColor: isDone
          ? isGood
            ? 'rgb(var(--accent-green) / 0.12)'
            : isWeak
            ? 'rgb(var(--accent-amber) / 0.12)'
            : 'rgb(var(--bg-tertiary))'
          : isActive
          ? 'rgb(var(--brand) / 0.15)'
          : 'rgb(var(--bg-tertiary))',
        color: isDone
          ? isGood
            ? 'rgb(var(--accent-green))'
            : isWeak
            ? 'rgb(var(--accent-amber))'
            : 'rgb(var(--text-muted))'
          : isActive
          ? 'rgb(var(--brand))'
          : 'rgb(var(--text-muted))',
        opacity: !isDone && !isActive ? 0.5 : 1,
      }}
    >
      {isActive ? (
        <span
          className="inline-block w-1.5 h-1.5 rounded-full animate-pulse-dot flex-shrink-0"
          style={{ backgroundColor: 'rgb(var(--brand))' }}
        />
      ) : isDone ? (
        isNoNew ? (
          <svg className="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        ) : (
          <svg className="w-3 h-3 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
          </svg>
        )
      ) : (
        <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ backgroundColor: 'currentColor' }} />
      )}
      <span className="font-medium font-mono">{toolLabel}</span>
      {isDone && (isGood || isWeak) && (
        <span
          className="ml-0.5 px-1 py-px rounded text-[9px] font-bold tracking-wide"
          style={{
            backgroundColor: isGood
              ? 'rgb(var(--accent-green) / 0.2)'
              : 'rgb(var(--accent-amber) / 0.2)',
          }}
        >
          {isGood ? 'GOOD' : 'WEAK'}
        </span>
      )}
    </div>
  )
}

const ChevronRight = () => (
  <svg className="w-3 h-3 text-muted flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
    <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
  </svg>
)

export function AgentTrace({ plan, agentActions, sourcesCount, confidence, hideRagUI, status }: Props) {
  if (hideRagUI) return null

  const hasSources   = (sourcesCount ?? 0) > 0
  const isStreaming  = status === 'streaming'
  const isDone       = status === 'complete'
  const hasActions   = (agentActions?.length ?? 0) > 0

  // Orchestrating step: active while streaming with no actions yet, done when first action arrives
  const orchestratingActive = isStreaming && !hasActions
  const orchestratingDone   = hasActions

  return (
    <div className="flex flex-wrap items-center gap-1.5 mb-2">
      {/* Step 1: Orchestrating */}
      <Step
        icon={<span>🤖</span>}
        label="Orchestrating"
        done={orchestratingDone}
        active={orchestratingActive}
      />

      {/* Tool call pills — one per agent_action event */}
      {agentActions?.map((action, i) => (
        <div key={`${action.tool}-${action.iteration}-${i}`} className="flex items-center gap-1.5">
          <ChevronRight />
          <ToolCallPill
            action={action}
            isLast={i === (agentActions?.length ?? 0) - 1}
            isStreaming={isStreaming}
          />
        </div>
      ))}

      {/* Step 2: Generating */}
      {(hasActions || hasSources || (!isStreaming && !hasActions)) && (
        <>
          <ChevronRight />
          <Step
            icon={<span>✍️</span>}
            label={hasSources ? `${sourcesCount} chunks` : 'Generating'}
            done={isDone}
            active={hasSources && isStreaming}
          />
        </>
      )}

      {/* Confidence badge */}
      {confidence !== undefined && confidence > 0 && confidence <= 1 && (
        <>
          <ChevronRight />
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
