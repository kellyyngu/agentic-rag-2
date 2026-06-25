import { useEffect, useRef } from 'react'
import { useChatStore } from '../../store/chatStore'
import { MessageBubble } from './MessageBubble'
import { EmptyState } from './EmptyState'
import { InputBar } from './InputBar'
import { useChat } from '../../hooks/useChat'

export function ChatWindow() {
  const { messages } = useChatStore()
  const { sendMessage } = useChat()
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
            {messages.map((msg) => (
              <MessageBubble
                key={msg.id}
                message={msg}
                onFollowUp={sendMessage}
              />
            ))}
            <div ref={bottomRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="max-w-3xl mx-auto w-full">
        <InputBar />
      </div>
    </div>
  )
}
