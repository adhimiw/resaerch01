import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import { Users } from 'lucide-react';
import './ChatInterface.css';

export default function ChatInterface({
  conversation,
  onSendMessage,
  isLoading,
  useMultiPerson,
  selectedPersonaIds = []
}) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  if (!conversation) {
    return (
      <div className="chat-welcome">
        <h2>Welcome to LLM Council</h2>
        <p>Create a new conversation to get started</p>
        {useMultiPerson && selectedPersonaIds.length > 0 && (
          <div className="welcome-multi-person">
            <Users size={20} />
            <span>{selectedPersonaCount} personas ready to discuss</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="chat-interface">
      {useMultiPerson && selectedPersonaIds.length > 0 && (
        <div className="multi-person-header">
          <Users size={16} />
          <span>Multi-Person Council Mode</span>
        </div>
      )}

      {conversation.messages.length === 0 ? (
        <div className="chat-welcome chat-welcome-small">
          <h2>Start a conversation</h2>
          <p>Ask a question to consult the council</p>
        </div>
      ) : (
        <div className="message-list">
          {conversation.messages.map((msg, index) => (
            <div key={index} className={`message-item ${msg.role}`}>
              {msg.role === 'user' ? (
                <div className="user-message">
                  <strong>You</strong>
                  <div className="message-content"><ReactMarkdown>{msg.content}</ReactMarkdown></div>
                </div>
              ) : (
                <div className="llm-council-message">
                  <strong>LLM Council</strong>
                  
                  {/* Show if this is a multi-person response */}
                  {msg.metadata?.use_multi_person && (
                    <div className="multi-person-badge">
                      <Users size={14} />
                      Multi-Person Discussion
                    </div>
                  )}
                  
                  {/* Stage 1 */}
                  {msg.loading?.stage1 && (
                    <div className="stage-loading">
                      <p>Running Stage 1: Collecting individual responses...</p>
                    </div>
                  )}
                  {msg.stage1 && <Stage1 stage1Data={msg.stage1} isMultiPerson={msg.metadata?.use_multi_person} />}
                  
                  {/* Stage 2 */}
                  {msg.loading?.stage2 && (
                    <div className="stage-loading">
                      <p>Running Stage 2: Peer rankings...</p>
                    </div>
                  )}
                  {msg.stage2 && (
                    <div className="message-content"><Stage2 stage2Data={msg.stage2} /></div>
                  )}
                  
                  {/* Stage 3 */}
                  {msg.loading?.stage3 && (
                    <div className="stage-loading">
                      <p>Running Stage 3: Final synthesis...</p>
                    </div>
                  )}
                  {msg.stage3 && <Stage3 stage3Data={msg.stage3} />}
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="message-item llm-council">
              <strong>LLM Council</strong>
              <div className="message-content loading-indicator">
                <p>Consulting the council...</p>
              </div>
            </div>
          )}
        </div>
      )}

      <div ref={messagesEndRef} />

      <form onSubmit={handleSubmit} className="chat-input-form">
        {conversation.messages.length === 0 && (
          <p className="hint-text">Example: "What are the pros and cons of implementing a carbon tax?"</p>
        )}
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          rows={3}
          placeholder={isLoading ? "Consulting the council..." : "Ask a question..."}
        />
        <button type="submit" disabled={!input.trim() || isLoading}>
          Send
        </button>
      </form>
    </div>
  );
}
