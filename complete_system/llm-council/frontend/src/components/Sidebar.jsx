import { useState, useEffect } from 'react';
import { Users, MessageCircle, Plus, User } from 'lucide-react';
import './Sidebar.css';

export default function Sidebar({
  conversations,
  currentConversationId,
  onSelectConversation,
  onNewConversation,
  activeTab,
  onTabChange,
  useMultiPerson,
  selectedPersonaCount
}) {
  const [showModeToggle, setShowModeToggle] = useState(false);

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h1 className="sidebar-title">LLM Council</h1>
        
        <div className="tab-switcher">
          <button
            className={`tab-btn ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => onTabChange('chat')}
          >
            <MessageCircle size={16} /> Chat
          </button>
          <button
            className={`tab-btn ${activeTab === 'personas' ? 'active' : ''}`}
            onClick={() => onTabChange('personas')}
          >
            <Users size={16} /> Personas
            {selectedPersonaCount > 0 && (
              <span className="persona-count">{selectedPersonaCount}</span>
            )}
          </button>
        </div>
      </div>

      {activeTab === 'chat' && (
        <>
          <div className="new-conversation-section">
            <button className="new-conversation-button" onClick={onNewConversation}>
              <Plus size={18} /> New Conversation
            </button>
            
            {useMultiPerson && selectedPersonaCount > 0 && (
              <div className="multi-person-badge">
                <Users size={14} />
                {selectedPersonaCount} personas selected
              </div>
            )}
          </div>

          <div className="conversation-list">
            {conversations.length === 0 ? (
              <p className="no-conversations-message">No conversations yet</p>
            ) : (
              conversations.map((conv) => (
                <div
                  key={conv.id}
                  className={`conversation-item ${conv.id === currentConversationId ? 'active' : ''}`}
                  onClick={() => onSelectConversation(conv.id)}
                >
                  {conv.use_multi_person && (
                    <span className="multi-person-indicator" title="Multi-person conversation">
                      <Users size={12} />
                    </span>
                  )}
                  <span className="conversation-title">{conv.title || 'New Conversation'}</span>
                  <span className="message-count">{conv.message_count} messages</span>
                </div>
              ))
            )}
          </div>
        </>
      )}

      {activeTab === 'personas' && (
        <div className="personas-sidebar-info">
          <div className="info-card">
            <User size={24} />
            <h3>Multi-Person Council</h3>
            <p>Create and manage personas that will discuss your questions. Each persona has unique expertise, personality, and perspective.</p>
            <div className="features-list">
              <span>✓ Define custom personas</span>
              <span>✓ Multiple expertise areas</span>
              <span>✓ Collaborative discussion</span>
              <span>✓ Single API key usage</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
