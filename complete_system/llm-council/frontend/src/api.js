/**
 * API client for the LLM Council backend with Multi-Person support.
 */
const API_BASE = 'http://localhost:8001';

export const api = {
  /**
   * List all conversations.
   */
  async listConversations() {
    const response = await fetch(`${API_BASE}/api/conversations`);
    if (!response.ok) {
      throw new Error('Failed to list conversations');
    }
    return response.json();
  },

  /**
   * Create a new conversation.
   * @param {Object} options
   * @param {boolean} options.useMultiPerson - Whether to use multi-person mode
   * @param {string[]} options.personaIds - List of persona IDs
   */
  async createConversation(options = {}) {
    const response = await fetch(`${API_BASE}/api/conversations`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        use_multi_person: options.useMultiPerson || false,
        persona_ids: options.personaIds || []
      }),
    });
    if (!response.ok) {
      throw new Error('Failed to create conversation');
    }
    return response.json();
  },

  /**
   * Get a specific conversation.
   */
  async getConversation(conversationId) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}`
    );
    if (!response.ok) {
      throw new Error('Failed to get conversation');
    }
    return response.json();
  },

  /**
   * Send a message in a conversation.
   */
  async sendMessage(conversationId, content) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    return response.json();
  },

  /**
   * Send a message and receive streaming updates.
   */
  async sendMessageStream(conversationId, content, onEvent) {
    const response = await fetch(
      `${API_BASE}/api/conversations/${conversationId}/message/stream`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ content }),
      }
    );
    if (!response.ok) {
      throw new Error('Failed to send message');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e);
          }
        }
      }
    }
  },

  // ==================== Persona Endpoints ====================

  /**
   * List all available personas.
   * @param {boolean} activeOnly - Only return active personas
   */
  async listPersonas(activeOnly = true) {
    const response = await fetch(
      `${API_BASE}/api/personas?active_only=${activeOnly}`
    );
    if (!response.ok) {
      throw new Error('Failed to list personas');
    }
    return response.json();
  },

  /**
   * Get a specific persona.
   */
  async getPersona(personaId) {
    const response = await fetch(`${API_BASE}/api/personas/${personaId}`);
    if (!response.ok) {
      throw new Error('Failed to get persona');
    }
    return response.json();
  },

  /**
   * Create a new persona.
   */
  async createPersona(personaData) {
    const response = await fetch(`${API_BASE}/api/personas`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(personaData),
    });
    if (!response.ok) {
      throw new Error('Failed to create persona');
    }
    return response.json();
  },

  /**
   * Update a persona.
   */
  async updatePersona(personaId, personaData) {
    const response = await fetch(`${API_BASE}/api/personas/${personaId}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(personaData),
    });
    if (!response.ok) {
      throw new Error('Failed to update persona');
    }
    return response.json();
  },

  /**
   * Delete a persona.
   */
  async deletePersona(personaId) {
    const response = await fetch(`${API_BASE}/api/personas/${personaId}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error('Failed to delete persona');
    }
    return response.json();
  },

  /**
   * Activate a persona.
   */
  async activatePersona(personaId) {
    const response = await fetch(
      `${API_BASE}/api/personas/${personaId}/activate`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error('Failed to activate persona');
    }
    return response.json();
  },

  /**
   * Deactivate a persona.
   */
  async deactivatePersona(personaId) {
    const response = await fetch(
      `${API_BASE}/api/personas/${personaId}/deactivate`,
      { method: 'POST' }
    );
    if (!response.ok) {
      throw new Error('Failed to deactivate persona');
    }
    return response.json();
  },

  /**
   * List all available persona roles.
   */
  async listPersonaRoles() {
    const response = await fetch(`${API_BASE}/api/personas/roles`);
    if (!response.ok) {
      throw new Error('Failed to list persona roles');
    }
    return response.json();
  },

  // ==================== Multi-Person Chat ====================

  /**
   * Run a multi-person council discussion.
   */
  async multiPersonChat(content, personaIds, chairmanPersonaId = null) {
    const response = await fetch(`${API_BASE}/api/multi-person/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content,
        persona_ids: personaIds,
        chairman_persona_id: chairmanPersonaId
      }),
    });
    if (!response.ok) {
      throw new Error('Failed to run multi-person chat');
    }
    return response.json();
  },

  /**
   * Run a multi-person council discussion with streaming.
   */
  async multiPersonChatStream(content, personaIds, chairmanPersonaId, onEvent) {
    const response = await fetch(`${API_BASE}/api/multi-person/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        content,
        persona_ids: personaIds,
        chairman_persona_id: chairmanPersonaId
      }),
    });
    if (!response.ok) {
      throw new Error('Failed to run multi-person chat');
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value);
      const lines = chunk.split('\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          try {
            const event = JSON.parse(data);
            onEvent(event.type, event);
          } catch (e) {
            console.error('Failed to parse SSE event:', e);
          }
        }
      }
    }
  },

  /**
   * Health check.
   */
  async healthCheck() {
    const response = await fetch(`${API_BASE}/api/health`);
    if (!response.ok) {
      throw new Error('Health check failed');
    }
    return response.json();
  }
};
