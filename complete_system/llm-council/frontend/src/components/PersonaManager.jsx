import { useState, useEffect } from 'react';
import { Plus, X, Edit2, Trash2, User, Check, AlertCircle } from 'lucide-react';
import { api } from '../api';
import './PersonaManager.css';

const ROLE_COLORS = {
  expert: '#3b82f6',
  moderator: '#8b5cf6',
  skeptic: '#f59e0b',
  advocate: '#ec4899',
  analyst: '#10b981'
};

const ROLE_LABELS = {
  expert: 'Expert',
  moderator: 'Moderator',
  skeptic: 'Skeptic',
  advocate: 'Advocate',
  analyst: 'Analyst'
};

export default function PersonaManager({ onSelectPersonas, selectedPersonaIds = [] }) {
  const [personas, setPersonas] = useState([]);
  const [roles, setRoles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [editingPersona, setEditingPersona] = useState(null);
  const [formData, setFormData] = useState({
    name: '',
    role: 'expert',
    description: '',
    expertise: '',
    personality: '',
    system_prompt: '',
    avatar_color: ''
  });
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    loadPersonas();
    loadRoles();
  }, []);

  const loadPersonas = async () => {
    try {
      const data = await api.listPersonas();
      setPersonas(data);
      setLoading(false);
    } catch (err) {
      setError('Failed to load personas');
      setLoading(false);
    }
  };

  const loadRoles = async () => {
    try {
      const data = await api.listPersonaRoles();
      setRoles(data.roles || []);
    } catch (err) {
      console.error('Failed to load roles:', err);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setSuccess(null);

    try {
      const expertiseList = formData.expertise
        .split(',')
        .map(s => s.trim())
        .filter(s => s);

      const data = {
        name: formData.name,
        role: formData.role,
        description: formData.description,
        expertise: expertiseList,
        personality: formData.personality,
        system_prompt: formData.system_prompt,
        avatar_color: formData.avatar_color || undefined
      };

      if (editingPersona) {
        await api.updatePersona(editingPersona.id, data);
        setSuccess('Persona updated successfully');
      } else {
        await api.createPersona(data);
        setSuccess('Persona created successfully');
      }

      setFormData({
        name: '',
        role: 'expert',
        description: '',
        expertise: '',
        personality: '',
        system_prompt: '',
        avatar_color: ''
      });
      setShowCreate(false);
      setEditingPersona(null);
      loadPersonas();
    } catch (err) {
      setError(err.message || 'Failed to save persona');
    }
  };

  const handleEdit = (persona) => {
    setEditingPersona(persona);
    setFormData({
      name: persona.name,
      role: persona.role,
      description: persona.description,
      expertise: persona.expertise.join(', '),
      personality: persona.personality,
      system_prompt: persona.system_prompt,
      avatar_color: persona.avatar_color || ''
    });
    setShowCreate(true);
  };

  const handleDelete = async (personaId) => {
    if (!confirm('Are you sure you want to delete this persona?')) return;
    
    try {
      await api.deletePersona(personaId);
      setSuccess('Persona deleted successfully');
      loadPersonas();
    } catch (err) {
      setError(err.message || 'Failed to delete persona');
    }
  };

  const handleToggleActive = async (persona) => {
    try {
      if (persona.is_active) {
        await api.deactivatePersona(persona.id);
      } else {
        await api.activatePersona(persona.id);
      }
      loadPersonas();
    } catch (err) {
      setError(err.message || 'Failed to update persona');
    }
  };

  const handlePersonaSelect = (personaId) => {
    onSelectPersonas(personaId);
  };

  if (loading) {
    return <div className="persona-manager loading">Loading personas...</div>;
  }

  return (
    <div className="persona-manager">
      <div className="persona-header">
        <h2><User size={20} /> Council Personas</h2>
        <button className="btn-primary" onClick={() => {
          setShowCreate(true);
          setEditingPersona(null);
          setFormData({
            name: '', role: 'expert', description: '',
            expertise: '', personality: '', system_prompt: '', avatar_color: ''
          });
        }}>
          <Plus size={16} /> Add Persona
        </button>
      </div>

      {error && (
        <div className="persona-alert error">
          <AlertCircle size={16} /> {error}
        </div>
      )}
      
      {success && (
        <div className="persona-alert success">
          <Check size={16} /> {success}
        </div>
      )}

      {showCreate && (
        <div className="persona-form-overlay">
          <div className="persona-form">
            <div className="form-header">
              <h3>{editingPersona ? 'Edit Persona' : 'Create New Persona'}</h3>
              <button onClick={() => {
                setShowCreate(false);
                setEditingPersona(null);
              }}><X size={20} /></button>
            </div>
            
            <form onSubmit={handleSubmit}>
              <div className="form-row">
                <label>Name</label>
                <input
                  type="text"
                  name="name"
                  value={formData.name}
                  onChange={handleInputChange}
                  placeholder="e.g., Dr. Sarah Chen"
                  required
                />
              </div>

              <div className="form-row">
                <label>Role</label>
                <select name="role" value={formData.role} onChange={handleInputChange}>
                  {roles.map(role => (
                    <option key={role} value={role}>{ROLE_LABELS[role] || role}</option>
                  ))}
                </select>
              </div>

              <div className="form-row">
                <label>Description</label>
                <textarea
                  name="description"
                  value={formData.description}
                  onChange={handleInputChange}
                  placeholder="Brief description of this persona..."
                  required
                />
              </div>

              <div className="form-row">
                <label>Expertise (comma-separated)</label>
                <input
                  type="text"
                  name="expertise"
                  value={formData.expertise}
                  onChange={handleInputChange}
                  placeholder="e.g., machine learning, statistics, data analysis"
                  required
                />
              </div>

              <div className="form-row">
                <label>Personality</label>
                <input
                  type="text"
                  name="personality"
                  value={formData.personality}
                  onChange={handleInputChange}
                  placeholder="e.g., Analytical, evidence-based, methodical"
                  required
                />
              </div>

              <div className="form-row">
                <label>System Prompt</label>
                <textarea
                  name="system_prompt"
                  value={formData.system_prompt}
                  onChange={handleInputChange}
                  placeholder="The system prompt that defines this persona's behavior..."
                  rows={4}
                  required
                />
              </div>

              <div className="form-row">
                <label>Avatar Color (optional)</label>
                <input
                  type="color"
                  name="avatar_color"
                  value={formData.avatar_color || '#3b82f6'}
                  onChange={handleInputChange}
                />
              </div>

              <div className="form-actions">
                <button type="button" className="btn-secondary" onClick={() => setShowCreate(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn-primary">
                  {editingPersona ? 'Update' : 'Create'} Persona
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <div className="persona-grid">
        {personas.length === 0 ? (
          <div className="no-personas">
            <p>No personas created yet. Create your first persona to get started!</p>
          </div>
        ) : (
          personas.map(persona => (
            <div
              key={persona.id}
              className={`persona-card ${!persona.is_active ? 'inactive' : ''} ${selectedPersonaIds.includes(persona.id) ? 'selected' : ''}`}
            >
              <div className="persona-avatar" style={{ backgroundColor: persona.avatar_color || ROLE_COLORS[persona.role] }}>
                {persona.name.charAt(0)}
              </div>
              
              <div className="persona-info">
                <h4>{persona.name}</h4>
                <span className="persona-role" style={{ color: ROLE_COLORS[persona.role] }}>
                  {ROLE_LABELS[persona.role] || persona.role}
                </span>
                <p className="persona-description">{persona.description}</p>
                
                <div className="persona-expertise">
                  {persona.expertise.slice(0, 3).map((skill, i) => (
                    <span key={i} className="skill-tag">{skill}</span>
                  ))}
                  {persona.expertise.length > 3 && (
                    <span className="skill-more">+{persona.expertise.length - 3}</span>
                  )}
                </div>
              </div>

              <div className="persona-actions">
                <button
                  className={`select-btn ${selectedPersonaIds.includes(persona.id) ? 'selected' : ''}`}
                  onClick={() => handlePersonaSelect(persona.id)}
                >
                  {selectedPersonaIds.includes(persona.id) ? 'Selected' : 'Select'}
                </button>
                
                <div className="action-buttons">
                  <button onClick={() => handleEdit(persona)} title="Edit"><Edit2 size={14} /></button>
                  <button onClick={() => handleToggleActive(persona)} title={persona.is_active ? 'Deactivate' : 'Activate'}>
                    {persona.is_active ? 'Disable' : 'Enable'}
                  </button>
                  <button onClick={() => handleDelete(persona.id)} title="Delete" className="delete"><Trash2 size={14} /></button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
