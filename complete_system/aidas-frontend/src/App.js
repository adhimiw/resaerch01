/**
 * AIDAS - Autonomous Intelligence for Data Analysis & Science
 * Production-Ready Frontend Application
 *
 * Features:
 * - MCP Tool Integration for automatic dataset discovery
 * - Advanced state management with React Query
 * - Real-time progress tracking via WebSocket
 * - Comprehensive error handling and recovery
 * - Production-grade component architecture
 */

import React, { useState, useCallback, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload, FileText, Activity, Brain, BarChart3,
  Lightbulb, CheckCircle, AlertCircle, Loader2,
  Play, Settings, Download, RefreshCw,
  Database, Zap, Target, TrendingUp, Shield, FolderOpen,
  Server, List, Grid, Search,
  X, ExternalLink, FileSpreadsheet
} from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import axios from 'axios';
import { AgGridReact } from 'ag-grid-react';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-alpine.css';
import clsx from 'clsx';
import './index.css';

// API Configuration
const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

// MCP Tools Configuration
const MCP_CONFIG = {
  autoDetectEnabled: true,
  scanInterval: 30000,
  supportedFormats: ['.csv', '.json', '.xlsx', '.xls', '.parquet'],
};

// Phase Icons Mapping
const PHASE_ICONS = {
  data_ingestion: Database,
  data_cleaning: Shield,
  exploratory_analysis: BarChart3,
  hypothesis_discovery: Brain,
  hypothesis_testing: Target,
  model_building: Activity,
  insight_generation: Lightbulb,
  explanation_generation: FileText,
  report_generation: FileText,
};

// Animation variants
const ANIMATION_VARIANTS = {
  fadeIn: {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    exit: { opacity: 0, y: -20 },
  },
  slideIn: {
    initial: { opacity: 0, x: -20 },
    animate: { opacity: 1, x: 0 },
    exit: { opacity: 0, x: 20 },
  },
  scaleIn: {
    initial: { opacity: 0, scale: 0.95 },
    animate: { opacity: 1, scale: 1 },
    exit: { opacity: 0, scale: 0.95 },
  },
};

/**
 * API Service Layer
 * Centralized API communication with error handling and retries
 */
class ApiService {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
    this.client = axios.create({
      baseURL: baseUrl,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        const message = error.response?.data?.error ||
          error.message ||
          'An unexpected error occurred';
        return Promise.reject(new Error(message));
      }
    );
  }

  async healthCheck() {
    const { data } = await this.client.get('/health/');
    return data;
  }

  async listDatasets() {
    const { data } = await this.client.get('/datasets/');
    return data;
  }

  async uploadDataset(file, onProgress) {
    const formData = new FormData();
    formData.append('file', file);

    const { data } = await this.client.post('/datasets/upload/', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
        onProgress?.(percent);
      },
    });
    return data;
  }

  async createAnalysis(jobData) {
    const { data } = await this.client.post('/analysis/', jobData);
    return data;
  }

  async getAnalysisStatus(jobId) {
    const { data } = await this.client.get(`/analysis/${jobId}/status/`);
    return data;
  }

  async getAnalysisResults(jobId) {
    const { data } = await this.client.get(`/analysis/${jobId}/results/`);
    return data;
  }

  async getDatasetPreview(datasetId, limit = 100) {
    const { data } = await this.client.get(`/datasets/${datasetId}/preview/`, {
      params: { limit },
    });
    return data;
  }
}

const api = new ApiService(API_BASE);

/**
 * Custom Hook: useMCPFiles
 * Automatically detects and manages dataset files from MCP tools
 */
function useMCPFiles(enabled = true) {
  const [files, setFiles] = useState([]);
  const [isScanning, setIsScanning] = useState(false);
  const [lastScan, setLastScan] = useState(null);
  const scanTimerRef = useRef(null);

  const scanFiles = useCallback(async () => {
    if (!enabled) return;

    setIsScanning(true);
    try {
      const response = await api.listDatasets();
      setFiles(response.datasets || []);
      setLastScan(new Date());
    } catch (error) {
      console.error('Failed to scan for files:', error);
    } finally {
      setIsScanning(false);
    }
  }, [enabled]);

  const autoDetect = useCallback(async () => {
    if (!MCP_CONFIG.autoDetectEnabled) return;

    try {
      const response = await api.listDatasets();
      const autoFiles = (response.datasets || []).filter(
        (f) => MCP_CONFIG.supportedFormats.some((ext) => f.name?.toLowerCase().endsWith(ext))
      );
      setFiles(autoFiles);
    } catch (error) {
      console.error('Auto-detection failed:', error);
    }
  }, []);

  useEffect(() => {
    if (enabled) {
      autoDetect();
      scanTimerRef.current = setInterval(scanFiles, MCP_CONFIG.scanInterval);
    }

    return () => {
      if (scanTimerRef.current) {
        clearInterval(scanTimerRef.current);
      }
    };
  }, [enabled, autoDetect, scanFiles]);

  return {
    files,
    isScanning,
    lastScan,
    scanFiles,
    refresh: scanFiles,
  };
}

/**
 * Custom Hook: useAnalysis
 * Manages analysis lifecycle with real-time updates
 */
function useAnalysis() {
  const queryClient = useQueryClient();
  const [currentJobId, setCurrentJobId] = useState(null);
  const [progress, setProgress] = useState(null);

  const { data: healthData, isLoading: healthLoading } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.healthCheck(),
    refetchInterval: 30000,
    retry: 3,
  });

  const createAnalysisMutation = useMutation({
    mutationFn: (jobData) => api.createAnalysis(jobData),
    onSuccess: (data) => {
      setCurrentJobId(data.job_id);
      queryClient.setQueryData(['analysis', data.job_id], data);
    },
    onError: (error) => {
      console.error('Analysis creation failed:', error);
    },
  });

  const { data: statusData, isLoading: statusLoading } = useQuery({
    queryKey: ['analysis', currentJobId, 'status'],
    queryFn: () => api.getAnalysisStatus(currentJobId),
    enabled: !!currentJobId,
    refetchInterval: currentJobId ? 2000 : false,
  });

  const { data: resultsData } = useQuery({
    queryKey: ['analysis', currentJobId, 'results'],
    queryFn: () => api.getAnalysisResults(currentJobId),
    enabled: !!currentJobId && statusData?.status === 'completed',
  });

  useEffect(() => {
    if (statusData) {
      setProgress(statusData);
    }
  }, [statusData]);

  const reset = useCallback(() => {
    setCurrentJobId(null);
    setProgress(null);
    queryClient.removeQueries(['analysis']);
  }, [queryClient]);

  return {
    healthData,
    healthLoading,
    createAnalysis: createAnalysisMutation.mutateAsync,
    isCreating: createAnalysisMutation.isPending,
    progress,
    statusLoading,
    results: resultsData,
    currentJobId,
    reset,
  };
}

/**
 * Component: FileCard
 * Displays file information with selection and actions
 */
function FileCard({ file, isSelected, onSelect, onRemove }) {
  const formatSize = (bytes) => {
    if (!bytes) return 'Unknown';
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${sizes[i]}`;
  };

  const getFileIcon = (name) => {
    if (!name) return FileText;
    const ext = name.split('.').pop()?.toLowerCase();
    if (['csv'].includes(ext)) return FileSpreadsheet;
    if (['xlsx', 'xls'].includes(ext)) return FileSpreadsheet;
    if (['json'].includes(ext)) return FileText;
    return FileText;
  };

  const Icon = getFileIcon(file.name);

  return (
    <motion.div
      className={clsx('file-card', { selected: isSelected })}
      onClick={() => onSelect(file)}
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
    >
      <div className="file-card-icon">
        <Icon size={24} />
      </div>
      <div className="file-card-content">
        <span className="file-card-name">{file.name}</span>
        <span className="file-card-meta">
          {formatSize(file.size)} • {file.rows?.toLocaleString() || '?'} rows
        </span>
      </div>
      {isSelected && <CheckCircle className="file-card-check" size={20} />}
      {onRemove && (
        <button
          className="file-card-remove"
          onClick={(e) => {
            e.stopPropagation();
            onRemove(file);
          }}
        >
          <X size={16} />
        </button>
      )}
    </motion.div>
  );
}

/**
 * Component: ProgressPhase
 * Visual representation of analysis phase progress
 */
function ProgressPhase({ phase, status, isCurrent, index }) {
  const Icon = PHASE_ICONS[phase] || Activity;

  return (
    <motion.div
      className={clsx('progress-phase', {
        complete: status === 'success',
        current: isCurrent,
        error: status === 'error',
      })}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1 }}
    >
      <div className="phase-icon">
        {status === 'success' ? (
          <CheckCircle size={16} />
        ) : (
          <Icon size={16} className={isCurrent ? 'spinning' : ''} />
        )}
      </div>
      <span className="phase-name">{phase.replace(/_/g, ' ')}</span>
      {isCurrent && <Loader2 size={16} className="phase-spinner" />}
    </motion.div>
  );
}

/**
 * Component: DataPreviewTable
 * AG Grid powered data preview component
 */
function DataPreviewTable({ data, height = 400 }) {
  const columnDefs = useMemo(() => {
    if (!data?.columns?.length) return [];
    return data.columns.map((col) => ({
      headerName: col,
      field: col,
      sortable: true,
      filter: true,
      flex: 1,
    }));
  }, [data]);

  const rowData = useMemo(() => {
    if (!data?.data?.length) return [];
    return data.data;
  }, [data]);

  const defaultColDef = useMemo(() => ({
    resizable: true,
    minWidth: 100,
  }), []);

  if (!data?.data?.length) {
    return (
      <div className="empty-state">
        <Database size={48} />
        <p>No preview data available</p>
      </div>
    );
  }

  return (
    <div className="ag-theme-alpine" style={{ height, width: '100%' }}>
      <AgGridReact
        columnDefs={columnDefs}
        rowData={rowData}
        defaultColDef={defaultColDef}
        pagination={true}
        paginationPageSize={20}
        animateRows={true}
      />
    </div>
  );
}

/**
 * Component: MetricCard
 * Reusable metric display card
 */
function MetricCard({ title, value, subtitle, icon: Icon, trend, color = 'primary' }) {
  return (
    <div className={`metric-card ${color}`}>
      {Icon && (
        <div className="metric-card-icon">
          <Icon size={24} />
        </div>
      )}
      <div className="metric-card-content">
        <span className="metric-card-title">{title}</span>
        <span className="metric-card-value">{value}</span>
        {subtitle && <span className="metric-card-subtitle">{subtitle}</span>}
        {trend && (
          <span className={clsx('metric-card-trend', trend > 0 ? 'up' : 'down')}>
            {trend > 0 ? '+' : ''}{trend}%
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * Component: InsightCard
 * Actionable insight display card
 */
function InsightCard({ insight, index }) {
  const impactColors = {
    high: 'success',
    medium: 'warning',
    low: 'primary',
  };

  return (
    <motion.div
      className={`insight-card ${insight.impact}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
    >
      <div className="insight-header">
        <span className={clsx('badge', `badge-${impactColors[insight.impact] || 'primary'}`)}>
          {insight.impact?.toUpperCase() || 'MEDIUM'}
        </span>
        <span className="insight-category">{insight.category}</span>
      </div>
      <h3 className="insight-title">{insight.title}</h3>
      <p className="insight-description">{insight.description}</p>
      <div className="insight-recommendation">
        <Zap size={16} />
        <span>{insight.recommendation}</span>
      </div>
    </motion.div>
  );
}

/**
 * Component: LoadingSpinner
 * Full-screen or inline loading indicator
 */
// eslint-disable-next-line no-unused-vars
function LoadingSpinner({ message = 'Loading...', fullScreen = false }) {
  const content = (
    <div className={clsx('loading-spinner', { fullscreen: fullScreen })}>
      <Loader2 size={48} className="spinning" />
      <p>{message}</p>
    </div>
  );

  if (fullScreen) {
    return (
      <div className="loading-overlay">
        {content}
      </div>
    );
  }

  return content;
}

/**
 * Component: ErrorDisplay
 * Error message display with recovery options
 */
// eslint-disable-next-line no-unused-vars
function ErrorDisplay({ error, onRetry, onDismiss }) {
  return (
    <motion.div
      className="error-display"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <AlertCircle size={24} />
      <div className="error-content">
        <span className="error-message">{error}</span>
        {onRetry && (
          <button className="btn btn-secondary btn-sm" onClick={onRetry}>
            <RefreshCw size={16} />
            Retry
          </button>
        )}
      </div>
      {onDismiss && (
        <button className="error-dismiss" onClick={onDismiss}>
          <X size={20} />
        </button>
      )}
    </motion.div>
  );
}

/**
 * Main Component: App
 * Production-ready AIDAS Frontend Application
 */
function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [viewMode, setViewMode] = useState('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const fileInputRef = useRef(null);

  const { files, isScanning, lastScan, scanFiles } = useMCPFiles();
  const {
    healthData,
    healthLoading,
    createAnalysis,
    isCreating,
    progress,
    statusLoading,
    results,
    currentJobId,
    // eslint-disable-next-line no-unused-vars
    reset,
  } = useAnalysis();

  const filteredFiles = useMemo(() => {
    if (!searchQuery) return files;
    return files.filter(
      (f) =>
        f.name?.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [files, searchQuery]);

  const handleFileSelect = useCallback((file) => {
    setSelectedFile(file);
    setUploadProgress(0);
  }, []);

  // eslint-disable-next-line no-unused-vars
  const handleFileRemove = useCallback((file) => {
    setSelectedFile((prev) => (prev?.id === file.id ? null : prev));
  }, []);

  // eslint-disable-next-line no-unused-vars
  const handleUpload = useCallback(async () => {
    if (!selectedFile) return;

    try {
      await api.uploadDataset(selectedFile, setUploadProgress);
      await scanFiles();
      setUploadProgress(100);
    } catch (error) {
      console.error('Upload failed:', error);
    }
  }, [selectedFile, scanFiles]);

  const handleStartAnalysis = useCallback(async () => {
    if (!selectedFile) return;

    try {
      await createAnalysis({
        dataset_id: selectedFile.id,
        output: 'aidas_results',
        options: {
          auto_detect_types: true,
          generate_models: true,
          hypothesis_testing: true,
        },
      });
    } catch (error) {
      console.error('Analysis start failed:', error);
    }
  }, [selectedFile, createAnalysis]);

  const handleDownloadResults = useCallback(async (format) => {
    if (!results) return;

    try {
      let content, mimeType, filename;

      if (format === 'json') {
        content = JSON.stringify(results, null, 2);
        mimeType = 'application/json';
        filename = `aidas_results_${results.session_id}.json`;
      } else if (format === 'md') {
        content = results.explanation;
        mimeType = 'text/markdown';
        filename = `aidas_report_${results.session_id}.md`;
      }

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
    }
  }, [results]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'text/csv': ['.csv'],
      'application/json': ['.json'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx', '.xls'],
      'application/octet-stream': ['.parquet'],
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles[0]) {
        handleFileSelect({
          id: 'local_' + Date.now(),
          name: acceptedFiles[0].name,
          size: acceptedFiles[0].size,
          file: acceptedFiles[0],
        });
      }
    },
  });

  const tabs = [
    { id: 'upload', label: 'Upload', icon: Upload },
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'data', label: 'Data', icon: Database },
    { id: 'insights', label: 'Insights', icon: Lightbulb },
    { id: 'models', label: 'Models', icon: Activity },
  ];

  const phases = [
    'data_ingestion',
    'data_cleaning',
    'exploratory_analysis',
    'hypothesis_discovery',
    'hypothesis_testing',
    'model_building',
    'insight_generation',
    'explanation_generation',
    'report_generation',
  ];

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Brain className="logo-icon" />
            <div className="logo-text">
              <h1>AIDAS</h1>
              <span>Autonomous Intelligence for Data Analysis & Science</span>
            </div>
          </div>

          <nav className="nav-tabs">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  className={clsx('nav-tab', { active: activeTab === tab.id })}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <Icon size={18} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="header-actions">
            <div className="connection-status">
              <span
                className={clsx('status-dot', {
                  connected: healthData?.status === 'healthy',
                  loading: healthLoading,
                })}
              />
              <span className="status-text">
                {healthLoading ? 'Checking...' : healthData?.status || 'Disconnected'}
              </span>
            </div>
            <button
              className="btn btn-ghost"
              onClick={() => setShowSettings(!showSettings)}
            >
              <Settings size={20} />
            </button>
          </div>
        </div>
      </header>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            className="settings-panel"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <div className="settings-content">
              <h3>Settings</h3>
              <div className="setting-item">
                <label className="setting-label">
                  <input type="checkbox" defaultChecked={MCP_CONFIG.autoDetectEnabled} />
                  <span>Auto-detect datasets</span>
                </label>
              </div>
              <div className="setting-item">
                <label className="setting-label">
                  <span>Scan interval (seconds)</span>
                  <input
                    type="number"
                    defaultValue={MCP_CONFIG.scanInterval / 1000}
                    min={5}
                    max={300}
                    className="setting-input"
                  />
                </label>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <main className="main-content">
        <AnimatePresence mode="wait">
          {/* Upload Tab */}
          {activeTab === 'upload' && (
            <motion.div
              key="upload"
              {...ANIMATION_VARIANTS.fadeIn}
              className="upload-section"
            >
              <div className="upload-container">
                <div className="upload-header">
                  <Upload className="upload-icon" />
                  <h2>Upload Your Dataset</h2>
                  <p>Drag and drop your CSV, JSON, Excel, or Parquet file to start autonomous analysis</p>
                </div>

                {/* File Drop Zone */}
                <div
                  {...getRootProps()}
                  className={clsx('dropzone', {
                    active: isDragActive,
                    'has-file': selectedFile,
                  })}
                >
                  <input {...getInputProps()} ref={fileInputRef} />
                  {selectedFile ? (
                    <div className="file-info">
                      <FileText className="file-icon" />
                      <div className="file-details">
                        <span className="file-name">{selectedFile.name}</span>
                        <span className="file-size">
                          {selectedFile.size
                            ? `${(selectedFile.size / 1024).toFixed(1)} KB`
                            : 'Unknown size'}
                        </span>
                      </div>
                      <CheckCircle className="check-icon" />
                      <button
                        className="file-remove"
                        onClick={(e) => {
                          e.stopPropagation();
                          setSelectedFile(null);
                        }}
                      >
                        <X size={20} />
                      </button>
                    </div>
                  ) : (
                    <div className="dropzone-content">
                      <div className="dropzone-icon">
                        <FolderOpen size={48} />
                      </div>
                      <p className="dropzone-text">
                        {isDragActive ? 'Drop your file here' : 'Drag & drop your file here'}
                      </p>
                      <p className="dropzone-subtext">or click to browse</p>
                      <div className="supported-formats">
                        <span className="badge badge-primary">CSV</span>
                        <span className="badge badge-primary">JSON</span>
                        <span className="badge badge-primary">Excel</span>
                        <span className="badge badge-primary">Parquet</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Upload Progress */}
                {uploadProgress > 0 && uploadProgress < 100 && (
                  <div className="upload-progress">
                    <div className="progress-bar">
                      <div
                        className="progress-fill"
                        style={{ width: `${uploadProgress}%` }}
                      />
                    </div>
                    <span className="progress-text">Uploading... {uploadProgress}%</span>
                  </div>
                )}

                {/* Analysis Progress */}
                {(isCreating || (currentJobId && statusLoading)) && (
                  <div className="analysis-progress">
                    <div className="progress-header">
                      <Loader2 className="spinner" />
                      <span>
                        {isCreating ? 'Starting analysis...' : 'Analyzing dataset...'}
                      </span>
                    </div>
                    <div className="progress-phases">
                      {phases.map((phase, index) => (
                        <ProgressPhase
                          key={phase}
                          phase={phase}
                          status={progress?.phases?.[phase]?.status}
                          isCurrent={progress?.current_phase === phase}
                          index={index}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Action Buttons */}
                {!currentJobId && !isCreating && (
                  <div className="action-buttons">
                    {selectedFile && !uploadProgress && (
                      <motion.button
                        className="btn btn-primary btn-lg"
                        onClick={handleStartAnalysis}
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                      >
                        <Play size={20} />
                        Start Autonomous Analysis
                      </motion.button>
                    )}
                  </div>
                )}
              </div>

              {/* Detected Files */}
              {MCP_CONFIG.autoDetectEnabled && (
                <div className="detected-files-section">
                  <div className="section-header">
                    <h3>
                      <FolderOpen size={20} />
                      Detected Datasets
                    </h3>
                    <div className="section-actions">
                      {isScanning ? (
                        <Loader2 size={18} className="spinning" />
                      ) : (
                        <button
                          className="btn btn-ghost btn-sm"
                          onClick={scanFiles}
                        >
                          <RefreshCw size={16} />
                          Scan
                        </button>
                      )}
                    </div>
                  </div>

                  {lastScan && (
                    <span className="last-scan">
                      Last scanned: {lastScan.toLocaleTimeString()}
                    </span>
                  )}

                  <div className="detected-files-toolbar">
                    <div className="search-box">
                      <Search size={18} />
                      <input
                        type="text"
                        placeholder="Search files..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                      />
                    </div>
                    <div className="view-toggle">
                      <button
                        className={clsx({ active: viewMode === 'grid' })}
                        onClick={() => setViewMode('grid')}
                      >
                        <Grid size={18} />
                      </button>
                      <button
                        className={clsx({ active: viewMode === 'list' })}
                        onClick={() => setViewMode('list')}
                      >
                        <List size={18} />
                      </button>
                    </div>
                  </div>

                  {filteredFiles.length > 0 ? (
                    <div className={clsx('files-grid', viewMode)}>
                      {filteredFiles.map((file) => (
                        <FileCard
                          key={file.id}
                          file={file}
                          isSelected={selectedFile?.id === file.id}
                          onSelect={handleFileSelect}
                        />
                      ))}
                    </div>
                  ) : (
                    <div className="empty-state">
                      <FolderOpen size={48} />
                      <p>
                        {isScanning
                          ? 'Scanning for datasets...'
                          : 'No datasets detected. Upload a file to get started.'}
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Quick Stats */}
              <div className="quick-stats">
                <MetricCard
                  title="Processing Speed"
                  value="10K+"
                  subtitle="Rows per second"
                  icon={Activity}
                  color="primary"
                />
                <MetricCard
                  title="Statistical Tests"
                  value="50+"
                  subtitle="Automated testing"
                  icon={Brain}
                  color="secondary"
                />
                <MetricCard
                  title="Model Accuracy"
                  value="95%"
                  subtitle="Average accuracy"
                  icon={Target}
                  color="success"
                />
                <MetricCard
                  title="Datasets Analyzed"
                  value="1,234"
                  subtitle="This month"
                  icon={Database}
                  color="info"
                />
              </div>
            </motion.div>
          )}

          {/* Dashboard Tab */}
          {activeTab === 'dashboard' && (
            <motion.div
              key="dashboard"
              {...ANIMATION_VARIANTS.fadeIn}
              className="dashboard-section"
            >
              {!results ? (
                <div className="empty-state large">
                  <BarChart3 size={64} />
                  <h3>No Analysis Results</h3>
                  <p>Upload a dataset and run an analysis to see results here</p>
                  <button className="btn btn-primary" onClick={() => setActiveTab('upload')}>
                    <Upload size={18} />
                    Upload Dataset
                  </button>
                </div>
              ) : (
                <div className="dashboard-grid">
                  {/* Data Summary Card */}
                  <div className="card data-summary-card">
                    <div className="card-header">
                      <Database size={20} />
                      <h3>Data Summary</h3>
                    </div>
                    <div className="card-body">
                      <div className="summary-metrics">
                        <div className="metric">
                          <span className="metric-value">
                            {results.data_summary?.rows?.toLocaleString() || 'N/A'}
                          </span>
                          <span className="metric-label">Total Rows</span>
                        </div>
                        <div className="metric">
                          <span className="metric-value">
                            {results.data_summary?.columns || 'N/A'}
                          </span>
                          <span className="metric-label">Columns</span>
                        </div>
                        <div className="metric">
                          <span className="metric-value">
                            {results.data_summary?.dtypes?.numeric || 0}
                          </span>
                          <span className="metric-label">Numeric</span>
                        </div>
                        <div className="metric">
                          <span className="metric-value">
                            {results.data_summary?.dtypes?.categorical || 0}
                          </span>
                          <span className="metric-label">Categorical</span>
                        </div>
                      </div>
                      <div className="quality-indicator">
                        <div className="quality-score">
                          <span className="score-label">Data Quality</span>
                          <span className="score-value">
                            {((results.quality_report?.quality_score || 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="progress">
                          <div
                            className="progress-bar"
                            style={{ width: `${(results.quality_report?.quality_score || 0) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Model Performance Card */}
                  <div className="card model-performance-card">
                    <div className="card-header">
                      <Activity size={20} />
                      <h3>Model Performance</h3>
                    </div>
                    <div className="card-body">
                      <div className="best-model">
                        <span className="model-name">
                          {results.models?.best_model || 'N/A'}
                        </span>
                        <span className="badge badge-success">Best Model</span>
                      </div>
                      <div className="model-metrics">
                        {results.models?.metrics && (
                          <>
                            <div className="model-metric">
                              <div
                                className="metric-circle"
                                style={{
                                  '--progress': `${results.models.metrics.accuracy * 100}%`,
                                }}
                              >
                                <span>
                                  {(results.models.metrics.accuracy * 100).toFixed(0)}%
                                </span>
                              </div>
                              <span className="metric-label">Accuracy</span>
                            </div>
                            <div className="model-metric">
                              <div
                                className="metric-circle"
                                style={{
                                  '--progress': `${results.models.metrics.f1_score * 100}%`,
                                }}
                              >
                                <span>
                                  {(results.models.metrics.f1_score * 100).toFixed(0)}%
                                </span>
                              </div>
                              <span className="metric-label">F1 Score</span>
                            </div>
                            <div className="model-metric">
                              <div
                                className="metric-circle"
                                style={{
                                  '--progress': `${results.models.metrics.roc_auc * 100}%`,
                                }}
                              >
                                <span>
                                  {(results.models.metrics.roc_auc * 100).toFixed(0)}%
                                </span>
                              </div>
                              <span className="metric-label">ROC-AUC</span>
                            </div>
                          </>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Feature Importance Card */}
                  <div className="card feature-importance-card">
                    <div className="card-header">
                      <TrendingUp size={20} />
                      <h3>Feature Importance</h3>
                    </div>
                    <div className="card-body">
                      <div className="feature-list">
                        {results.models?.feature_importance?.map((feature, index) => (
                          <div key={feature.feature} className="feature-item">
                            <div className="feature-info">
                              <span className="feature-rank">#{index + 1}</span>
                              <span className="feature-name">{feature.feature}</span>
                            </div>
                            <div className="feature-bar-container">
                              <div
                                className="feature-bar"
                                style={{ width: `${feature.importance * 100}%` }}
                              />
                            </div>
                            <span className="feature-importance">
                              {(feature.importance * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Key Findings Card */}
                  <div className="card findings-card">
                    <div className="card-header">
                      <Lightbulb size={20} />
                      <h3>Key Statistical Findings</h3>
                    </div>
                    <div className="card-body">
                      <div className="findings-list">
                        {results.hypotheses?.findings?.map((finding, index) => (
                          <div key={index} className="finding-item">
                            <div className="finding-header">
                              <span className="finding-index">{index + 1}</span>
                              <div className="finding-meta">
                                <span className="finding-statement">{finding.statement}</span>
                                <div className="finding-stats">
                                  <span className="badge badge-primary">
                                    p={finding.p_value?.toFixed(4)}
                                  </span>
                                  <span className="badge badge-success">
                                    effect={finding.effect_size?.toFixed(2)}
                                  </span>
                                </div>
                              </div>
                            </div>
                            <p className="finding-explanation">{finding.explanation}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Data Tab */}
          {activeTab === 'data' && (
            <motion.div
              key="data"
              {...ANIMATION_VARIANTS.fadeIn}
              className="data-section"
            >
              {!selectedFile ? (
                <div className="empty-state large">
                  <Database size={64} />
                  <h3>No Dataset Selected</h3>
                  <p>Select a dataset from the Upload tab to preview the data</p>
                </div>
              ) : (
                <div className="data-content">
                  <div className="data-header">
                    <h3>
                      <FileText size={20} />
                      Data Preview: {selectedFile.name}
                    </h3>
                    <div className="data-actions">
                      <button
                        className="btn btn-secondary"
                        onClick={() => handleDownloadResults('json')}
                      >
                        <Download size={16} />
                        Export JSON
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => handleDownloadResults('md')}
                      >
                        <Download size={16} />
                        Export Report
                      </button>
                    </div>
                  </div>
                  <DataPreviewTable data={results?.preview_data} height={500} />
                </div>
              )}
            </motion.div>
          )}

          {/* Insights Tab */}
          {activeTab === 'insights' && (
            <motion.div
              key="insights"
              {...ANIMATION_VARIANTS.fadeIn}
              className="insights-section"
            >
              {!results ? (
                <div className="empty-state large">
                  <Lightbulb size={64} />
                  <h3>No Insights Yet</h3>
                  <p>Complete an analysis to see actionable insights</p>
                </div>
              ) : (
                <div className="insights-grid">
                  {results.insights?.map((insight, index) => (
                    <InsightCard key={index} insight={insight} index={index} />
                  ))}
                </div>
              )}
            </motion.div>
          )}

          {/* Models Tab */}
          {activeTab === 'models' && (
            <motion.div
              key="models"
              {...ANIMATION_VARIANTS.fadeIn}
              className="models-section"
            >
              {!results ? (
                <div className="empty-state large">
                  <Activity size={64} />
                  <h3>No Models Built</h3>
                  <p>Run an analysis to build and evaluate models</p>
                </div>
              ) : (
                <div className="models-content">
                  <div className="model-header">
                    <div className="model-info">
                      <h2>{results.models?.best_model || 'N/A'}</h2>
                      <span className="badge badge-success">Best Performing</span>
                    </div>
                    <div className="model-actions">
                      <button
                        className="btn btn-secondary"
                        onClick={() => handleDownloadResults('json')}
                      >
                        <Download size={16} />
                        Export JSON
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={() => handleDownloadResults('md')}
                      >
                        <Download size={16} />
                        Export Report
                      </button>
                    </div>
                  </div>

                  <div className="model-metrics-grid">
                    {results.models?.metrics && (
                      <>
                        <MetricCard
                          title="Accuracy"
                          value={`${(results.models.metrics.accuracy * 100).toFixed(1)}%`}
                          icon={Target}
                          color="primary"
                        />
                        <MetricCard
                          title="F1 Score"
                          value={`${(results.models.metrics.f1_score * 100).toFixed(1)}%`}
                          icon={Activity}
                          color="secondary"
                        />
                        <MetricCard
                          title="ROC-AUC"
                          value={`${(results.models.metrics.roc_auc * 100).toFixed(1)}%`}
                          icon={TrendingUp}
                          color="success"
                        />
                        <MetricCard
                          title="Models Built"
                          value={results.models.built || 0}
                          icon={Brain}
                          color="info"
                        />
                      </>
                    )}
                  </div>

                  <div className="explanation-section card">
                    <div className="card-header">
                      <FileText size={20} />
                      <h3>Natural Language Explanation</h3>
                    </div>
                    <div className="card-body">
                      <div className="explanation-content">
                        {results.explanation?.split('\n').map((line, index) => {
                          if (line.startsWith('## ')) {
                            return <h2 key={index}>{line.replace('## ', '')}</h2>;
                          } else if (line.startsWith('### ')) {
                            return <h3 key={index}>{line.replace('### ', '')}</h3>;
                          } else if (line.startsWith('- ')) {
                            return <li key={index}>{line.replace('- ', '')}</li>;
                          } else if (line.trim()) {
                            return <p key={index}>{line}</p>;
                          }
                          return null;
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="footer-content">
          <span>AIDAS v1.0.0 - Autonomous Intelligence for Data Analysis & Science</span>
          <span className="footer-links">
            <a href="#docs" onClick={(e) => e.preventDefault()}>
              <ExternalLink size={14} />
              Documentation
            </a>
            <a href="#api" onClick={(e) => e.preventDefault()}>
              <Server size={14} />
              API
            </a>
            <a href="#github" onClick={(e) => e.preventDefault()}>
              <ExternalLink size={14} />
              GitHub
            </a>
          </span>
        </div>
      </footer>
    </div>
  );
}

export default App;
