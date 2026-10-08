import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  loadOpenCV,
  extractFramesFromVideo,
  processFrames,
  selectOptimalFrames,
  mergeCoverage,
  calibrateCamera,
  drawDetection,
  exportCalibrationJSON,
  exportCalibrationYAML,
} from './calibration';

const STEPS = [
  { id: 'config', label: 'Configure', icon: '⚙️' },
  { id: 'capture', label: 'Capture', icon: '📹' },
  { id: 'review', label: 'Review', icon: '🔍' },
  { id: 'calibrate', label: 'Calibrate', icon: '📐' },
  { id: 'export', label: 'Export', icon: '💾' },
];

const DEFAULT_CONFIG = {
  patternType: 'checkerboard',
  rows: 6,
  cols: 9,
  squareSize: 25,
  distortionModel: 'standard',
  coverageGridRows: 4,
  coverageGridCols: 5,
};

export default function App() {
  const [currentStep, setCurrentStep] = useState(0);
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [cvStatus, setCvStatus] = useState('loading'); // loading | ready | error
  const [frames, setFrames] = useState([]);
  const [calibResult, setCalibResult] = useState(null);
  const [processing, setProcessing] = useState(null); // null | { message, progress }

  // Load OpenCV on mount
  useEffect(() => {
    loadOpenCV()
      .then(() => setCvStatus('ready'))
      .catch(() => setCvStatus('error'));
  }, []);

  const canProceed = useCallback(() => {
    switch (currentStep) {
      case 0: return config.rows > 0 && config.cols > 0 && config.squareSize > 0;
      case 1: return frames.filter(f => f.selected).length >= 4;
      case 2: return frames.filter(f => f.selected).length >= 4;
      case 3: return calibResult !== null;
      default: return true;
    }
  }, [currentStep, config, frames, calibResult]);

  const goNext = () => {
    if (currentStep < STEPS.length - 1) setCurrentStep(currentStep + 1);
  };

  const goBack = () => {
    if (currentStep > 0) setCurrentStep(currentStep - 1);
  };

  const goToStep = (idx) => {
    // Allow going back to any completed step or the current step
    if (idx <= currentStep) setCurrentStep(idx);
  };

  return (
    <div className="app-layout">
      <header className="app-header">
        <div className="app-logo">
          <div className="app-logo-icon">📷</div>
          <span className="app-logo-text">Camera Calibrator</span>
        </div>

        <div className="stepper">
          {STEPS.map((step, idx) => (
            <React.Fragment key={step.id}>
              {idx > 0 && <div className="stepper-divider" />}
              <div
                className={`stepper-step ${idx === currentStep ? 'active' : ''} ${idx < currentStep ? 'completed' : ''}`}
                onClick={() => goToStep(idx)}
              >
                <span className="stepper-step-number">
                  {idx < currentStep ? '✓' : idx + 1}
                </span>
                <span>{step.label}</span>
              </div>
            </React.Fragment>
          ))}
        </div>

        <div className="btn-group">
          <div className={`tooltip-wrapper`}>
            <div
              className={`cv-status-dot`}
              style={{
                width: 10, height: 10, borderRadius: '50%',
                background: cvStatus === 'ready' ? 'var(--accent-success)' : cvStatus === 'error' ? 'var(--accent-danger)' : 'var(--accent-warning)',
                boxShadow: cvStatus === 'ready' ? '0 0 8px var(--accent-success-glow)' : 'none',
              }}
            />
            <div className="tooltip">
              OpenCV: {cvStatus === 'ready' ? 'Ready' : cvStatus === 'error' ? 'Error' : 'Loading...'}
            </div>
          </div>
        </div>
      </header>

      <div className="app-content animate-in" key={currentStep}>
        {currentStep === 0 && (
          <ConfigStep
            config={config}
            setConfig={setConfig}
            cvStatus={cvStatus}
          />
        )}
        {currentStep === 1 && (
          <CaptureStep
            config={config}
            frames={frames}
            setFrames={setFrames}
            processing={processing}
            setProcessing={setProcessing}
            cvStatus={cvStatus}
          />
        )}
        {currentStep === 2 && (
          <ReviewStep
            config={config}
            frames={frames}
            setFrames={setFrames}
          />
        )}
        {currentStep === 3 && (
          <CalibrateStep
            config={config}
            frames={frames}
            setFrames={setFrames}
            calibResult={calibResult}
            setCalibResult={setCalibResult}
            processing={processing}
            setProcessing={setProcessing}
          />
        )}
        {currentStep === 4 && (
          <ExportStep
            config={config}
            frames={frames}
            calibResult={calibResult}
          />
        )}

        <div className="btn-group" style={{ justifyContent: 'space-between', marginTop: 'var(--space-8)' }}>
          <button
            className="btn btn-secondary"
            onClick={goBack}
            disabled={currentStep === 0}
          >
            ← Back
          </button>
          {currentStep < STEPS.length - 1 && (
            <button
              className="btn btn-primary"
              onClick={goNext}
              disabled={!canProceed()}
            >
              Next →
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

/* ========================================
   Step 1: Configuration
   ======================================== */

function ConfigStep({ config, setConfig, cvStatus }) {
  const update = (key, value) => {
    setConfig(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div>
      <div className="section-header">
        <h1 className="section-title">Configure Calibration</h1>
        <p className="section-desc">
          Set up your calibration pattern and camera parameters. These settings determine how the tool detects and measures your calibration target.
        </p>
      </div>

      {cvStatus === 'loading' && (
        <div className="alert alert-info" style={{ marginBottom: 'var(--space-6)' }}>
          <span className="alert-icon">⏳</span>
          <div>
            <strong>Loading OpenCV.js...</strong>
            <br />
            This may take a few seconds on first load. The library runs entirely in your browser.
          </div>
        </div>
      )}

      {cvStatus === 'error' && (
        <div className="alert alert-danger" style={{ marginBottom: 'var(--space-6)' }}>
          <span className="alert-icon">❌</span>
          <div>
            <strong>Failed to load OpenCV.js</strong>
            <br />
            Check your internet connection and try refreshing.
          </div>
        </div>
      )}

      <div className="two-col">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Calibration Pattern</h2>
            <p className="card-subtitle">Select the type of pattern you're using</p>
          </div>

          <div className="pattern-grid">
            {[
              { id: 'checkerboard', name: 'Checkerboard', icon: '♟️', desc: 'Standard black & white grid' },
              { id: 'circles', name: 'Circles Grid', icon: '⚫', desc: 'Symmetric circles pattern' },
              { id: 'circles_asymmetric', name: 'Asymmetric Circles', icon: '◐', desc: 'Offset circles for better accuracy' },
            ].map(p => (
              <div
                key={p.id}
                className={`pattern-card ${config.patternType === p.id ? 'selected' : ''}`}
                onClick={() => update('patternType', p.id)}
              >
                <div className="pattern-card-icon">{p.icon}</div>
                <div className="pattern-card-name">{p.name}</div>
                <div className="pattern-card-desc">{p.desc}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Board Dimensions</h2>
            <p className="card-subtitle">
              {config.patternType === 'checkerboard'
                ? 'Inner corners (not squares) count'
                : 'Number of circle centers'
              }
            </p>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Rows (inner)</label>
              <input
                type="number"
                className="form-input"
                value={config.rows}
                min={2}
                max={20}
                onChange={e => update('rows', parseInt(e.target.value) || 0)}
              />
              <span className="form-hint">
                {config.patternType === 'checkerboard'
                  ? `${config.rows + 1} squares tall`
                  : `${config.rows} circles tall`
                }
              </span>
            </div>
            <div className="form-group">
              <label className="form-label">Columns (inner)</label>
              <input
                type="number"
                className="form-input"
                value={config.cols}
                min={2}
                max={20}
                onChange={e => update('cols', parseInt(e.target.value) || 0)}
              />
              <span className="form-hint">
                {config.patternType === 'checkerboard'
                  ? `${config.cols + 1} squares wide`
                  : `${config.cols} circles wide`
                }
              </span>
            </div>
          </div>

          <div className="form-group">
            <label className="form-label">Square / Circle spacing (mm)</label>
            <input
              type="number"
              className="form-input"
              value={config.squareSize}
              min={1}
              step={0.1}
              onChange={e => update('squareSize', parseFloat(e.target.value) || 0)}
            />
            <span className="form-hint">Physical distance between adjacent corners/centers</span>
          </div>

          <div className="form-group">
            <label className="form-label">Distortion Model</label>
            <select
              className="form-select"
              value={config.distortionModel}
              onChange={e => update('distortionModel', e.target.value)}
            >
              <option value="standard">Standard (5 params) — rigid flat board</option>
              <option value="rational">Rational (8 params) — paper / slight flex</option>
              <option value="thin_prism">Thin Prism (12+ params) — non-flat surfaces</option>
            </select>
            <span className="form-hint">
              Use "Rational" or "Thin Prism" if your board is printed on paper and may not be perfectly flat
            </span>
          </div>
        </div>
      </div>

      {/* Pattern preview */}
      <div className="card" style={{ marginTop: 'var(--space-6)' }}>
        <div className="card-header">
          <h2 className="card-title">Pattern Preview</h2>
          <p className="card-subtitle">
            Your {config.patternType === 'checkerboard' ? 'checkerboard' : 'circles'} pattern:
            {' '}{config.rows} × {config.cols} inner points = {config.rows * config.cols} total detection points
          </p>
        </div>
        <PatternPreview config={config} />
      </div>
    </div>
  );
}

function PatternPreview({ config }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const { rows, cols, patternType } = config;

    const cellSize = 24;
    const padding = cellSize;
    const totalRows = patternType === 'checkerboard' ? rows + 1 : rows;
    const totalCols = patternType === 'checkerboard' ? cols + 1 : cols;

    canvas.width = totalCols * cellSize + padding * 2;
    canvas.height = totalRows * cellSize + padding * 2;

    // Background
    ctx.fillStyle = '#1a2235';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (patternType === 'checkerboard') {
      for (let r = 0; r < totalRows; r++) {
        for (let c = 0; c < totalCols; c++) {
          ctx.fillStyle = (r + c) % 2 === 0 ? '#ffffff' : '#222222';
          ctx.fillRect(
            padding + c * cellSize,
            padding + r * cellSize,
            cellSize,
            cellSize
          );
        }
      }

      // Draw inner corners
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const x = padding + (c + 1) * cellSize;
          const y = padding + (r + 1) * cellSize;
          ctx.fillStyle = '#6366f1';
          ctx.beginPath();
          ctx.arc(x, y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    } else {
      // Circles grid
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(padding, padding, totalCols * cellSize, totalRows * cellSize);

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          let x, y;
          if (patternType === 'circles_asymmetric') {
            x = padding + ((2 * c + r % 2) + 0.5) * cellSize;
            y = padding + (r + 0.5) * cellSize;
          } else {
            x = padding + (c + 0.5) * cellSize;
            y = padding + (r + 0.5) * cellSize;
          }
          ctx.fillStyle = '#222222';
          ctx.beginPath();
          ctx.arc(x, y, cellSize * 0.3, 0, Math.PI * 2);
          ctx.fill();

          ctx.fillStyle = '#6366f1';
          ctx.beginPath();
          ctx.arc(x, y, 2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }, [config]);

  return (
    <div style={{ display: 'flex', justifyContent: 'center' }}>
      <canvas
        ref={canvasRef}
        style={{ borderRadius: 'var(--radius-md)', maxWidth: '100%', height: 'auto' }}
      />
    </div>
  );
}

/* ========================================
   Step 2: Capture
   ======================================== */

function CaptureStep({ config, frames, setFrames, processing, setProcessing, cvStatus }) {
  const videoInputRef = useRef(null);
  const liveVideoRef = useRef(null);
  const [captureMode, setCaptureMode] = useState('upload'); // upload | live
  const [mediaStream, setMediaStream] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);

  // Clean up media stream on unmount
  useEffect(() => {
    return () => {
      if (mediaStream) {
        mediaStream.getTracks().forEach(t => t.stop());
      }
    };
  }, [mediaStream]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      setMediaStream(stream);
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream;
        liveVideoRef.current.play();
      }
    } catch (err) {
      console.error('Camera access denied:', err);
      alert('Camera access denied. Please enable camera permissions.');
    }
  };

  const stopCamera = () => {
    if (mediaStream) {
      mediaStream.getTracks().forEach(t => t.stop());
      setMediaStream(null);
    }
    if (isRecording) {
      stopRecording();
    }
  };

  const startRecording = () => {
    if (!mediaStream) return;
    chunksRef.current = [];
    const mr = new MediaRecorder(mediaStream, { mimeType: 'video/webm' });
    mr.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      const file = new File([blob], 'recording.webm', { type: 'video/webm' });
      handleVideoFile(file);
    };
    mr.start();
    mediaRecorderRef.current = mr;
    setIsRecording(true);
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleVideoFile = async (file) => {
    if (cvStatus !== 'ready') {
      alert('OpenCV is still loading. Please wait.');
      return;
    }

    setProcessing({ message: 'Extracting frames from video...', progress: 0 });
    const cv = window.cv;

    try {
      // Extract raw frames
      const rawFrames = await extractFramesFromVideo(file, {
        maxFrames: 150,
        frameInterval: 0.4,
        onProgress: (p) => setProcessing({
          message: `Extracting frames... ${Math.round(p * 100)}%`,
          progress: p * 0.3,
        }),
      });

      // Process frames — detect pattern + sharpness
      setProcessing({ message: 'Detecting calibration pattern...', progress: 0.3 });

      // Use setTimeout to avoid blocking UI
      await new Promise(resolve => setTimeout(resolve, 50));

      const processed = processFrames(cv, rawFrames, config, (p, msg) => {
        setProcessing({
          message: msg || 'Processing...',
          progress: 0.3 + p * 0.5,
        });
      });

      if (processed.length === 0) {
        setProcessing(null);
        alert(`No calibration pattern detected in the video.\n\nMake sure your ${config.patternType} (${config.rows}×${config.cols}) is fully visible in the frames.`);
        return;
      }

      // Select optimal subset
      setProcessing({ message: 'Selecting optimal frames...', progress: 0.85 });
      await new Promise(resolve => setTimeout(resolve, 50));

      const selectedIndices = selectOptimalFrames(processed, 25);
      const finalFrames = processed.map((frame, i) => ({
        ...frame,
        selected: selectedIndices.includes(i),
      }));

      setFrames(prev => [...prev, ...finalFrames]);
      setProcessing(null);
    } catch (err) {
      console.error('Processing error:', err);
      setProcessing(null);
      alert('Error processing video: ' + err.message);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      handleVideoFile(file);
    }
  };

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) handleVideoFile(file);
  };

  const selectedCount = frames.filter(f => f.selected).length;

  return (
    <div>
      <div className="section-header">
        <h1 className="section-title">Capture Video</h1>
        <p className="section-desc">
          Record or upload a video of your calibration board. Move the board slowly through all areas of the frame, tilting it at various angles.
        </p>
      </div>

      {/* Mode selector */}
      <div className="tabs">
        <button
          className={`tab ${captureMode === 'upload' ? 'active' : ''}`}
          onClick={() => { setCaptureMode('upload'); stopCamera(); }}
        >
          📁 Upload Video
        </button>
        <button
          className={`tab ${captureMode === 'live' ? 'active' : ''}`}
          onClick={() => { setCaptureMode('live'); }}
        >
          📹 Live Camera
        </button>
      </div>

      <div className="two-col-wide">
        <div>
          <div
            className={`capture-area ${processing ? 'active' : ''}`}
            onDrop={handleDrop}
            onDragOver={e => e.preventDefault()}
          >
            {captureMode === 'live' && mediaStream ? (
              <video ref={liveVideoRef} autoPlay playsInline muted style={{ transform: 'scaleX(-1)' }} />
            ) : captureMode === 'live' ? (
              <div className="capture-placeholder">
                <div className="capture-placeholder-icon">📹</div>
                <div>
                  <strong>Camera not started</strong>
                  <p style={{ fontSize: 'var(--font-sm)', color: 'var(--text-tertiary)', marginTop: 'var(--space-2)' }}>
                    Click "Start Camera" to begin capturing
                  </p>
                </div>
                <button className="btn btn-primary" onClick={startCamera}>
                  Start Camera
                </button>
              </div>
            ) : (
              <div className="capture-placeholder">
                <div className="capture-placeholder-icon">📁</div>
                <div>
                  <strong>Drop video here</strong>
                  <p style={{ fontSize: 'var(--font-sm)', color: 'var(--text-tertiary)', marginTop: 'var(--space-2)' }}>
                    or click to browse. MP4, WebM, MOV supported.
                  </p>
                </div>
                <button className="btn btn-primary" onClick={() => videoInputRef.current?.click()}>
                  Choose Video
                </button>
                <input
                  ref={videoInputRef}
                  type="file"
                  accept="video/*"
                  style={{ display: 'none' }}
                  onChange={handleFileSelect}
                />
              </div>
            )}

            {processing && (
              <div className="processing-overlay">
                <div className="spinner" style={{ width: 40, height: 40 }} />
                <div className="processing-text">{processing.message}</div>
                <div style={{ width: '60%' }}>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${processing.progress * 100}%` }} />
                  </div>
                </div>
              </div>
            )}
          </div>

          {captureMode === 'live' && mediaStream && (
            <div className="capture-controls">
              {!isRecording ? (
                <button className="btn btn-danger btn-lg" onClick={startRecording}>
                  🔴 Start Recording
                </button>
              ) : (
                <button className="btn btn-secondary btn-lg" onClick={stopRecording}>
                  ⏹️ Stop Recording
                </button>
              )}
              <button className="btn btn-ghost" onClick={stopCamera}>
                Stop Camera
              </button>
            </div>
          )}
        </div>

        <div>
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Capture Guide</h2>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-4)' }}>
              {[
                { icon: '🎯', text: 'Keep the full board visible in frame' },
                { icon: '🔄', text: 'Tilt the board at different angles (15-45°)' },
                { icon: '↔️', text: 'Move board to all corners & edges of frame' },
                { icon: '📏', text: 'Vary the distance (close, medium, far)' },
                { icon: '💡', text: 'Ensure even lighting, avoid harsh shadows' },
                { icon: '🐌', text: 'Move slowly to avoid motion blur' },
              ].map((tip, i) => (
                <div key={i} style={{ display: 'flex', gap: 'var(--space-3)', alignItems: 'flex-start' }}>
                  <span style={{ fontSize: '18px', flexShrink: 0 }}>{tip.icon}</span>
                  <span style={{ fontSize: 'var(--font-sm)', color: 'var(--text-secondary)' }}>{tip.text}</span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: 'var(--space-6)', padding: 'var(--space-4)', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)' }}>
              <div style={{ fontSize: 'var(--font-sm)', color: 'var(--text-tertiary)', marginBottom: 'var(--space-2)' }}>
                FRAMES DETECTED
              </div>
              <div style={{ fontSize: 'var(--font-2xl)', fontWeight: 800 }}>
                {frames.length}
                <span style={{ fontSize: 'var(--font-sm)', color: 'var(--text-secondary)', fontWeight: 400, marginLeft: 'var(--space-2)' }}>
                  ({selectedCount} selected)
                </span>
              </div>
              {selectedCount < 4 && selectedCount > 0 && (
                <div style={{ fontSize: 'var(--font-xs)', color: 'var(--accent-warning)', marginTop: 'var(--space-2)' }}>
                  ⚠️ Minimum 4 frames needed. {4 - selectedCount} more required.
                </div>
              )}
              {selectedCount >= 4 && (
                <div style={{ fontSize: 'var(--font-xs)', color: 'var(--accent-success)', marginTop: 'var(--space-2)' }}>
                  ✓ Enough frames for calibration. 15-25 recommended for best results.
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ========================================
   Step 3: Review
   ======================================== */

function ReviewStep({ config, frames, setFrames }) {
  const [viewMode, setViewMode] = useState('grid'); // grid | coverage
  const [selectedPreview, setSelectedPreview] = useState(null);

  const toggleFrame = (index) => {
    setFrames(prev => prev.map((f, i) =>
      i === index ? { ...f, selected: !f.selected } : f
    ));
  };

  const removeFrame = (index) => {
    setFrames(prev => prev.filter((_, i) => i !== index));
  };

  const selectedFrames = frames.filter(f => f.selected);
  const mergedCoverage = mergeCoverage(selectedFrames.map(f => f.coverage));

  // Compute coverage stats
  let coveredCells = 0;
  let totalCells = 0;
  const gridRows = config.coverageGridRows || 4;
  const gridCols = config.coverageGridCols || 5;

  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      totalCells++;
      const count = mergedCoverage.get(`${r},${c}`) || 0;
      if (count > 0) coveredCells++;
    }
  }

  const coveragePercent = totalCells > 0 ? Math.round((coveredCells / totalCells) * 100) : 0;

  return (
    <div>
      <div className="section-header">
        <h1 className="section-title">Review & Select Frames</h1>
        <p className="section-desc">
          Review detected frames, deselect bad ones, and check spatial coverage.
          Green = selected, dimmed = excluded.
        </p>
      </div>

      <div className="stats-grid" style={{ marginBottom: 'var(--space-6)' }}>
        <div className="stat-card">
          <div className="stat-value">{frames.length}</div>
          <div className="stat-label">Total Detected</div>
        </div>
        <div className="stat-card">
          <div className={`stat-value ${selectedFrames.length >= 15 ? 'good' : selectedFrames.length >= 4 ? 'warning' : 'bad'}`}>
            {selectedFrames.length}
          </div>
          <div className="stat-label">Selected</div>
        </div>
        <div className="stat-card">
          <div className={`stat-value ${coveragePercent > 80 ? 'good' : coveragePercent > 50 ? 'warning' : 'bad'}`}>
            {coveragePercent}%
          </div>
          <div className="stat-label">Coverage</div>
        </div>
        <div className="stat-card">
          <div className="stat-value" style={{ color: 'var(--accent-secondary)' }}>
            {selectedFrames.length > 0
              ? Math.round(selectedFrames.reduce((s, f) => s + f.sharpness, 0) / selectedFrames.length)
              : 0
            }
          </div>
          <div className="stat-label">Avg Sharpness</div>
        </div>
      </div>

      <div className="tabs" style={{ maxWidth: 300 }}>
        <button className={`tab ${viewMode === 'grid' ? 'active' : ''}`} onClick={() => setViewMode('grid')}>
          Frames
        </button>
        <button className={`tab ${viewMode === 'coverage' ? 'active' : ''}`} onClick={() => setViewMode('coverage')}>
          Coverage Map
        </button>
      </div>

      {viewMode === 'grid' ? (
        <div className="frame-grid">
          {frames.map((frame, i) => (
            <div
              key={i}
              className={`frame-card ${frame.selected ? 'selected' : 'rejected'}`}
              onClick={() => toggleFrame(i)}
            >
              <img src={frame.thumbnailUrl} alt={`Frame ${i + 1}`} />
              <div className="frame-card-overlay">
                <span className={`frame-card-badge ${frame.sharpness > 200 ? 'good' : frame.sharpness > 80 ? 'warning' : 'bad'}`}>
                  S: {Math.round(frame.sharpness)}
                </span>
                <span style={{ fontSize: 'var(--font-xs)', color: 'white' }}>
                  {frame.timestamp?.toFixed(1)}s
                </span>
              </div>
              <button
                className="frame-card-remove"
                onClick={(e) => { e.stopPropagation(); removeFrame(i); }}
              >
                ×
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="two-col">
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Spatial Coverage</h2>
              <p className="card-subtitle">Where in the image do your frames provide data?</p>
            </div>
            <div
              className="coverage-grid"
              style={{ gridTemplateColumns: `repeat(${gridCols}, 1fr)`, gridTemplateRows: `repeat(${gridRows}, 1fr)` }}
            >
              {Array.from({ length: gridRows * gridCols }, (_, idx) => {
                const r = Math.floor(idx / gridCols);
                const c = idx % gridCols;
                const count = mergedCoverage.get(`${r},${c}`) || 0;
                const level = count === 0 ? 'empty' : count < 3 ? 'partial' : count < 8 ? 'good' : 'excellent';
                return (
                  <div key={idx} className={`coverage-cell ${level}`}>
                    {count}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Coverage Analysis</h2>
            </div>

            {coveragePercent < 100 && (
              <div className="alert alert-warning" style={{ marginBottom: 'var(--space-4)' }}>
                <span className="alert-icon">💡</span>
                <div>
                  <strong>Missing coverage areas</strong>
                  <p style={{ marginTop: 'var(--space-1)' }}>
                    {(() => {
                      const missing = [];
                      for (let r = 0; r < gridRows; r++) {
                        for (let c = 0; c < gridCols; c++) {
                          if ((mergedCoverage.get(`${r},${c}`) || 0) === 0) {
                            const vPos = r < gridRows / 3 ? 'top' : r < (gridRows * 2) / 3 ? 'center' : 'bottom';
                            const hPos = c < gridCols / 3 ? 'left' : c < (gridCols * 2) / 3 ? 'center' : 'right';
                            missing.push(`${vPos}-${hPos}`);
                          }
                        }
                      }
                      const unique = [...new Set(missing)];
                      return `Hold the board in the ${unique.slice(0, 3).join(', ')} area${unique.length > 3 ? ` and ${unique.length - 3} more` : ''} of the frame.`;
                    })()}
                  </p>
                </div>
              </div>
            )}

            {coveragePercent === 100 && (
              <div className="alert alert-success" style={{ marginBottom: 'var(--space-4)' }}>
                <span className="alert-icon">✅</span>
                <div>
                  <strong>Full coverage achieved!</strong>
                  <p style={{ marginTop: 'var(--space-1)' }}>
                    All areas of the image have calibration data. This is great for accurate distortion correction across the full frame.
                  </p>
                </div>
              </div>
            )}

            <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-3)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 'var(--font-sm)' }}>
                <span style={{ color: 'var(--text-secondary)' }}>Coverage Quality</span>
                <span className={coveragePercent > 80 ? '' : ''} style={{
                  color: coveragePercent > 80 ? 'var(--accent-success)' : coveragePercent > 50 ? 'var(--accent-warning)' : 'var(--accent-danger)',
                  fontWeight: 600
                }}>
                  {coveragePercent > 80 ? 'Excellent' : coveragePercent > 50 ? 'Adequate' : 'Poor'}
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{
                    width: `${coveragePercent}%`,
                    background: coveragePercent > 80
                      ? 'var(--gradient-success)'
                      : coveragePercent > 50
                        ? 'linear-gradient(135deg, var(--accent-warning), #f59e0b)'
                        : 'linear-gradient(135deg, var(--accent-danger), #ef4444)',
                  }}
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ========================================
   Step 4: Calibrate
   ======================================== */

function CalibrateStep({ config, frames, setFrames, calibResult, setCalibResult, processing, setProcessing }) {
  const runCalibration = async () => {
    const cv = window.cv;
    const selectedFrames = frames.filter(f => f.selected);

    if (selectedFrames.length < 4) {
      alert('Need at least 4 selected frames.');
      return;
    }

    setProcessing({ message: 'Running camera calibration...', progress: 0.5 });

    // Use setTimeout to avoid blocking UI
    await new Promise(resolve => setTimeout(resolve, 100));

    try {
      const result = calibrateCamera(cv, selectedFrames, config);
      setCalibResult(result);
      setProcessing(null);
    } catch (err) {
      console.error('Calibration error:', err);
      setProcessing(null);
      alert('Calibration failed: ' + err.message);
    }
  };

  const removeWorstFrame = () => {
    if (!calibResult || calibResult.perFrameErrors.length === 0) return;

    // Find the frame with highest error
    const selectedFrames = frames.filter(f => f.selected);
    const worst = calibResult.perFrameErrors.reduce((a, b) => a.error > b.error ? a : b);

    // Map back to the frames array index
    let selectedIdx = 0;
    for (let i = 0; i < frames.length; i++) {
      if (frames[i].selected) {
        if (selectedIdx === worst.frameIndex) {
          setFrames(prev => prev.map((f, fi) =>
            fi === i ? { ...f, selected: false } : f
          ));
          setCalibResult(null);
          break;
        }
        selectedIdx++;
      }
    }
  };

  const errClass = (err) => {
    if (err < 0.3) return 'good';
    if (err < 0.8) return 'warning';
    return 'bad';
  };

  return (
    <div>
      <div className="section-header">
        <h1 className="section-title">Calibrate Camera</h1>
        <p className="section-desc">
          Run the calibration algorithm on selected frames and review the results.
        </p>
      </div>

      {!calibResult && (
        <div className="card" style={{ textAlign: 'center', padding: 'var(--space-12)' }}>
          {processing ? (
            <div>
              <div className="spinner" style={{ width: 48, height: 48, margin: '0 auto var(--space-4)' }} />
              <div className="processing-text">{processing.message}</div>
            </div>
          ) : (
            <div>
              <div style={{ fontSize: 48, marginBottom: 'var(--space-4)' }}>📐</div>
              <h2 style={{ marginBottom: 'var(--space-2)' }}>Ready to Calibrate</h2>
              <p style={{ color: 'var(--text-secondary)', marginBottom: 'var(--space-6)' }}>
                {frames.filter(f => f.selected).length} frames selected • {config.patternType} • {config.distortionModel} model
              </p>
              <button className="btn btn-primary btn-lg" onClick={runCalibration}>
                🚀 Run Calibration
              </button>
            </div>
          )}
        </div>
      )}

      {calibResult && (
        <div className="animate-in">
          <div className="stats-grid" style={{ marginBottom: 'var(--space-6)' }}>
            <div className="stat-card">
              <div className={`stat-value ${errClass(calibResult.reprojError)}`}>
                {calibResult.reprojError.toFixed(4)}
              </div>
              <div className="stat-label">Reprojection Error (px)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: 'var(--accent-secondary)' }}>
                {calibResult.cameraMatrix.fx.toFixed(1)}
              </div>
              <div className="stat-label">Focal Length X (px)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: 'var(--accent-secondary)' }}>
                {calibResult.cameraMatrix.fy.toFixed(1)}
              </div>
              <div className="stat-label">Focal Length Y (px)</div>
            </div>
            <div className="stat-card">
              <div className="stat-value" style={{ color: 'var(--text-accent)' }}>
                ({calibResult.cameraMatrix.cx.toFixed(1)}, {calibResult.cameraMatrix.cy.toFixed(1)})
              </div>
              <div className="stat-label">Principal Point</div>
            </div>
          </div>

          <div className="two-col">
            {/* Camera Matrix */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Camera Matrix (K)</h2>
              </div>
              <div className="matrix-display">
                <div className="matrix-label">Intrinsic Parameters</div>
                {calibResult.cameraMatrix.raw.map((row, r) => (
                  <div key={r} className="matrix-row">
                    {row.map((val, c) => (
                      <span key={c} className="matrix-value">
                        {val.toFixed(4)}
                      </span>
                    ))}
                  </div>
                ))}
              </div>

              <div style={{ marginTop: 'var(--space-4)' }}>
                <div className="matrix-label">Distortion Coefficients</div>
                <div className="matrix-display">
                  <div className="matrix-row" style={{ flexWrap: 'wrap' }}>
                    {calibResult.distCoeffs.values.map((v, i) => (
                      <span key={i} className="matrix-value" style={{ minWidth: 80 }}>
                        k{i + 1}: {v.toFixed(6)}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Per-frame errors */}
            <div className="card">
              <div className="card-header">
                <h2 className="card-title">Per-Frame Error</h2>
                <p className="card-subtitle">Reprojection error per frame — remove outliers to improve</p>
              </div>

              <div style={{ maxHeight: 300, overflowY: 'auto' }}>
                {calibResult.perFrameErrors
                  .sort((a, b) => b.error - a.error)
                  .map((fe, i) => (
                    <div
                      key={i}
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: 'var(--space-2) var(--space-3)',
                        borderRadius: 'var(--radius-sm)',
                        marginBottom: 2,
                        background: i === 0 ? 'rgba(239, 68, 68, 0.1)' : 'transparent',
                      }}
                    >
                      <span style={{ fontSize: 'var(--font-sm)', color: 'var(--text-secondary)' }}>
                        Frame {fe.frameIndex + 1}
                      </span>
                      <span
                        style={{
                          fontSize: 'var(--font-sm)',
                          fontWeight: 600,
                          color: fe.error < 0.3 ? 'var(--accent-success)' : fe.error < 0.8 ? 'var(--accent-warning)' : 'var(--accent-danger)',
                        }}
                      >
                        {fe.error.toFixed(4)} px
                      </span>
                    </div>
                  ))
                }
              </div>

              <div className="btn-group" style={{ marginTop: 'var(--space-4)' }}>
                <button className="btn btn-danger btn-sm" onClick={removeWorstFrame}>
                  Remove Worst Frame
                </button>
                <button className="btn btn-primary btn-sm" onClick={runCalibration}>
                  Re-calibrate
                </button>
              </div>
            </div>
          </div>

          {/* Quality assessment */}
          <div className="card" style={{ marginTop: 'var(--space-6)' }}>
            <div className="card-header">
              <h2 className="card-title">Calibration Quality Assessment</h2>
            </div>
            <div className={`alert ${calibResult.reprojError < 0.3 ? 'alert-success' : calibResult.reprojError < 0.8 ? 'alert-warning' : 'alert-danger'}`}>
              <span className="alert-icon">
                {calibResult.reprojError < 0.3 ? '🏆' : calibResult.reprojError < 0.8 ? '⚠️' : '❌'}
              </span>
              <div>
                <strong>
                  {calibResult.reprojError < 0.3
                    ? 'Excellent Calibration!'
                    : calibResult.reprojError < 0.8
                      ? 'Acceptable Calibration'
                      : 'Poor Calibration — Needs Improvement'
                  }
                </strong>
                <p style={{ marginTop: 'var(--space-2)' }}>
                  {calibResult.reprojError < 0.3
                    ? 'Your reprojection error is very low. This calibration should work well for most applications including visual odometry and 3D reconstruction.'
                    : calibResult.reprojError < 0.8
                      ? 'The calibration is usable but could be improved. Try removing high-error frames and re-calibrating, or capture more frames with better coverage.'
                      : 'The reprojection error is too high. Try: (1) removing blurry or poorly-detected frames, (2) using a different distortion model, (3) capturing more frames with better angular diversity.'
                  }
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ========================================
   Step 5: Export
   ======================================== */

function ExportStep({ config, frames, calibResult }) {
  const [exportFormat, setExportFormat] = useState('json');

  const downloadFile = (content, filename, mimeType) => {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadCalibration = () => {
    if (!calibResult) return;

    if (exportFormat === 'json') {
      const json = exportCalibrationJSON(calibResult, config);
      downloadFile(json, 'calibration.json', 'application/json');
    } else {
      const yaml = exportCalibrationYAML(calibResult, config);
      downloadFile(yaml, 'calibration.yaml', 'text/yaml');
    }
  };

  const downloadFrames = () => {
    const selectedFrames = frames.filter(f => f.selected);
    selectedFrames.forEach((frame, i) => {
      // Create full-res image from imageData
      const canvas = document.createElement('canvas');
      canvas.width = frame.width;
      canvas.height = frame.height;
      const ctx = canvas.getContext('2d');
      ctx.putImageData(frame.imageData, 0, 0);

      canvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `calibration_frame_${String(i + 1).padStart(3, '0')}.png`;
        a.click();
        URL.revokeObjectURL(url);
      }, 'image/png');

      canvas.remove();
    });
  };

  if (!calibResult) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📋</div>
        <h2 className="empty-state-title">No Calibration Data</h2>
        <p className="empty-state-desc">
          Complete the calibration step first before exporting results.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="section-header">
        <h1 className="section-title">Export Results</h1>
        <p className="section-desc">
          Download your calibration data and reference frames for future use.
        </p>
      </div>

      <div className="two-col">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Calibration Data</h2>
            <p className="card-subtitle">Camera intrinsics and distortion coefficients</p>
          </div>

          <div className="form-group">
            <label className="form-label">Format</label>
            <div className="tabs" style={{ maxWidth: 300 }}>
              <button
                className={`tab ${exportFormat === 'json' ? 'active' : ''}`}
                onClick={() => setExportFormat('json')}
              >
                JSON
              </button>
              <button
                className={`tab ${exportFormat === 'yaml' ? 'active' : ''}`}
                onClick={() => setExportFormat('yaml')}
              >
                OpenCV YAML
              </button>
            </div>
          </div>

          {/* Preview */}
          <pre className="matrix-display" style={{
            maxHeight: 300,
            overflowY: 'auto',
            fontSize: 'var(--font-xs)',
            whiteSpace: 'pre-wrap',
            wordBreak: 'break-all',
          }}>
            {exportFormat === 'json'
              ? exportCalibrationJSON(calibResult, config)
              : exportCalibrationYAML(calibResult, config)
            }
          </pre>

          <button
            className="btn btn-primary"
            style={{ marginTop: 'var(--space-4)', width: '100%' }}
            onClick={downloadCalibration}
          >
            💾 Download {exportFormat.toUpperCase()}
          </button>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Reference Frames</h2>
            <p className="card-subtitle">Save the selected frames for future reference or re-calibration</p>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 'var(--space-2)', marginBottom: 'var(--space-4)' }}>
            {frames.filter(f => f.selected).slice(0, 8).map((frame, i) => (
              <img
                key={i}
                src={frame.thumbnailUrl}
                alt={`Frame ${i + 1}`}
                style={{
                  width: 80,
                  height: 50,
                  objectFit: 'cover',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                }}
              />
            ))}
            {frames.filter(f => f.selected).length > 8 && (
              <div style={{
                width: 80, height: 50, borderRadius: 'var(--radius-sm)',
                background: 'var(--bg-tertiary)', display: 'flex',
                alignItems: 'center', justifyContent: 'center',
                fontSize: 'var(--font-xs)', color: 'var(--text-tertiary)',
              }}>
                +{frames.filter(f => f.selected).length - 8} more
              </div>
            )}
          </div>

          <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(2, 1fr)', marginBottom: 'var(--space-4)' }}>
            <div className="stat-card" style={{ padding: 'var(--space-3)' }}>
              <div className="stat-value" style={{ fontSize: 'var(--font-xl)' }}>
                {frames.filter(f => f.selected).length}
              </div>
              <div className="stat-label">Frames</div>
            </div>
            <div className="stat-card" style={{ padding: 'var(--space-3)' }}>
              <div className="stat-value" style={{ fontSize: 'var(--font-xl)', color: 'var(--accent-secondary)' }}>
                PNG
              </div>
              <div className="stat-label">Format</div>
            </div>
          </div>

          <button
            className="btn btn-secondary"
            style={{ width: '100%' }}
            onClick={downloadFrames}
          >
            📦 Download All Frames
          </button>
        </div>
      </div>

      {/* Summary card */}
      <div className="card" style={{ marginTop: 'var(--space-6)' }}>
        <div className="card-header">
          <h2 className="card-title">Calibration Summary</h2>
        </div>
        <div className="stats-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))' }}>
          <div>
            <div style={{ fontSize: 'var(--font-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
              Pattern
            </div>
            <div style={{ fontWeight: 600 }}>
              {config.patternType} ({config.rows}×{config.cols})
            </div>
          </div>
          <div>
            <div style={{ fontSize: 'var(--font-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
              Square Size
            </div>
            <div style={{ fontWeight: 600 }}>{config.squareSize} mm</div>
          </div>
          <div>
            <div style={{ fontSize: 'var(--font-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
              Distortion Model
            </div>
            <div style={{ fontWeight: 600 }}>{config.distortionModel}</div>
          </div>
          <div>
            <div style={{ fontSize: 'var(--font-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
              Resolution
            </div>
            <div style={{ fontWeight: 600 }}>
              {calibResult.imageSize.width}×{calibResult.imageSize.height}
            </div>
          </div>
          <div>
            <div style={{ fontSize: 'var(--font-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
              Reprojection Error
            </div>
            <div style={{
              fontWeight: 600,
              color: calibResult.reprojError < 0.3 ? 'var(--accent-success)' : calibResult.reprojError < 0.8 ? 'var(--accent-warning)' : 'var(--accent-danger)',
            }}>
              {calibResult.reprojError.toFixed(4)} px
            </div>
          </div>
          <div>
            <div style={{ fontSize: 'var(--font-xs)', color: 'var(--text-tertiary)', textTransform: 'uppercase', fontWeight: 600, marginBottom: 'var(--space-1)' }}>
              Frames Used
            </div>
            <div style={{ fontWeight: 600 }}>{calibResult.perFrameErrors.length}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
