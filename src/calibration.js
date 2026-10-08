/**
 * OpenCV.js loader and calibration engine.
 * All computer-vision logic lives here — no backend needed.
 */

const OPENCV_URL = 'https://docs.opencv.org/4.10.0/opencv.js';

let cvReady = false;
let cvLoadPromise = null;

/**
 * Load OpenCV.js once — returns a promise that resolves when cv is ready.
 */
export function loadOpenCV() {
  if (cvReady) return Promise.resolve(window.cv);
  if (cvLoadPromise) return cvLoadPromise;

  cvLoadPromise = new Promise((resolve, reject) => {
    // Check if already loaded
    if (window.cv && window.cv.Mat) {
      cvReady = true;
      resolve(window.cv);
      return;
    }

    const script = document.createElement('script');
    script.src = OPENCV_URL;
    script.async = true;

    // OpenCV.js uses a global onRuntimeInitialized callback
    window.Module = window.Module || {};
    const origOnReady = window.Module.onRuntimeInitialized;

    window.Module.onRuntimeInitialized = () => {
      if (origOnReady) origOnReady();
      cvReady = true;
      console.log('[Calibrator] OpenCV.js loaded successfully');
      resolve(window.cv);
    };

    script.onerror = () => {
      reject(new Error('Failed to load OpenCV.js'));
    };

    document.head.appendChild(script);
  });

  return cvLoadPromise;
}

/**
 * Check if a frame is blurry using the Laplacian variance method.
 * Returns the variance — higher = sharper.
 */
export function computeSharpness(cv, mat) {
  const gray = new cv.Mat();
  const laplacian = new cv.Mat();
  const mean = new cv.Mat();
  const stddev = new cv.Mat();

  try {
    if (mat.channels() > 1) {
      cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    } else {
      mat.copyTo(gray);
    }

    cv.Laplacian(gray, laplacian, cv.CV_64F);
    cv.meanStdDev(laplacian, mean, stddev);

    const variance = stddev.doubleAt(0, 0) ** 2;
    return variance;
  } finally {
    gray.delete();
    laplacian.delete();
    mean.delete();
    stddev.delete();
  }
}

/**
 * Detect checkerboard corners in a frame.
 * Returns { found, corners, mat } or { found: false }.
 */
export function detectCheckerboard(cv, mat, rows, cols) {
  const gray = new cv.Mat();
  const corners = new cv.Mat();

  try {
    if (mat.channels() > 1) {
      cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    } else {
      mat.copyTo(gray);
    }

    const patternSize = new cv.Size(cols, rows);
    const flags =
      cv.CALIB_CB_ADAPTIVE_THRESH |
      cv.CALIB_CB_NORMALIZE_IMAGE |
      cv.CALIB_CB_FAST_CHECK;

    const found = cv.findChessboardCorners(gray, patternSize, corners, flags);

    if (found) {
      // Sub-pixel refinement
      const criteria = new cv.TermCriteria(
        cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER,
        30,
        0.001
      );
      cv.cornerSubPix(gray, corners, new cv.Size(11, 11), new cv.Size(-1, -1), criteria);

      return {
        found: true,
        corners: corners.clone(),
        cornerCount: corners.rows,
      };
    }

    return { found: false };
  } finally {
    gray.delete();
    corners.delete();
  }
}

/**
 * Detect circles grid.
 */
export function detectCirclesGrid(cv, mat, rows, cols, asymmetric = false) {
  const gray = new cv.Mat();
  const centers = new cv.Mat();

  try {
    if (mat.channels() > 1) {
      cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
    } else {
      mat.copyTo(gray);
    }

    const patternSize = new cv.Size(cols, rows);
    const flags = asymmetric
      ? cv.CALIB_CB_ASYMMETRIC_GRID
      : cv.CALIB_CB_SYMMETRIC_GRID;

    const found = cv.findCirclesGrid(gray, patternSize, centers, flags);

    if (found) {
      return {
        found: true,
        corners: centers.clone(),
        cornerCount: centers.rows,
      };
    }

    return { found: false };
  } finally {
    gray.delete();
    centers.delete();
  }
}

/**
 * Draw detected corners/centers on a mat for visualization.
 */
export function drawDetection(cv, mat, corners, rows, cols, found) {
  const display = mat.clone();
  const patternSize = new cv.Size(cols, rows);
  cv.drawChessboardCorners(display, patternSize, corners, found);
  return display;
}

/**
 * Compute which coverage zone (grid cell) each corner falls into.
 * Returns a Map from "row,col" → count of corners in that zone.
 */
export function computeCoverage(corners, imageWidth, imageHeight, gridRows = 4, gridCols = 5) {
  const coverage = new Map();
  const cellW = imageWidth / gridCols;
  const cellH = imageHeight / gridRows;

  // Initialize all cells
  for (let r = 0; r < gridRows; r++) {
    for (let c = 0; c < gridCols; c++) {
      coverage.set(`${r},${c}`, 0);
    }
  }

  // Count corners per cell
  for (let i = 0; i < corners.rows; i++) {
    const x = corners.floatAt(i, 0);
    const y = corners.floatAt(i, 1);
    const gc = Math.min(Math.floor(x / cellW), gridCols - 1);
    const gr = Math.min(Math.floor(y / cellH), gridRows - 1);
    const key = `${gr},${gc}`;
    coverage.set(key, (coverage.get(key) || 0) + 1);
  }

  return coverage;
}

/**
 * Merge coverage maps from multiple frames.
 */
export function mergeCoverage(coverageMaps) {
  const merged = new Map();
  for (const cmap of coverageMaps) {
    for (const [key, count] of cmap) {
      merged.set(key, (merged.get(key) || 0) + count);
    }
  }
  return merged;
}

/**
 * Score a frame for diversity — how different its corners are from existing corners.
 * Returns a score between 0 and 1.
 */
export function computeFrameDiversity(existingCorners, newCorners, imageWidth, imageHeight) {
  if (existingCorners.length === 0) return 1.0;

  const gridSize = 10;
  const cellW = imageWidth / gridSize;
  const cellH = imageHeight / gridSize;

  // Build existing occupancy
  const occupied = new Set();
  for (const corners of existingCorners) {
    for (let i = 0; i < corners.rows; i++) {
      const x = corners.floatAt(i, 0);
      const y = corners.floatAt(i, 1);
      const gc = Math.floor(x / cellW);
      const gr = Math.floor(y / cellH);
      occupied.add(`${gr},${gc}`);
    }
  }

  // Count how many new corners land in unoccupied cells
  let newCells = 0;
  let totalNew = 0;
  const checked = new Set();
  for (let i = 0; i < newCorners.rows; i++) {
    const x = newCorners.floatAt(i, 0);
    const y = newCorners.floatAt(i, 1);
    const gc = Math.floor(x / cellW);
    const gr = Math.floor(y / cellH);
    const key = `${gr},${gc}`;
    if (!checked.has(key)) {
      checked.add(key);
      totalNew++;
      if (!occupied.has(key)) newCells++;
    }
  }

  return totalNew > 0 ? newCells / totalNew : 0;
}

/**
 * Generate 3D object points for the calibration pattern.
 */
function generateObjectPoints(cv, rows, cols, squareSize, patternType = 'checkerboard') {
  const objPoints = [];

  if (patternType === 'circles_asymmetric') {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        objPoints.push((2 * c + r % 2) * squareSize, r * squareSize, 0);
      }
    }
  } else {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        objPoints.push(c * squareSize, r * squareSize, 0);
      }
    }
  }

  return objPoints;
}

/**
 * Run camera calibration on the selected frames.
 * Returns { cameraMatrix, distCoeffs, rvecs, tvecs, reprojError, perFrameErrors }.
 */
export function calibrateCamera(cv, frames, config) {
  const { rows, cols, squareSize, patternType, distortionModel } = config;
  const numPoints = rows * cols;
  const objPointsFlat = generateObjectPoints(cv, rows, cols, squareSize, patternType);

  // Build object points and image points vectors
  const objectPointsVec = new cv.MatVector();
  const imagePointsVec = new cv.MatVector();

  const imageSize = frames.length > 0
    ? new cv.Size(frames[0].width, frames[0].height)
    : new cv.Size(640, 480);

  for (const frame of frames) {
    // Object points for this frame
    const objPts = cv.matFromArray(numPoints, 1, cv.CV_32FC3, objPointsFlat);
    objectPointsVec.push_back(objPts);
    objPts.delete();

    // Image points for this frame
    const imgPts = frame.corners.clone();
    imagePointsVec.push_back(imgPts);
    imgPts.delete();
  }

  const cameraMatrix = new cv.Mat();
  const distCoeffs = new cv.Mat();
  const rvecs = new cv.MatVector();
  const tvecs = new cv.MatVector();

  // Select calibration flags based on distortion model
  let flags = 0;
  if (distortionModel === 'rational') {
    flags = cv.CALIB_RATIONAL_MODEL;
  } else if (distortionModel === 'thin_prism') {
    flags = cv.CALIB_RATIONAL_MODEL | cv.CALIB_THIN_PRISM_MODEL;
  } else if (distortionModel === 'fixed_5') {
    flags = 0; // default 5-param
  }

  const reprojError = cv.calibrateCamera(
    objectPointsVec,
    imagePointsVec,
    imageSize,
    cameraMatrix,
    distCoeffs,
    rvecs,
    tvecs,
    flags
  );

  // Compute per-frame reprojection errors
  const perFrameErrors = [];
  for (let i = 0; i < frames.length; i++) {
    const projectedPoints = new cv.Mat();
    const rvec = rvecs.get(i);
    const tvec = tvecs.get(i);

    cv.projectPoints(
      cv.matFromArray(numPoints, 1, cv.CV_32FC3, objPointsFlat),
      rvec,
      tvec,
      cameraMatrix,
      distCoeffs,
      projectedPoints
    );

    // Compute error for this frame
    let totalErr = 0;
    const imgPts = frames[i].corners;
    for (let j = 0; j < numPoints; j++) {
      const dx = projectedPoints.floatAt(j, 0) - imgPts.floatAt(j, 0);
      const dy = projectedPoints.floatAt(j, 1) - imgPts.floatAt(j, 1);
      totalErr += Math.sqrt(dx * dx + dy * dy);
    }

    perFrameErrors.push({
      frameIndex: i,
      error: totalErr / numPoints,
    });

    projectedPoints.delete();
    rvec.delete();
    tvec.delete();
  }

  // Extract results before cleanup
  const result = {
    cameraMatrix: {
      fx: cameraMatrix.doubleAt(0, 0),
      fy: cameraMatrix.doubleAt(1, 1),
      cx: cameraMatrix.doubleAt(0, 2),
      cy: cameraMatrix.doubleAt(1, 2),
      raw: matToArray2D(cameraMatrix),
    },
    distCoeffs: {
      values: matToArray1D(distCoeffs),
      raw: matToArray1D(distCoeffs),
    },
    reprojError,
    perFrameErrors,
    imageSize: { width: imageSize.width, height: imageSize.height },
  };

  // Cleanup
  objectPointsVec.delete();
  imagePointsVec.delete();
  cameraMatrix.delete();
  distCoeffs.delete();
  rvecs.delete();
  tvecs.delete();

  return result;
}

/**
 * Convert cv.Mat to 2D array.
 */
function matToArray2D(mat) {
  const result = [];
  for (let r = 0; r < mat.rows; r++) {
    const row = [];
    for (let c = 0; c < mat.cols; c++) {
      row.push(mat.doubleAt(r, c));
    }
    result.push(row);
  }
  return result;
}

/**
 * Convert cv.Mat to 1D array.
 */
function matToArray1D(mat) {
  const result = [];
  const len = Math.max(mat.rows, mat.cols);
  for (let i = 0; i < len; i++) {
    result.push(mat.doubleAt(0, i));
  }
  return result;
}

/**
 * Extract frames from a video file.
 * Yields frames at a specified interval, skipping blurry ones.
 * Returns array of { imageData, timestamp, sharpness }.
 */
export async function extractFramesFromVideo(videoFile, options = {}) {
  const {
    maxFrames = 100,
    frameInterval = 0.5,  // seconds between frame samples
    sharpnessThreshold = 50,
    onProgress = () => {},
  } = options;

  return new Promise((resolve, reject) => {
    const video = document.createElement('video');
    video.preload = 'auto';
    video.muted = true;
    video.playsInline = true;

    const url = URL.createObjectURL(videoFile);
    video.src = url;

    video.onloadedmetadata = () => {
      const duration = video.duration;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');

      const frames = [];
      const times = [];

      for (let t = 0; t < duration && times.length < maxFrames * 3; t += frameInterval) {
        times.push(t);
      }

      let idx = 0;

      const seekNext = () => {
        if (idx >= times.length || frames.length >= maxFrames) {
          URL.revokeObjectURL(url);
          video.remove();
          canvas.remove();
          resolve(frames);
          return;
        }

        video.currentTime = times[idx];
      };

      video.onseeked = () => {
        ctx.drawImage(video, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        frames.push({
          imageData,
          timestamp: times[idx],
          width: canvas.width,
          height: canvas.height,
        });

        idx++;
        onProgress(idx / times.length);
        seekNext();
      };

      video.onerror = reject;
      seekNext();
    };

    video.onerror = reject;
  });
}

/**
 * Process extracted frames: detect pattern, compute sharpness, filter.
 */
export function processFrames(cv, rawFrames, config, onProgress = () => {}) {
  const { rows, cols, patternType, sharpnessThreshold = 50 } = config;
  const processed = [];

  for (let i = 0; i < rawFrames.length; i++) {
    onProgress(i / rawFrames.length, `Analyzing frame ${i + 1}/${rawFrames.length}`);

    const frame = rawFrames[i];
    const mat = cv.matFromImageData(frame.imageData);

    // Check sharpness
    const sharpness = computeSharpness(cv, mat);

    if (sharpness < sharpnessThreshold) {
      mat.delete();
      continue;
    }

    // Detect pattern
    let detection;
    if (patternType === 'checkerboard') {
      detection = detectCheckerboard(cv, mat, rows, cols);
    } else if (patternType === 'circles' || patternType === 'circles_asymmetric') {
      detection = detectCirclesGrid(cv, mat, rows, cols, patternType === 'circles_asymmetric');
    }

    if (detection && detection.found) {
      // Compute coverage for this frame
      const coverage = computeCoverage(detection.corners, frame.width, frame.height);

      // Create a thumbnail
      const thumbCanvas = document.createElement('canvas');
      const scale = 240 / frame.width;
      thumbCanvas.width = 240;
      thumbCanvas.height = Math.round(frame.height * scale);
      const thumbCtx = thumbCanvas.getContext('2d');

      // Draw frame to thumbnail
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = frame.width;
      tempCanvas.height = frame.height;
      const tempCtx = tempCanvas.getContext('2d');
      tempCtx.putImageData(frame.imageData, 0, 0);
      thumbCtx.drawImage(tempCanvas, 0, 0, thumbCanvas.width, thumbCanvas.height);
      tempCanvas.remove();

      const thumbnailUrl = thumbCanvas.toDataURL('image/jpeg', 0.7);
      thumbCanvas.remove();

      processed.push({
        index: i,
        timestamp: frame.timestamp,
        width: frame.width,
        height: frame.height,
        sharpness,
        corners: detection.corners,
        cornerCount: detection.cornerCount,
        coverage,
        thumbnailUrl,
        imageData: frame.imageData,
        selected: true,
      });
    }

    mat.delete();
  }

  return processed;
}

/**
 * Select optimal frames from processed frames for best coverage + diversity.
 * Uses a greedy selection algorithm.
 */
export function selectOptimalFrames(processedFrames, targetCount = 20) {
  if (processedFrames.length <= targetCount) {
    return processedFrames.map((_, i) => i);
  }

  const selected = [];
  const available = processedFrames.map((_, i) => i);

  // Always include the frame with highest sharpness first
  available.sort((a, b) => processedFrames[b].sharpness - processedFrames[a].sharpness);
  selected.push(available.shift());

  while (selected.length < targetCount && available.length > 0) {
    let bestIdx = -1;
    let bestScore = -1;

    for (const idx of available) {
      const frame = processedFrames[idx];
      const existingCorners = selected.map(si => processedFrames[si].corners);
      const diversity = computeFrameDiversity(
        existingCorners,
        frame.corners,
        frame.width,
        frame.height
      );
      const sharpnessScore = Math.min(frame.sharpness / 500, 1);
      const score = diversity * 0.7 + sharpnessScore * 0.3;

      if (score > bestScore) {
        bestScore = score;
        bestIdx = idx;
      }
    }

    if (bestIdx >= 0) {
      selected.push(bestIdx);
      available.splice(available.indexOf(bestIdx), 1);
    } else {
      break;
    }
  }

  return selected;
}

/**
 * Export calibration result as JSON.
 */
export function exportCalibrationJSON(result, config) {
  const data = {
    calibration: {
      camera_matrix: result.cameraMatrix.raw,
      distortion_coefficients: result.distCoeffs.values,
      image_size: result.imageSize,
      reprojection_error: result.reprojError,
    },
    config: {
      pattern_type: config.patternType,
      pattern_size: { rows: config.rows, cols: config.cols },
      square_size_mm: config.squareSize,
      distortion_model: config.distortionModel,
      num_frames_used: result.perFrameErrors.length,
    },
    timestamp: new Date().toISOString(),
  };

  return JSON.stringify(data, null, 2);
}

/**
 * Export as OpenCV YAML format.
 */
export function exportCalibrationYAML(result, config) {
  const cm = result.cameraMatrix;
  const dc = result.distCoeffs.values;

  let yaml = `%YAML:1.0\n---\n`;
  yaml += `calibration_time: "${new Date().toISOString()}"\n`;
  yaml += `image_width: ${result.imageSize.width}\n`;
  yaml += `image_height: ${result.imageSize.height}\n`;
  yaml += `pattern_type: "${config.patternType}"\n`;
  yaml += `board_width: ${config.cols}\n`;
  yaml += `board_height: ${config.rows}\n`;
  yaml += `square_size: ${config.squareSize}\n`;
  yaml += `distortion_model: "${config.distortionModel}"\n`;
  yaml += `avg_reprojection_error: ${result.reprojError.toFixed(6)}\n`;
  yaml += `camera_matrix: !!opencv-matrix\n`;
  yaml += `   rows: 3\n   cols: 3\n   dt: d\n`;
  yaml += `   data: [ ${cm.raw.flat().map(v => v.toFixed(8)).join(', ')} ]\n`;
  yaml += `distortion_coefficients: !!opencv-matrix\n`;
  yaml += `   rows: 1\n   cols: ${dc.length}\n   dt: d\n`;
  yaml += `   data: [ ${dc.map(v => v.toFixed(8)).join(', ')} ]\n`;

  return yaml;
}

/**
 * Create a ZIP of selected frames and calibration data.
 */
export async function exportSessionZip(frames, calibResult, config) {
  // We'll use a simple approach — create download links
  const files = [];

  // Calibration JSON
  files.push({
    name: 'calibration.json',
    content: exportCalibrationJSON(calibResult, config),
    type: 'application/json',
  });

  // Calibration YAML
  files.push({
    name: 'calibration.yaml',
    content: exportCalibrationYAML(calibResult, config),
    type: 'text/yaml',
  });

  return files;
}
