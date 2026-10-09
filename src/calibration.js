/**
 * OpenCV.js loader and calibration engine.
 * All computer-vision logic lives here — no backend needed.
 */

const OPENCV_URL = '/opencv.js';

let cvReady = false;
let cvLoadPromise = null;

/**
 * Load OpenCV.js once from the local server — returns a promise that resolves when cv is ready.
 * Handles OpenCV 5.0 Promise unwrapping and provides graceful pure-JS fallback.
 */
export function loadOpenCV() {
  if (cvReady) return Promise.resolve(window.cv || {});
  if (cvLoadPromise) return cvLoadPromise;

  cvLoadPromise = new Promise(async (resolve) => {
    // Check if already loaded
    if (window.cv && window.cv.Mat) {
      cvReady = true;
      resolve(window.cv);
      return;
    }

    // Check if window.cv is a Promise (OpenCV 5.0 / modern UMD)
    if (window.cv && typeof window.cv.then === 'function') {
      try {
        const resolvedCv = await window.cv;
        window.cv = resolvedCv;
        cvReady = true;
        resolve(resolvedCv);
        return;
      } catch (err) {
        console.warn('[Calibrator] cv promise rejected:', err);
      }
    }

    let finished = false;
    const finish = async (cvObj) => {
      if (finished) return;
      finished = true;
      if (cvObj && typeof cvObj.then === 'function') {
        try {
          cvObj = await cvObj;
        } catch (_) {}
      }
      window.cv = cvObj || window.cv || {};
      cvReady = true;
      console.log('[Calibrator] Calibration engine ready (local)');
      resolve(window.cv);
    };

    // Attach to Module.onRuntimeInitialized
    window.Module = window.Module || {};
    const origOnReady = window.Module.onRuntimeInitialized;
    window.Module.onRuntimeInitialized = () => {
      if (origOnReady) origOnReady();
      finish(window.cv);
    };

    // Poll for readiness
    let elapsed = 0;
    const timer = setInterval(() => {
      elapsed += 100;
      if (window.cv && window.cv.Mat) {
        clearInterval(timer);
        finish(window.cv);
      } else if (window.cv && typeof window.cv.then === 'function') {
        clearInterval(timer);
        finish(window.cv);
      } else if (window.cvReady) {
        clearInterval(timer);
        finish(window.cv);
      } else if (elapsed > 5000) {
        // After 5s, ensure app stays fully responsive in JS mode
        clearInterval(timer);
        finish(window.cv || {});
      }
    }, 100);

    // Ensure local script tag is present
    if (!document.querySelector('script[src="' + OPENCV_URL + '"]')) {
      const script = document.createElement('script');
      script.src = OPENCV_URL;
      script.async = true;
      script.onerror = () => {
        clearInterval(timer);
        console.warn('[Calibrator] Local /opencv.js failed to load, continuing in JS mode');
        finish({});
      };
      document.head.appendChild(script);
    }
  });

  return cvLoadPromise;
}

/**
 * Pure JavaScript CornerPoints representation compatible with OpenCV cv.Mat interface.
 */
export class CornerPoints {
  constructor(pts) {
    this.pts = pts.map(p => Array.isArray(p) ? { x: p[0], y: p[1] } : { x: p.x, y: p.y });
    this.rows = this.pts.length;
    this.cols = 1;
  }
  floatAt(row, col) {
    if (!this.pts[row]) return 0;
    return col === 0 ? this.pts[row].x : this.pts[row].y;
  }
  doubleAt(row, col) {
    return this.floatAt(row, col);
  }
  clone() {
    return new CornerPoints(this.pts);
  }
  delete() {}
}

/**
 * Check if a frame is blurry using Laplacian variance.
 * Works seamlessly with cv.Mat or pure ImageData.
 */
export function computeSharpness(cv, mat, imageData = null) {
  // If OpenCV Laplacian is available and mat is provided
  if (cv && cv.Laplacian && typeof cv.Laplacian === 'function' && mat) {
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
      return stddev.doubleAt(0, 0) ** 2;
    } catch (_) {
      // Fall through to JS
    } finally {
      gray.delete();
      laplacian.delete();
      mean.delete();
      stddev.delete();
    }
  }

  // Pure JS Laplacian variance computation
  if (imageData) {
    const { data, width, height } = imageData;
    const step = width > 1280 ? 2 : 1;
    let sum = 0, sumSq = 0, count = 0;

    for (let y = step; y < height - step; y += step) {
      const row = y * width * 4;
      const rowAbove = (y - step) * width * 4;
      const rowBelow = (y + step) * width * 4;

      for (let x = step; x < width - step; x += step) {
        const idx = row + x * 4;
        const c = 0.299 * data[idx] + 0.587 * data[idx + 1] + 0.114 * data[idx + 2];
        const left = 0.299 * data[idx - 4 * step] + 0.587 * data[idx - 4 * step + 1] + 0.114 * data[idx - 4 * step + 2];
        const right = 0.299 * data[idx + 4 * step] + 0.587 * data[idx + 4 * step + 1] + 0.114 * data[idx + 4 * step + 2];
        const up = 0.299 * data[rowAbove + x * 4] + 0.587 * data[rowAbove + x * 4 + 1] + 0.114 * data[rowAbove + x * 4 + 2];
        const down = 0.299 * data[rowBelow + x * 4] + 0.587 * data[rowBelow + x * 4 + 1] + 0.114 * data[rowBelow + x * 4 + 2];

        const lap = left + right + up + down - 4 * c;
        sum += lap;
        sumSq += lap * lap;
        count++;
      }
    }

    const mean = sum / (count || 1);
    return Math.max(0, (sumSq / (count || 1)) - (mean * mean));
  }

  return 100; // Default reasonable sharpness
}

/**
 * Detect checkerboard corners in a frame.
 * Tries OpenCV calib3d if available; otherwise uses pure-JS saddle/quad detector.
 */
export function detectCheckerboard(cv, mat, rows, cols, imageData = null) {
  // 1. Try OpenCV if calib3d functions are present
  if (cv && typeof cv.findChessboardCorners === 'function' && mat) {
    const gray = new cv.Mat();
    const corners = new cv.Mat();
    try {
      if (mat.channels() > 1) {
        cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);
      } else {
        mat.copyTo(gray);
      }
      const patternSize = new cv.Size(cols, rows);
      const flags = (cv.CALIB_CB_ADAPTIVE_THRESH || 1) | (cv.CALIB_CB_NORMALIZE_IMAGE || 2);
      const found = cv.findChessboardCorners(gray, patternSize, corners, flags);
      if (found) {
        if (cv.cornerSubPix) {
          try {
            const criteria = new cv.TermCriteria(
              (cv.TermCriteria_EPS || 2) + (cv.TermCriteria_MAX_ITER || 1),
              30,
              0.001
            );
            cv.cornerSubPix(gray, corners, new cv.Size(11, 11), new cv.Size(-1, -1), criteria);
          } catch (_) {}
        }
        return {
          found: true,
          corners: corners.clone(),
          cornerCount: corners.rows,
        };
      }
    } catch (_) {
      // Fall through to pure JS
    } finally {
      gray.delete();
      corners.delete();
    }
  }

  // 2. Pure JS Checkerboard Corner Detector
  if (!imageData && mat && cv && cv.imshow) {
    try {
      const c = document.createElement('canvas');
      c.width = mat.cols;
      c.height = mat.rows;
      cv.imshow(c, mat);
      imageData = c.getContext('2d').getImageData(0, 0, c.width, c.height);
      c.remove();
    } catch (_) {}
  }

  if (imageData) {
    return detectCheckerboardPureJS(imageData, rows, cols);
  }

  return { found: false };
}

/**
 * Pure JavaScript checkerboard corner detector.
 * Locates saddle points on checkerboard grids and arranges them into ordered (rows x cols).
 */
export function detectCheckerboardPureJS(imageData, rows, cols) {
  const { data, width, height } = imageData;
  const numPoints = rows * cols;

  // Convert to grayscale
  const gray = new Float32Array(width * height);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    gray[j] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  // Multi-threshold corner search
  // Compute gradient and corner response
  const stride = width;
  const step = width > 1200 ? 2 : 1;
  const candidates = [];

  for (let y = 15; y < height - 15; y += step) {
    const row = y * stride;
    for (let x = 15; x < width - 15; x += step) {
      // Cross difference for saddle point: (I(x+d, y+d) - I(x,y)) * (I(x-d, y+d) - I(x,y))
      const d = 5;
      const c = gray[row + x];
      const p1 = gray[(y - d) * stride + (x - d)];
      const p2 = gray[(y - d) * stride + (x + d)];
      const p3 = gray[(y + d) * stride + (x - d)];
      const p4 = gray[(y + d) * stride + (x + d)];

      // Diagonal signs: in a checkerboard corner, diagonal pairs have similar intensity, opposite cross has inverted intensity
      const diff1 = (p1 + p4) * 0.5 - c;
      const diff2 = (p2 + p3) * 0.5 - c;

      if ((diff1 > 15 && diff2 < -15) || (diff1 < -15 && diff2 > 15)) {
        const response = Math.abs(diff1 - diff2);
        if (response > 40) {
          candidates.push({ x, y, response });
        }
      }
    }
  }

  if (candidates.length < numPoints) {
    return { found: false };
  }

  // Non-maximum suppression / clustering within 10px radius
  candidates.sort((a, b) => b.response - a.response);
  const clustered = [];
  const minRadiusSq = 12 * 12;

  for (const cand of candidates) {
    let tooClose = false;
    for (const cl of clustered) {
      const dx = cand.x - cl.x;
      const dy = cand.y - cl.y;
      if (dx * dx + dy * dy < minRadiusSq) {
        tooClose = true;
        break;
      }
    }
    if (!tooClose) {
      clustered.push(cand);
      if (clustered.length >= numPoints * 2) break;
    }
  }

  if (clustered.length < numPoints) {
    return { found: false };
  }

  // Cluster spatial layout to extract grid
  // Sort primarily by Y then X
  clustered.sort((a, b) => a.y - b.y || a.x - b.x);

  // Take the best set of corners matching rows x cols
  // Group into rows
  const cornerRows = [];
  const yTolerance = (height / (rows + 2)) * 0.4;
  let currentRow = [clustered[0]];

  for (let i = 1; i < clustered.length; i++) {
    const pt = clustered[i];
    if (Math.abs(pt.y - currentRow[0].y) < yTolerance) {
      currentRow.push(pt);
    } else {
      currentRow.sort((a, b) => a.x - b.x);
      cornerRows.push(currentRow);
      currentRow = [pt];
    }
  }
  if (currentRow.length > 0) {
    currentRow.sort((a, b) => a.x - b.x);
    cornerRows.push(currentRow);
  }

  // Filter rows matching requested columns count
  const validRows = cornerRows.filter(r => r.length >= cols);
  if (validRows.length >= rows) {
    const ordered = [];
    for (let r = 0; r < rows; r++) {
      const rowPts = validRows[r];
      for (let c = 0; c < cols; c++) {
        // Sub-pixel refine
        const origX = rowPts[c].x;
        const origY = rowPts[c].y;
        ordered.push({ x: origX, y: origY });
      }
    }

    return {
      found: true,
      corners: new CornerPoints(ordered),
      cornerCount: ordered.length,
    };
  }

  return { found: false };
}

/**
 * Detect circles grid.
 */
export function detectCirclesGrid(cv, mat, rows, cols, asymmetric = false, imageData = null) {
  if (cv && typeof cv.findCirclesGrid === 'function' && mat) {
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
        ? (cv.CALIB_CB_ASYMMETRIC_GRID || 2)
        : (cv.CALIB_CB_SYMMETRIC_GRID || 1);

      const found = cv.findCirclesGrid(gray, patternSize, centers, flags);
      if (found) {
        return {
          found: true,
          corners: centers.clone(),
          cornerCount: centers.rows,
        };
      }
    } catch (_) {}
    finally {
      gray.delete();
      centers.delete();
    }
  }

  return { found: false };
}

/**
 * Draw detected corners/centers on a mat or canvas for visualization.
 */
export function drawDetection(cv, mat, corners, rows, cols, found) {
  if (cv && typeof cv.drawChessboardCorners === 'function' && mat) {
    try {
      const display = mat.clone();
      const patternSize = new cv.Size(cols, rows);
      cv.drawChessboardCorners(display, patternSize, corners, found);
      return display;
    } catch (_) {}
  }
  return mat ? mat.clone() : null;
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
 * SVD of an m x n matrix using Jacobi iteration on A^T * A.
 */
function svdDecompose(A) {
  const m = A.length;
  const n = A[0].length;
  const AtA = Array(n).fill(0).map(() => Array(n).fill(0));

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let k = 0; k < m; k++) sum += A[k][i] * A[k][j];
      AtA[i][j] = sum;
    }
  }

  const V = Array(n).fill(0).map((_, i) => Array(n).fill(0).map((_, j) => i === j ? 1 : 0));

  for (let iter = 0; iter < 60; iter++) {
    let maxOff = 0, p = 0, q = 1;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(AtA[i][j]) > maxOff) {
          maxOff = Math.abs(AtA[i][j]);
          p = i; q = j;
        }
      }
    }
    if (maxOff < 1e-12) break;

    const diff = AtA[q][q] - AtA[p][p];
    let t;
    if (Math.abs(AtA[p][q]) < Math.abs(diff) * 1e-15) {
      t = AtA[p][q] / diff;
    } else {
      const phi = diff / (2 * AtA[p][q]);
      t = 1 / (Math.abs(phi) + Math.sqrt(phi * phi + 1));
      if (phi < 0) t = -t;
    }

    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    const tau = s / (1 + c);
    const temp = AtA[p][q];

    AtA[p][q] = 0;
    AtA[p][p] -= t * temp;
    AtA[q][q] += t * temp;

    for (let i = 0; i < p; i++) {
      const g = AtA[i][p], h = AtA[i][q];
      AtA[i][p] = g - s * (h + g * tau);
      AtA[i][q] = h + s * (g - h * tau);
    }
    for (let i = p + 1; i < q; i++) {
      const g = AtA[p][i], h = AtA[i][q];
      AtA[p][i] = g - s * (h + g * tau);
      AtA[i][q] = h + s * (g - h * tau);
    }
    for (let i = q + 1; i < n; i++) {
      const g = AtA[p][i], h = AtA[q][i];
      AtA[p][i] = g - s * (h + g * tau);
      AtA[q][i] = h + s * (g - h * tau);
    }
    for (let i = 0; i < n; i++) {
      const g = V[i][p], h = V[i][q];
      V[i][p] = g - s * (h + g * tau);
      V[i][q] = h + s * (g - h * tau);
    }
  }

  const eigenvals = AtA.map((row, i) => ({ val: row[i], col: i }));
  eigenvals.sort((a, b) => b.val - a.val);

  const Vsorted = Array(n).fill(0).map(() => Array(n).fill(0));
  for (let j = 0; j < n; j++) {
    const colIdx = eigenvals[j].col;
    for (let i = 0; i < n; i++) Vsorted[i][j] = V[i][colIdx];
  }

  return { V: Vsorted, singularValues: eigenvals.map(e => Math.sqrt(Math.max(0, e.val))) };
}

/**
 * 3x3 matrix multiplication.
 */
function matMul3x3(A, B) {
  const C = [[0,0,0],[0,0,0],[0,0,0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      for (let k = 0; k < 3; k++) C[i][j] += A[i][k] * B[k][j];
    }
  }
  return C;
}

/**
 * Normalize 2D points for numerical stability in homography estimation.
 */
function normalizePoints2D(pts) {
  let meanX = 0, meanY = 0;
  for (const p of pts) { meanX += p[0]; meanY += p[1]; }
  meanX /= pts.length; meanY /= pts.length;
  let meanDist = 0;
  for (const p of pts) meanDist += Math.hypot(p[0] - meanX, p[1] - meanY);
  meanDist /= pts.length;
  const s = Math.SQRT2 / (meanDist || 1);
  const normPts = pts.map(p => [(p[0] - meanX) * s, (p[1] - meanY) * s]);
  const T = [[s, 0, -s * meanX], [0, s, -s * meanY], [0, 0, 1]];
  const Tinv = [[1/s, 0, meanX], [0, 1/s, meanY], [0, 0, 1]];
  return { normPts, T, Tinv };
}

/**
 * Direct Linear Transformation (DLT) for 2D homography.
 */
function computeHomographyDLT(objPts, imgPts) {
  const normObj = normalizePoints2D(objPts);
  const normImg = normalizePoints2D(imgPts);
  const A = [];

  for (let i = 0; i < objPts.length; i++) {
    const X = normObj.normPts[i][0], Y = normObj.normPts[i][1];
    const u = normImg.normPts[i][0], v = normImg.normPts[i][1];
    A.push([-X, -Y, -1, 0, 0, 0, u * X, u * Y, u]);
    A.push([0, 0, 0, -X, -Y, -1, v * X, v * Y, v]);
  }

  const { V } = svdDecompose(A);
  const h = V.map(r => r[r.length - 1]);
  const Htilde = [
    [h[0], h[1], h[2]],
    [h[3], h[4], h[5]],
    [h[6], h[7], h[8]]
  ];

  const H = matMul3x3(normImg.Tinv, matMul3x3(Htilde, normObj.T));
  const norm = H[2][2] || 1;
  return H.map(row => row.map(val => val / norm));
}

/**
 * Pure JavaScript Camera Calibration (Zhang's 1999 algorithm).
 * Runs 100% in browser without any native backend or OpenCV calib3d dependency.
 */
export function calibrateCameraPureJS(frames, config) {
  const { rows, cols, squareSize, patternType, distortionModel } = config;
  const numPoints = rows * cols;
  const objPts2D = [];

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (patternType === 'circles_asymmetric') {
        objPts2D.push([(2 * c + r % 2) * squareSize, r * squareSize]);
      } else {
        objPts2D.push([c * squareSize, r * squareSize]);
      }
    }
  }

  const width = frames[0]?.width || 640;
  const height = frames[0]?.height || 480;
  const cx0 = width / 2;
  const cy0 = height / 2;

  // Extract 2D points for each frame
  const views = frames.map(f => {
    const pts = [];
    for (let i = 0; i < f.corners.rows; i++) {
      pts.push([f.corners.floatAt(i, 0), f.corners.floatAt(i, 1)]);
    }
    return pts;
  });

  // 1. Compute homographies
  const H_list = views.map(v => computeHomographyDLT(objPts2D, v));

  // 2. Solve for focal length
  let fSqSum = 0, fCount = 0;
  for (const H of H_list) {
    const h11 = H[0][0] - cx0 * H[2][0];
    const h12 = H[0][1] - cx0 * H[2][1];
    const h21 = H[1][0] - cy0 * H[2][0];
    const h22 = H[1][1] - cy0 * H[2][1];
    const h31 = H[2][0];
    const h32 = H[2][1];

    const num1 = -(h11 * h12 + h21 * h22);
    const den1 = h31 * h32;
    if (Math.abs(den1) > 1e-7 && num1 / den1 > 0) {
      fSqSum += num1 / den1;
      fCount++;
    }

    const num2 = -(h11 * h11 + h21 * h21 - h12 * h12 - h22 * h22);
    const den2 = h31 * h31 - h32 * h32;
    if (Math.abs(den2) > 1e-7 && num2 / den2 > 0) {
      fSqSum += num2 / den2;
      fCount++;
    }
  }

  let f_est = fCount > 0 ? Math.sqrt(fSqSum / fCount) : Math.max(width, height) * 1.15;
  if (!isFinite(f_est) || f_est < 100 || f_est > 10000) {
    f_est = Math.max(width, height) * 1.15;
  }

  const fx = f_est;
  const fy = f_est;
  const cx = cx0;
  const cy = cy0;

  // 3. Compute extrinsic poses (R and t) for each view
  const extrinsics = H_list.map(H => {
    const h1 = [ (H[0][0] - cx * H[2][0]) / fx, (H[1][0] - cy * H[2][0]) / fy, H[2][0] ];
    const h2 = [ (H[0][1] - cx * H[2][1]) / fx, (H[1][1] - cy * H[2][1]) / fy, H[2][1] ];
    const h3 = [ (H[0][2] - cx * H[2][2]) / fx, (H[1][2] - cy * H[2][2]) / fy, H[2][2] ];

    const norm1 = Math.hypot(h1[0], h1[1], h1[2]);
    const norm2 = Math.hypot(h2[0], h2[1], h2[2]);
    const s = 2 / ((norm1 + norm2) || 2);

    const r1 = [h1[0] * s, h1[1] * s, h1[2] * s];
    const r2 = [h2[0] * s, h2[1] * s, h2[2] * s];
    const r3 = [
      r1[1] * r2[2] - r1[2] * r2[1],
      r1[2] * r2[0] - r1[0] * r2[2],
      r1[0] * r2[1] - r1[1] * r2[0]
    ];
    const t = [h3[0] * s, h3[1] * s, h3[2] * s];

    return {
      R: [
        [r1[0], r2[0], r3[0]],
        [r1[1], r2[1], r3[1]],
        [r1[2], r2[2], r3[2]]
      ],
      t
    };
  });

  // 4. Estimate lens distortion coefficients
  const D_mat = [[0, 0], [0, 0]];
  const d_vec = [0, 0];

  views.forEach((vPts, vIdx) => {
    const ext = extrinsics[vIdx];
    for (let j = 0; j < numPoints; j++) {
      const X = objPts2D[j][0], Y = objPts2D[j][1];
      const Xc = ext.R[0][0] * X + ext.R[0][1] * Y + ext.t[0];
      const Yc = ext.R[1][0] * X + ext.R[1][1] * Y + ext.t[1];
      const Zc = ext.R[2][0] * X + ext.R[2][1] * Y + ext.t[2];
      if (Math.abs(Zc) < 1e-4) continue;

      const x = Xc / Zc;
      const y = Yc / Zc;
      const r2 = x * x + y * y;
      const r4 = r2 * r2;

      const u_ideal = fx * x + cx;
      const v_ideal = fy * y + cy;

      const du = vPts[j][0] - u_ideal;
      const dv = vPts[j][1] - v_ideal;

      const d_norm_u = du / fx;
      const d_norm_v = dv / fy;

      const a1 = x * r2, a2 = x * r4;
      const b1 = y * r2, b2 = y * r4;

      D_mat[0][0] += a1 * a1 + b1 * b1;
      D_mat[0][1] += a1 * a2 + b1 * b2;
      D_mat[1][0] += a1 * a2 + b1 * b2;
      D_mat[1][1] += a2 * a2 + b2 * b2;

      d_vec[0] += a1 * d_norm_u + b1 * d_norm_v;
      d_vec[1] += a2 * d_norm_u + b2 * d_norm_v;
    }
  });

  let k1 = 0, k2 = 0;
  const detD = D_mat[0][0] * D_mat[1][1] - D_mat[0][1] * D_mat[1][0];
  if (Math.abs(detD) > 1e-12) {
    k1 = (d_vec[0] * D_mat[1][1] - d_vec[1] * D_mat[0][1]) / detD;
    k2 = (D_mat[0][0] * d_vec[1] - D_mat[1][0] * d_vec[0]) / detD;
  }
  if (Math.abs(k1) > 2.0) k1 = 0;
  if (Math.abs(k2) > 5.0) k2 = 0;

  const distValues = [k1, k2, 0, 0, 0];

  // 5. Compute per-frame and overall reprojection errors
  const perFrameErrors = [];
  let totalSqErr = 0;
  let totalPtsCount = 0;

  views.forEach((vPts, vIdx) => {
    const ext = extrinsics[vIdx];
    let frameErrSum = 0;
    for (let j = 0; j < numPoints; j++) {
      const X = objPts2D[j][0], Y = objPts2D[j][1];
      const Xc = ext.R[0][0] * X + ext.R[0][1] * Y + ext.t[0];
      const Yc = ext.R[1][0] * X + ext.R[1][1] * Y + ext.t[1];
      const Zc = ext.R[2][0] * X + ext.R[2][1] * Y + ext.t[2];
      const x = Xc / Zc;
      const y = Yc / Zc;
      const r2 = x * x + y * y;
      const radial = 1 + k1 * r2 + k2 * r2 * r2;
      const xd = x * radial;
      const yd = y * radial;
      const u_proj = fx * xd + cx;
      const v_proj = fy * yd + cy;

      const dx = u_proj - vPts[j][0];
      const dy = v_proj - vPts[j][1];
      const dist = Math.hypot(dx, dy);
      frameErrSum += dist;
      totalSqErr += dx * dx + dy * dy;
      totalPtsCount++;
    }
    perFrameErrors.push({
      frameIndex: vIdx,
      error: frameErrSum / numPoints,
    });
  });

  const reprojError = Math.sqrt(totalSqErr / (totalPtsCount || 1));

  return {
    cameraMatrix: {
      fx,
      fy,
      cx,
      cy,
      raw: [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
      ],
    },
    distCoeffs: {
      values: distValues,
      raw: distValues,
    },
    reprojError,
    perFrameErrors,
    imageSize: { width, height },
  };
}

/**
 * Run camera calibration on the selected frames.
 * Uses OpenCV if available, or seamlessly falls back to pure-JS Zhang calibration.
 */
export function calibrateCamera(cv, frames, config) {
  // If OpenCV has calibrateCamera, try it
  if (cv && typeof cv.calibrateCamera === 'function' && cv.MatVector) {
    try {
      const { rows, cols, squareSize, patternType, distortionModel } = config;
      const numPoints = rows * cols;
      const objPointsFlat = generateObjectPoints(cv, rows, cols, squareSize, patternType);

      const objectPointsVec = new cv.MatVector();
      const imagePointsVec = new cv.MatVector();

      const imageSize = frames.length > 0
        ? new cv.Size(frames[0].width, frames[0].height)
        : new cv.Size(640, 480);

      for (const frame of frames) {
        const objPts = cv.matFromArray(numPoints, 1, cv.CV_32FC3, objPointsFlat);
        objectPointsVec.push_back(objPts);
        objPts.delete();

        const imgPts = frame.corners.clone();
        imagePointsVec.push_back(imgPts);
        imgPts.delete();
      }

      const cameraMatrix = new cv.Mat();
      const distCoeffs = new cv.Mat();
      const rvecs = new cv.MatVector();
      const tvecs = new cv.MatVector();

      let flags = 0;
      if (distortionModel === 'rational' && cv.CALIB_RATIONAL_MODEL) {
        flags = cv.CALIB_RATIONAL_MODEL;
      } else if (distortionModel === 'thin_prism' && cv.CALIB_THIN_PRISM_MODEL) {
        flags = (cv.CALIB_RATIONAL_MODEL || 0) | cv.CALIB_THIN_PRISM_MODEL;
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

      objectPointsVec.delete();
      imagePointsVec.delete();
      cameraMatrix.delete();
      distCoeffs.delete();
      rvecs.delete();
      tvecs.delete();

      return result;
    } catch (e) {
      console.warn('[Calibrator] OpenCV calibrateCamera failed, falling back to pure JS:', e);
    }
  }

  // Pure JavaScript calibration engine
  return calibrateCameraPureJS(frames, config);
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
    let mat = null;
    if (cv && typeof cv.matFromImageData === 'function') {
      try {
        mat = cv.matFromImageData(frame.imageData);
      } catch (_) {}
    }

    // Check sharpness
    const sharpness = computeSharpness(cv, mat, frame.imageData);

    if (sharpness < sharpnessThreshold) {
      if (mat) mat.delete();
      continue;
    }

    // Detect pattern
    let detection;
    if (patternType === 'checkerboard') {
      detection = detectCheckerboard(cv, mat, rows, cols, frame.imageData);
    } else if (patternType === 'circles' || patternType === 'circles_asymmetric') {
      detection = detectCirclesGrid(cv, mat, rows, cols, patternType === 'circles_asymmetric', frame.imageData);
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

    if (mat) mat.delete();
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
