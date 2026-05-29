export const FFT_N = 2048;
export const HOP   = 512;

export const PLASMA = [[5,5,25],[100,0,150],[200,50,100],[255,140,20],[255,255,100]];
export const OCEAN  = [[5,10,40],[0,60,120],[0,140,180],[40,200,220],[180,240,255]];
export const HOT    = [[0,0,0],[191,0,0],[255,128,0],[255,255,64],[255,255,255]];

function hann(n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (n - 1)));
  return w;
}
const WINDOW = hann(FFT_N);

function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let b = n >> 1;
    for (; j & b; b >>= 1) j ^= b;
    j ^= b;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
          t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < len >> 1; j++) {
        const ur = re[i+j],        ui = im[i+j];
        const vr = re[i+j+len/2]*cr - im[i+j+len/2]*ci;
        const vi = re[i+j+len/2]*ci + im[i+j+len/2]*cr;
        re[i+j] = ur+vr; im[i+j] = ui+vi;
        re[i+j+len/2] = ur-vr; im[i+j+len/2] = ui-vi;
        const nr = cr*wr - ci*wi; ci = cr*wi + ci*wr; cr = nr;
      }
    }
  }
}

// Returns { magDB, nFrames, nBins, dBmin, dBmax }.
export function computeSTFT(samples) {
  const nBins   = FFT_N >> 1;
  const nFrames = Math.max(1, Math.floor((samples.length - FFT_N) / HOP) + 1);
  const magDB   = new Float32Array(nFrames * nBins);
  const re = new Float32Array(FFT_N), im = new Float32Array(FFT_N);
  let dBmax = -Infinity;

  for (let f = 0; f < nFrames; f++) {
    const s = f * HOP;
    re.fill(0); im.fill(0);
    for (let i = 0; i < FFT_N; i++) re[i] = (samples[s + i] || 0) * WINDOW[i];
    fft(re, im);
    for (let k = 0; k < nBins; k++) {
      const m  = Math.sqrt(re[k]*re[k] + im[k]*im[k]);
      const db = m > 1e-9 ? 20 * Math.log10(m) : -120;
      magDB[f * nBins + k] = db;
      if (db > dBmax) dBmax = db;
    }
  }
  return { magDB, nFrames, nBins, dBmin: Math.max(dBmax - 100, -120), dBmax };
}

// 5-stop linear colormap interpolation. Returns [r, g, b].
export function cmap(t, stops) {
  t = Math.max(0, Math.min(1, t));
  const n = stops.length - 1;
  const i = Math.min(Math.floor(t * n), n - 1);
  const f = t * n - i;
  const a = stops[i], b = stops[i + 1];
  return [a[0]+f*(b[0]-a[0])|0, a[1]+f*(b[1]-a[1])|0, a[2]+f*(b[2]-a[2])|0];
}

// Renders an STFT result to an off-screen canvas element using the given colormap.
export function stftToOffscreen(stft, stops) {
  const { magDB, nFrames, nBins, dBmin, dBmax } = stft;
  const range = dBmax - dBmin || 1;
  const off = document.createElement('canvas');
  off.width = nFrames; off.height = nBins;
  const ctx = off.getContext('2d');
  const img = ctx.createImageData(nFrames, nBins);
  for (let f = 0; f < nFrames; f++) {
    for (let k = 0; k < nBins; k++) {
      const t  = (magDB[f * nBins + k] - dBmin) / range;
      const [r, g, b] = cmap(t, stops);
      const px = ((nBins - 1 - k) * nFrames + f) * 4;
      img.data[px]=r; img.data[px+1]=g; img.data[px+2]=b; img.data[px+3]=255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return off;
}

// Draws a vertical frequency-magnitude strip (middle STFT frame) onto canvas.
export function drawStrip(canvas, stft, stops, dpr) {
  if (!stft) return;
  const cW  = Math.round(canvas.clientWidth  * dpr) || Math.round(2.5 * 16 * dpr);
  const cH  = Math.round(canvas.clientHeight * dpr) || 400;
  canvas.width = cW; canvas.height = cH;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, cW, cH);

  const { magDB, nFrames, nBins, dBmin, dBmax } = stft;
  const range = dBmax - dBmin || 1;
  const f     = Math.floor(nFrames / 2);
  for (let k = 0; k < nBins; k++) {
    const t    = (magDB[f * nBins + k] - dBmin) / range;
    const [r, g, b] = cmap(t, stops);
    const y    = (1 - (k + 0.5) / nBins) * cH;
    const barH = Math.ceil(cH / nBins) + 1;
    const barW = Math.max(1, t * cW * 0.85);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(cW - barW, y, barW, barH);
  }
}
