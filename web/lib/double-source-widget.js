/**
 * createDoubleSourceWidget — two source panels (A + B) + optional duration section.
 *
 * Options:
 *   defaultDur   {number}   Initial / fallback duration (s) for synth sources.
 *   minDur       {number}   Duration input min.
 *   maxDur       {number}   Duration input max (defaults to MAX_S).
 *   stepDur      {number}   Duration input step.
 *   showDuration {boolean}  Show the Duration section (default true).
 *                           When false, synth sources use defaultDur; the panel
 *                           is not rendered.  buildAudio still works (uses
 *                           getDuration() internally).
 *   defaultFreqA {number}   Initial frequency for panel A (Hz).
 *   defaultFreqB {number}   Initial frequency for panel B (Hz).
 *   onChange     {fn(delay)}  Forwarded from both panels and duration input.
 *
 * Returns:
 *   element                     — wrapper div (display:contents) to append to #controls
 *   panelA / panelB             — createSourcePanel return objects
 *   getDuration()               — current duration value, clamped to maxDur
 *   buildAudio(panel, len, sr)  — async; loads/generates, trims/loops to len, applies vol + gate
 *   buildAudioAuto(panel, sr)   — async; loads file at its natural length (capped at MAX_S);
 *                                 for synth sources uses getDuration() / defaultDur.
 *   buildBothAudios(len, sr)    — async convenience; returns { audioA, audioB }
 */
import { createSourcePanel } from './source-panel.js';
import { generateSource, applyPulseGate, loadFile, loopToLength, getAudioFileDuration, MAX_S } from './audio-utils.js';

export function createDoubleSourceWidget({
  defaultDur   = 4,
  minDur       = 0.5,
  maxDur       = MAX_S,
  stepDur      = 0.5,
  showDuration = true,
  defaultFreqA = 440,
  defaultFreqB = 880,
  onChange     = () => {},
} = {}) {
  const el = document.createElement('div');
  el.style.display = 'contents';

  // Duration section — always created so getDuration() works, but only appended
  // to the DOM when showDuration is true.
  const durSection = document.createElement('div');
  durSection.className = 'section';
  durSection.innerHTML = `
    <div class="section-title">Duration</div>
    <div class="row">
      <span class="lbl">Target</span>
      <input type="number" value="${defaultDur}" min="${minDur}" max="${maxDur}" step="${stepDur}">
      <span class="unit">s</span>
    </div>
  `.trim();
  const durInput = durSection.querySelector('input');

  async function onFilePicked(file) {
    if (!showDuration) return;   // don't clobber a hidden input or confuse the user
    const fileDur = await getAudioFileDuration(file);
    durInput.value = Math.min(Math.max(fileDur, minDur), maxDur).toFixed(2);
  }

  const panelA = createSourcePanel({
    title: 'Source A', defaultFreq: defaultFreqA,
    showVol: true, showGate: true, showSilence: true,
    onChange, onFilePicked,
  });

  const panelB = createSourcePanel({
    title: 'Source B', defaultFreq: defaultFreqB,
    showVol: true, showGate: true, showSilence: true,
    onChange, onFilePicked,
  });

  const swapSection = document.createElement('div');
  swapSection.className = 'section';
  swapSection.innerHTML = `<button class="tog" style="width:100%">⇅ Swap A / B</button>`;
  swapSection.querySelector('button').addEventListener('click', () => {
    const stateA = panelA.getState();
    panelA.setState(panelB.getState());
    panelB.setState(stateA);
    onChange(0);
  });

  el.appendChild(panelA.element);
  el.appendChild(swapSection);
  el.appendChild(panelB.element);

  if (showDuration) {
    durInput.addEventListener('input', () => onChange(500));
    el.appendChild(durSection);
  }

  function getDuration() {
    return Math.min(parseFloat(durInput.value) || defaultDur, maxDur);
  }

  // ── buildAudio ─────────────────────────────────────────────────────────────
  // Classic API: loads/generates to an exact finalLen, trims or loops as needed.
  // Used by experiments 1–5.
  async function buildAudio(panel, finalLen, sr) {
    const type = panel.getType();
    let audio;
    if (type === 'file') {
      const f = panel.getFile();
      if (!f) throw new Error(`No file selected for "${panel.element.querySelector('.section-title').textContent}".`);
      const raw = await loadFile(f, sr);
      audio = raw.length === finalLen ? raw
            : raw.length  >  finalLen ? raw.slice(0, finalLen)
            : loopToLength(raw, finalLen);
    } else {
      audio = generateSource(type, panel.getFreq(), finalLen / sr, sr);
    }
    const vol = panel.getVolume();
    if (vol !== 1) audio = audio.map(x => x * vol);
    if (panel.isGateEnabled()) audio = applyPulseGate(audio, sr, panel.getGateFreq(), panel.getGateDecay());
    return audio;
  }

  // ── buildAudioAuto ─────────────────────────────────────────────────────────
  // Loads each file at its natural length, capped at MAX_S (10 s).
  // A and B are independent — they need not be the same length.
  // Synth sources use getDuration() (or defaultDur when the panel is hidden).
  // channels = 2 → planar stereo Float32Array(2*T), L plane then R plane.
  // Files keep their real channels; synth sources are duplicated to both planes.
  async function buildAudioAuto(panel, sr, channels = 1) {
    const type = panel.getType();
    let audio;
    if (type === 'file') {
      const f = panel.getFile();
      if (!f) throw new Error(`No file selected for "${panel.element.querySelector('.section-title').textContent}".`);
      // When "load full file" is checked, pass Infinity so loadFile uses the
      // complete file duration.  Otherwise clamp to the panel's trim value.
      const maxS = panel.getLoadFull() ? Infinity : panel.getTrimS();
      audio = await loadFile(f, sr, maxS, channels);
    } else {
      const len = Math.round(getDuration() * sr);
      audio = generateSource(type, panel.getFreq(), len / sr, sr);
      if (channels === 2) {
        const st = new Float32Array(2 * audio.length);
        st.set(audio, 0);
        st.set(audio, audio.length);
        audio = st;
      }
    }
    const vol = panel.getVolume();
    if (vol !== 1) audio = audio.map(x => x * vol);
    if (panel.isGateEnabled()) audio = applyPulseGate(audio, sr, panel.getGateFreq(), panel.getGateDecay(), channels);
    return audio;
  }

  // ── buildBothAudios ────────────────────────────────────────────────────────
  async function buildBothAudios(finalLen, sr) {
    const audioA = await buildAudio(panelA, finalLen, sr);
    const audioB = await buildAudio(panelB, finalLen, sr);
    return { audioA, audioB };
  }

  return { element: el, panelA, panelB, getDuration, buildAudio, buildAudioAuto, buildBothAudios };
}
