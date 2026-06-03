/**
 * createDoubleSourceWidget — two source panels (A + B) + duration section, bundled.
 *
 * Options:
 *   defaultDur   {number}   Initial duration (s).
 *   minDur       {number}   Duration input min.
 *   maxDur       {number}   Duration input max (defaults to MAX_S).
 *   stepDur      {number}   Duration input step.
 *   defaultFreqA {number}   Initial frequency for panel A (Hz).
 *   defaultFreqB {number}   Initial frequency for panel B (Hz).
 *   onChange     {fn(delay)}  Forwarded from both panels and duration input.
 *
 * Returns:
 *   element                    — wrapper div (display:contents) to append to #controls
 *   panelA / panelB            — createSourcePanel return objects
 *   getDuration()              — current duration value, clamped to maxDur
 *   buildAudio(panel, len, sr) — async; loads/generates, trims/loops to len, applies vol + gate
 *   buildBothAudios(len, sr)   — async convenience; returns { audioA, audioB }
 */
import { createSourcePanel } from './source-panel.js';
import { generateSource, applyPulseGate, loadFile, loopToLength, getAudioFileDuration, MAX_S } from './audio-utils.js';

export function createDoubleSourceWidget({
  defaultDur   = 4,
  minDur       = 0.5,
  maxDur       = MAX_S,
  stepDur      = 0.5,
  defaultFreqA = 440,
  defaultFreqB = 880,
  onChange     = () => {},
} = {}) {
  const el = document.createElement('div');
  el.style.display = 'contents';

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
    const fileDur = await getAudioFileDuration(file);
    durInput.value = Math.min(Math.max(fileDur, minDur), maxDur).toFixed(2);
  }

  const panelA = createSourcePanel({
    title: 'Source A', defaultFreq: defaultFreqA,
    showVol: true, showGate: true, showSilence: true,
    onChange, onFilePicked,
  });
  el.appendChild(panelA.element);

  const panelB = createSourcePanel({
    title: 'Source B', defaultFreq: defaultFreqB,
    showVol: true, showGate: true, showSilence: true,
    onChange, onFilePicked,
  });
  el.appendChild(panelB.element);

  durInput.addEventListener('input', () => onChange(500));
  el.appendChild(durSection);

  function getDuration() {
    return Math.min(parseFloat(durInput.value) || defaultDur, maxDur);
  }

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

  async function buildBothAudios(finalLen, sr) {
    const audioA = await buildAudio(panelA, finalLen, sr);
    const audioB = await buildAudio(panelB, finalLen, sr);
    return { audioA, audioB };
  }

  return { element: el, panelA, panelB, getDuration, buildAudio, buildBothAudios };
}
