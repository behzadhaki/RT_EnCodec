/**
 * createSingleSourceWidget — one source panel + duration section, bundled.
 *
 * Options:
 *   defaultFreq  {number}   Initial frequency (Hz).
 *   defaultDur   {number}   Initial duration (s).
 *   minDur       {number}   Duration input min.
 *   maxDur       {number}   Duration input max.
 *   stepDur      {number}   Duration input step.
 *   onChange     {fn(delay)}  Forwarded from the source panel and duration input.
 *
 * Returns:
 *   element        — wrapper div (display:contents) to append to #controls
 *   panel          — the createSourcePanel return object
 *   getDuration()  — current duration value, clamped to maxDur
 *   buildAudio(sr) — async; loads/generates audio then applies volume and gate
 */
import { createSourcePanel } from './source-panel.js';
import { generateSource, applyPulseGate, loadFile } from './audio-utils.js';

export function createSingleSourceWidget({
  defaultFreq = 440,
  defaultDur  = 2,
  minDur      = 0.1,
  maxDur      = 10,
  stepDur     = 0.1,
  onChange    = () => {},
} = {}) {
  const el = document.createElement('div');
  el.style.display = 'contents';

  const panel = createSourcePanel({
    title: 'Source', defaultFreq,
    showVol: true, showGate: true, showSilence: true,
    onChange,
  });
  el.appendChild(panel.element);

  const durSection = document.createElement('div');
  durSection.className = 'section';
  durSection.innerHTML = `
    <div class="section-title">Duration</div>
    <div class="row">
      <span class="lbl">Duration</span>
      <input type="number" value="${defaultDur}" min="${minDur}" max="${maxDur}" step="${stepDur}">
      <span class="unit">s</span>
    </div>
  `.trim();
  el.appendChild(durSection);

  const durInput = durSection.querySelector('input');
  durInput.addEventListener('input', () => onChange(500));

  function getDuration() {
    return Math.min(parseFloat(durInput.value) || defaultDur, maxDur);
  }

  async function buildAudio(sr) {
    const dur = getDuration();
    let audio;
    if (panel.getType() === 'file') {
      const f = panel.getFile();
      if (!f) throw new Error('No file selected.');
      audio = await loadFile(f, sr);
    } else {
      audio = generateSource(panel.getType(), panel.getFreq(), dur, sr);
    }
    const vol = panel.getVolume();
    if (vol !== 1) audio = audio.map(x => x * vol);
    if (panel.isGateEnabled()) audio = applyPulseGate(audio, sr, panel.getGateFreq(), panel.getGateDecay());
    return audio;
  }

  return { element: el, panel, getDuration, buildAudio };
}
