/**
 * createSourcePanel — builds a .section element with source controls.
 *
 * Options:
 *   title        {string}   Section heading text.
 *   defaultFreq  {number}   Initial frequency value (Hz).
 *   defaultVol   {number}   Initial volume value 0–1 (only when showVol=true).
 *   showVol      {boolean}  Include a volume slider row.
 *   showGate     {boolean}  Include gate toggle + rate/decay rows.
 *   showSilence  {boolean}  Include "Silence" in the type dropdown.
 *   onChange     {fn(delay)}  Called on any change. `delay` is 0 for instant
 *                             controls (selects, sliders, toggles) and 500 for
 *                             number inputs so callers can debounce.
 *   onFilePicked {async fn(File)|null}  Awaited before onChange fires after a
 *                             file selection — lets callers update dependent
 *                             fields (e.g. duration) before the recalc starts.
 *
 * Returns an object with:
 *   element       — the <div class="section"> to append to #controls
 *   getType()     — current source type string
 *   getFreq()     — current frequency (Hz)
 *   getFile()     — selected File | null
 *   getVolume()   — 0–1 (1 if showVol=false)
 *   isGateEnabled()
 *   getGateFreq() — gate repetition rate (Hz)
 *   getGateDecay()— gate decay (ms)
 *   setEnabled(bool) — enable/disable all controls
 */
export function createSourcePanel({
  title        = 'Source',
  defaultFreq  = 440,
  defaultVol   = 0.8,
  showVol      = false,
  showGate     = false,
  showSilence  = false,
  onChange     = () => {},
  onFilePicked = null,
} = {}) {

  const section = document.createElement('div');
  section.className = 'section';
  section.innerHTML = `
    <div class="section-title">${title}</div>
    <div class="row">
      <span class="lbl">Type</span>
      <select class="src-type">
        <option value="sine">Sine</option>
        <option value="triangle">Triangle</option>
        <option value="sawtooth">Sawtooth</option>
        <option value="square">Square</option>
        <option value="sweep">Sweep</option>
        ${showSilence ? '<option value="silence">Silence</option>' : ''}
        <option value="file">File</option>
      </select>
    </div>
    <div class="row src-freq-row">
      <span class="lbl">Freq</span>
      <input type="number" class="src-freq" value="${defaultFreq}" min="1" max="20000" step="1">
      <span class="unit">Hz</span>
    </div>
    <div class="row src-file-row" style="display:none">
      <button class="file-btn src-file-btn">choose file…</button>
      <input type="file" class="src-file-input" accept="audio/*" style="display:none">
    </div>
    ${showVol ? `
    <div class="row">
      <span class="lbl">Vol</span>
      <input type="range" class="vol-slider src-vol" min="0" max="1" step="0.01" value="${defaultVol}">
    </div>` : ''}
    ${showGate ? `
    <div class="row">
      <span class="lbl">Gate</span>
      <button class="tog src-gate-tog">Off</button>
    </div>
    <div class="row src-gate-rate-row" style="display:none">
      <span class="lbl">Rate</span>
      <input type="number" class="src-gate-freq" value="4" min="0.1" max="500" step="0.1">
      <span class="unit">Hz</span>
    </div>
    <div class="row src-gate-decay-row" style="display:none">
      <span class="lbl">Decay</span>
      <input type="number" class="src-gate-decay" value="50" min="0" max="2000" step="1">
      <span class="unit">ms</span>
    </div>` : ''}
  `.trim();

  const typeEl      = section.querySelector('.src-type');
  const freqRow     = section.querySelector('.src-freq-row');
  const freqEl      = section.querySelector('.src-freq');
  const fileRow     = section.querySelector('.src-file-row');
  const fileBtnEl   = section.querySelector('.src-file-btn');
  const fileInputEl = section.querySelector('.src-file-input');
  const volEl       = section.querySelector('.src-vol');
  const gateTogEl   = section.querySelector('.src-gate-tog');
  const gateRateRow = section.querySelector('.src-gate-rate-row');
  const gateDecRow  = section.querySelector('.src-gate-decay-row');
  const gateFreqEl  = section.querySelector('.src-gate-freq');
  const gateDecEl   = section.querySelector('.src-gate-decay');

  let gateEnabled = false;
  let currentFile = null;

  function updateVisibility() {
    const v = typeEl.value;
    fileRow.style.display = v === 'file' ? '' : 'none';
    freqRow.style.display = (v === 'file' || v === 'sweep' || v === 'silence') ? 'none' : '';
  }

  typeEl.addEventListener('change', () => { updateVisibility(); onChange(0); });
  freqEl.addEventListener('input',  () => onChange(500));

  fileBtnEl.addEventListener('click', () => fileInputEl.click());
  fileInputEl.addEventListener('change', async () => {
    if (!fileInputEl.files[0]) return;
    currentFile = fileInputEl.files[0];
    fileBtnEl.textContent = currentFile.name;
    if (onFilePicked) await onFilePicked(currentFile);
    onChange(0);
  });

  if (volEl) volEl.addEventListener('input', () => onChange(0));

  if (gateTogEl) {
    gateTogEl.addEventListener('click', () => {
      gateEnabled = !gateEnabled;
      gateTogEl.textContent = gateEnabled ? 'On' : 'Off';
      gateTogEl.classList.toggle('active', gateEnabled);
      gateRateRow.style.display = gateEnabled ? '' : 'none';
      gateDecRow.style.display  = gateEnabled ? '' : 'none';
      onChange(0);
    });
    if (gateFreqEl) gateFreqEl.addEventListener('input', () => onChange(500));
    if (gateDecEl)  gateDecEl .addEventListener('input', () => onChange(500));
  }

  updateVisibility();

  return {
    element:       section,
    getType:       () => typeEl.value,
    getFreq:       () => parseFloat(freqEl.value) || 440,
    getFile:       () => currentFile,
    getVolume:     () => volEl ? parseFloat(volEl.value) : 1,
    isGateEnabled: () => gateEnabled,
    getGateFreq:   () => gateFreqEl ? (parseFloat(gateFreqEl.value) || 4)  : 4,
    getGateDecay:  () => gateDecEl  ? (parseFloat(gateDecEl.value)  || 0)  : 0,
    setEnabled(on) {
      [typeEl, freqEl, fileInputEl, volEl, gateTogEl, gateFreqEl, gateDecEl]
        .filter(Boolean).forEach(el => { el.disabled = !on; });
      fileBtnEl.style.opacity = on ? '' : '0.4';
      fileBtnEl.style.pointerEvents = on ? '' : 'none';
    },
  };
}
