/**
 * createModelPanel — builds Model and Mode .section elements.
 *
 * Options:
 *   defaultModel     {'24k'|'48k'}  Initial model selection.
 *   defaultBw        {number}       Initial bandwidth (kbps); nearest option chosen.
 *   defaultStreaming  {boolean}      Initial streaming mode state.
 *   enableSnac       {boolean}      Add SNAC model variants (24/32/44 kHz) with
 *                                   chunk/OLA processing controls. Off by default
 *                                   so experiments that only speak EnCodec are
 *                                   unaffected.
 *   onChange         {fn()}         Called whenever any control changes.
 *
 * Returns an object with:
 *   modelSection  — <div class="section"> containing Model + BW (+ SNAC proc) selectors
 *   modeSection   — <div class="section"> containing streaming mode toggles
 *   getCodec()    — 'encodec' | 'snac'
 *   getModelHz()  — '24k' | '48k'  (EnCodec); raw select value for SNAC
 *   getSnacModel()— '24khz' | '32khz' | '44khz' | null
 *   getBwKbps()   — number
 *   isStreaming() — boolean
 *   getFrameSize()— number (samples per frame, 24k streaming only)
 *   getSampleRate()— 24000 | 32000 | 44100 | 48000
 *   getChunkConfig()— { lenSec, olaSec } | null   (null => whole file / EnCodec)
 */

const BW_OPTIONS = { '24k': [1.5, 3, 6, 12, 24], '48k': [3, 6, 12, 24] };

// SNAC variants → sample rate. Values are prefixed 'snac:' in the <select>.
const SNAC_SR = { '24khz': 24000, '32khz': 32000, '44khz': 44100 };

export function createModelPanel({
  defaultModel    = '24k',
  defaultBw       = 6,
  defaultStreaming = false,
  enableSnac      = false,
  onChange        = () => {},
} = {}) {

  const snacOptions = enableSnac ? `
        <optgroup label="SNAC">
          <option value="snac:24khz">SNAC 24 kHz · 3 levels</option>
          <option value="snac:32khz">SNAC 32 kHz · 4 levels</option>
          <option value="snac:44khz">SNAC 44.1 kHz · 4 levels</option>
        </optgroup>` : '';

  // ── Model section ──────────────────────────────────────────────────────
  const modelSection = document.createElement('div');
  modelSection.className = 'section';
  modelSection.innerHTML = `
    <div class="section-title">Model</div>
    <div class="row">
      <span class="lbl">Model</span>
      <select class="mp-model">
        ${enableSnac ? '<optgroup label="EnCodec">' : ''}
        <option value="24k">24 kHz</option>
        <option value="48k">48 kHz</option>
        ${enableSnac ? '</optgroup>' : ''}
        ${snacOptions}
      </select>
    </div>
    <div class="row mp-bw-row">
      <span class="lbl">Bandwidth</span>
      <select class="mp-bw"></select>
      <span class="unit">kbps</span>
    </div>
    <div class="row mp-proc-row" style="display:none">
      <span class="lbl">Process</span>
      <select class="mp-proc">
        <option value="whole">Whole file</option>
        <option value="chunked">Chunked</option>
      </select>
    </div>
    <div class="row mp-chunk-row" style="display:none">
      <span class="lbl">Chunk</span>
      <input class="mp-chunk-sec" type="number" value="1.0" min="0.1" step="0.1" style="width:4em">
      <span class="unit">s</span>
      <label style="margin-left:.6em"><input type="checkbox" class="mp-ola"> OLA</label>
      <input class="mp-ola-sec" type="number" value="0.2" min="0" step="0.05" style="width:4em">
      <span class="unit">s</span>
    </div>
  `.trim();

  // ── Mode section ───────────────────────────────────────────────────────
  const modeSection = document.createElement('div');
  modeSection.className = 'section';
  modeSection.innerHTML = `
    <div class="section-title">Mode</div>
    <div class="toggle-row mp-mode-row" style="margin-bottom:0.45em">
      <button class="tog mp-ns active">Non-streaming</button>
      <button class="tog mp-s">Streaming</button>
    </div>
    <div class="row mp-frame-row" style="display:none">
      <span class="lbl">Frame</span>
      <select class="mp-frame">
        <option value="320">320 smp · 13 ms</option>
        <option value="640">640 smp · 27 ms</option>
        <option value="1280">1280 smp · 53 ms</option>
        <option value="2560">2560 smp · 107 ms</option>
        <option value="4800">4800 smp · 200 ms</option>
        <option value="9600">9600 smp · 400 ms</option>
      </select>
    </div>
  `.trim();

  const modelEl  = modelSection.querySelector('.mp-model');
  const bwRow    = modelSection.querySelector('.mp-bw-row');
  const bwEl     = modelSection.querySelector('.mp-bw');
  const procRow  = modelSection.querySelector('.mp-proc-row');
  const procEl   = modelSection.querySelector('.mp-proc');
  const chunkRow = modelSection.querySelector('.mp-chunk-row');
  const chunkSecEl = modelSection.querySelector('.mp-chunk-sec');
  const olaEl    = modelSection.querySelector('.mp-ola');
  const olaSecEl = modelSection.querySelector('.mp-ola-sec');
  const modeRow  = modeSection.querySelector('.mp-mode-row');
  const togNS    = modeSection.querySelector('.mp-ns');
  const togS     = modeSection.querySelector('.mp-s');
  const frameRow = modeSection.querySelector('.mp-frame-row');
  const frameEl  = modeSection.querySelector('.mp-frame');

  let streaming = defaultStreaming;
  procEl.value = 'chunked';   // SNAC defaults to chunked (parallel; scales to long audio)

  const isSnac = () => modelEl.value.startsWith('snac:');

  function populateBW() {
    bwEl.innerHTML = '';
    const opts = BW_OPTIONS[isSnac() ? '24k' : modelEl.value] || [];
    opts.forEach(bw => {
      const o = document.createElement('option');
      o.value = bw; o.textContent = bw;
      if (bw === defaultBw) o.selected = true;
      bwEl.appendChild(o);
    });
  }

  // Show/hide rows based on codec + processing mode.
  function refreshVisibility() {
    const snac = isSnac();
    bwRow.style.display = snac ? 'none' : '';
    modeSection.style.display = snac ? 'none' : '';
    procRow.style.display = snac ? '' : 'none';
    chunkRow.style.display = (snac && procEl.value === 'chunked') ? '' : 'none';
    olaSecEl.style.display = olaEl.checked ? '' : 'none';
    // EnCodec mode/frame visibility (unchanged behaviour)
    if (!snac) {
      const is48 = modelEl.value === '48k';
      modeRow.style.display = is48 ? 'none' : '';
      if (is48) { streaming = false; togNS.classList.add('active'); togS.classList.remove('active'); }
      frameRow.style.display = (streaming && modelEl.value === '24k') ? '' : 'none';
    }
  }

  modelEl.value = defaultModel;
  populateBW();
  refreshVisibility();

  modelEl.addEventListener('change', () => { populateBW(); refreshVisibility(); onChange(); });
  bwEl   .addEventListener('change', () => onChange());
  frameEl.addEventListener('change', () => onChange());
  procEl .addEventListener('change', () => { refreshVisibility(); onChange(); });
  olaEl  .addEventListener('change', () => { refreshVisibility(); onChange(); });
  [chunkSecEl, olaSecEl].forEach(el => el.addEventListener('change', () => onChange()));

  [togNS, togS].forEach(btn => btn.addEventListener('click', () => {
    togNS.classList.toggle('active', btn === togNS);
    togS .classList.toggle('active', btn === togS);
    streaming = btn === togS;
    frameRow.style.display = (streaming && modelEl.value === '24k') ? '' : 'none';
    onChange();
  }));

  return {
    modelSection,
    modeSection,
    getCodec:     () => isSnac() ? 'snac' : 'encodec',
    getModelHz:   () => modelEl.value,
    getSnacModel: () => isSnac() ? modelEl.value.slice(5) : null,
    getBwKbps:    () => parseFloat(bwEl.value),
    isStreaming:  () => isSnac() ? false : streaming,
    getFrameSize: () => parseInt(frameEl.value),
    getSampleRate:() => isSnac() ? SNAC_SR[modelEl.value.slice(5)]
                                 : (modelEl.value === '24k' ? 24000 : 48000),
    getChunkConfig() {
      if (!isSnac() || procEl.value !== 'chunked') return null;
      return {
        lenSec: parseFloat(chunkSecEl.value) || 1.0,
        olaSec: olaEl.checked ? (parseFloat(olaSecEl.value) || 0) : 0,
      };
    },
    /** Programmatically switch model (rebuilds BW options). Does NOT fire onChange. */
    setModelHz(hz) {
      modelEl.value = hz;
      populateBW();
      refreshVisibility();
    },
    /** Programmatically select a bandwidth. Call after setModelHz. Does NOT fire onChange. */
    setBwKbps(kbps) {
      bwEl.value = kbps;
    },
  };
}
