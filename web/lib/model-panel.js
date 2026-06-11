/**
 * createModelPanel — builds Model and Mode .section elements.
 *
 * Options:
 *   defaultModel     {'24k'|'48k'}  Initial model selection.
 *   defaultBw        {number}       Initial bandwidth (kbps); nearest option chosen.
 *   defaultStreaming  {boolean}      Initial streaming mode state.
 *   onChange         {fn()}         Called whenever any control changes.
 *
 * Returns an object with:
 *   modelSection  — <div class="section"> containing Model + BW selectors
 *   modeSection   — <div class="section"> containing streaming mode toggles
 *   getModelHz()  — '24k' | '48k'
 *   getBwKbps()   — number
 *   isStreaming() — boolean
 *   getFrameSize()— number (samples per frame, 24k streaming only)
 *   getSampleRate()— 24000 | 48000
 */

const BW_OPTIONS = { '24k': [1.5, 3, 6, 12, 24], '48k': [3, 6, 12, 24] };

export function createModelPanel({
  defaultModel    = '24k',
  defaultBw       = 6,
  defaultStreaming = false,
  onChange        = () => {},
} = {}) {

  // ── Model section ──────────────────────────────────────────────────────
  const modelSection = document.createElement('div');
  modelSection.className = 'section';
  modelSection.innerHTML = `
    <div class="section-title">Model</div>
    <div class="row">
      <span class="lbl">Model</span>
      <select class="mp-model">
        <option value="24k">24 kHz</option>
        <option value="48k">48 kHz</option>
      </select>
    </div>
    <div class="row">
      <span class="lbl">Bandwidth</span>
      <select class="mp-bw"></select>
      <span class="unit">kbps</span>
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
  const bwEl     = modelSection.querySelector('.mp-bw');
  const modeRow  = modeSection.querySelector('.mp-mode-row');
  const togNS    = modeSection.querySelector('.mp-ns');
  const togS     = modeSection.querySelector('.mp-s');
  const frameRow = modeSection.querySelector('.mp-frame-row');
  const frameEl  = modeSection.querySelector('.mp-frame');

  let streaming = defaultStreaming;

  function populateBW() {
    bwEl.innerHTML = '';
    BW_OPTIONS[modelEl.value].forEach(bw => {
      const o = document.createElement('option');
      o.value = bw; o.textContent = bw;
      if (bw === defaultBw) o.selected = true;
      bwEl.appendChild(o);
    });
    const is48 = modelEl.value === '48k';
    modeRow.style.display = is48 ? 'none' : '';
    if (is48) {
      streaming = false;
      togNS.classList.add('active');
      togS.classList.remove('active');
    }
    frameRow.style.display = (streaming && modelEl.value === '24k') ? '' : 'none';
  }

  modelEl.value = defaultModel;
  populateBW();

  modelEl.addEventListener('change', () => { populateBW(); onChange(); });
  bwEl   .addEventListener('change', () => onChange());
  frameEl.addEventListener('change', () => onChange());

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
    getModelHz:   () => modelEl.value,
    getBwKbps:    () => parseFloat(bwEl.value),
    isStreaming:  () => streaming,
    getFrameSize: () => parseInt(frameEl.value),
    getSampleRate:() => modelEl.value === '24k' ? 24000 : 48000,
    /** Programmatically switch model (rebuilds BW options). Does NOT fire onChange. */
    setModelHz(hz) {
      modelEl.value = hz;
      populateBW();
    },
    /** Programmatically select a bandwidth. Call after setModelHz. Does NOT fire onChange. */
    setBwKbps(kbps) {
      bwEl.value = kbps;
    },
  };
}
