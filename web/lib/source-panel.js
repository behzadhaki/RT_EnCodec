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
 *   getState()    — snapshot of all panel state (for swap / copy)
 *   setState(s)   — restore a snapshot; updates UI and visibility
 *   setEnabled(bool) — enable/disable all controls
 */
export function createSourcePanel({
  title        = 'Source',
  defaultFreq  = 440,
  defaultVol   = 0.8,
  showVol      = false,
  showGate     = false,
  showSilence  = false,
  showFolder   = false,
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
        ${showFolder ? '<option value="folder">Folder (join WAVs)</option>' : ''}
      </select>
    </div>
    <div class="row src-freq-row">
      <span class="lbl">Freq</span>
      <input type="number" class="src-freq" value="${defaultFreq}" min="1" max="20000" step="1">
      <span class="unit">Hz</span>
    </div>
    <div class="row src-file-row" style="display:none">
      <button class="file-btn src-file-btn">choose file…</button>
      <input type="file" class="src-file-input" accept="audio/*,audio/mp4,audio/x-m4a,.m4a,.m4b,.caf,.aiff,.aif" style="display:none">
    </div>
    ${showFolder ? `
    <div class="row src-folder-row" style="display:none">
      <button class="file-btn src-folder-btn">choose folder…</button>
      <input type="file" class="src-folder-input" webkitdirectory directory multiple style="display:none">
    </div>` : ''}
    <div class="row src-load-row" style="display:none">
      <span class="lbl">Load</span>
      <label style="display:flex;align-items:center;gap:4px;font-size:11px;cursor:pointer;color:var(--muted)">
        <input type="checkbox" class="src-load-full" checked>
        full file
      </label>
    </div>
    <div class="row src-trim-row" style="display:none">
      <span class="lbl">Trim to</span>
      <input type="number" class="src-trim-s" value="10" min="0.1" max="7200" step="0.5">
      <span class="unit">s</span>
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
  const folderRow     = section.querySelector('.src-folder-row');
  const folderBtnEl   = section.querySelector('.src-folder-btn');
  const folderInputEl = section.querySelector('.src-folder-input');
  const loadRow     = section.querySelector('.src-load-row');
  const loadFullEl  = section.querySelector('.src-load-full');
  const trimRow     = section.querySelector('.src-trim-row');
  const trimSEl     = section.querySelector('.src-trim-s');
  const volEl       = section.querySelector('.src-vol');
  const gateTogEl   = section.querySelector('.src-gate-tog');
  const gateRateRow = section.querySelector('.src-gate-rate-row');
  const gateDecRow  = section.querySelector('.src-gate-decay-row');
  const gateFreqEl  = section.querySelector('.src-gate-freq');
  const gateDecEl   = section.querySelector('.src-gate-decay');

  let gateEnabled       = false;
  let currentFile       = null;
  let currentFileHandle = null;   // FileSystemFileHandle when available (null on Firefox)
  let currentFolder     = null;   // Array<File> of .wav files, name-sorted (folder mode)
  let currentFolderName = '';

  function updateVisibility() {
    const v        = typeEl.value;
    const isFile   = v === 'file';
    const isFolder = v === 'folder';
    const loadFull = loadFullEl.checked;
    fileRow.style.display = isFile ? '' : 'none';
    if (folderRow) folderRow.style.display = isFolder ? '' : 'none';
    // Folder length is governed by the total-seconds budget, not per-clip trim.
    loadRow.style.display = isFile ? '' : 'none';
    trimRow.style.display = (isFile && !loadFull) ? '' : 'none';
    freqRow.style.display = (isFile || isFolder || v === 'sweep' || v === 'silence') ? 'none' : '';
  }

  typeEl.addEventListener('change',   () => { updateVisibility(); onChange(0); });
  freqEl.addEventListener('input',    () => onChange(500));
  loadFullEl.addEventListener('change', () => { updateVisibility(); onChange(0); });
  trimSEl.addEventListener('input',   () => onChange(500));

  // File button — prefer showOpenFilePicker (gives a handle for cross-session
  // persistence); fall back to the hidden <input type="file"> on Firefox.
  fileBtnEl.addEventListener('click', async () => {
    if ('showOpenFilePicker' in window) {
      let handle;
      try {
        [handle] = await window.showOpenFilePicker({
          types: [{
            description: 'Audio files',
            accept: {
              'audio/mpeg': ['.mp3'],
              'audio/wav':  ['.wav', '.wave'],
              'audio/flac': ['.flac'],
              'audio/ogg':  ['.ogg', '.oga', '.opus'],
              'audio/mp4':  ['.m4a', '.m4b', '.mp4', '.aac'],
              'audio/aiff': ['.aiff', '.aif'],
              'audio/x-caf': ['.caf'],
              'audio/webm': ['.webm'],
            },
          }],
          excludeAcceptAllOption: false,
          multiple: false,
        });
      } catch (err) {
        if (err.name !== 'AbortError') throw err;
        return; // user cancelled
      }
      currentFileHandle = handle;
      currentFile = await handle.getFile();
    } else {
      // Firefox fallback — input change handler takes over
      fileInputEl.click();
      return;
    }
    fileBtnEl.textContent = currentFile.name;
    if (onFilePicked) await onFilePicked(currentFile);
    onChange(0);
  });

  // Firefox fallback: fired when user picks via the hidden input
  fileInputEl.addEventListener('change', async () => {
    if (!fileInputEl.files[0]) return;
    currentFileHandle = null;
    currentFile = fileInputEl.files[0];
    fileBtnEl.textContent = currentFile.name;
    if (onFilePicked) await onFilePicked(currentFile);
    onChange(0);
  });

  // Folder picker — collects every .wav in the chosen directory, name-sorted,
  // to be concatenated into one source. Prefers showDirectoryPicker (gives a
  // folder name + clean iteration); falls back to a webkitdirectory <input>.
  function setFolder(files, name) {
    // Keep every .wav, name-sorted. The joined length is bounded later by the
    // total-seconds budget (see loadFolder / the overflow dialog), not a count.
    const wavs = [...files].filter(f => /\.wave?$/i.test(f.name))
      .sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: 'base' }));
    currentFolder     = wavs.length ? wavs : null;
    currentFolderName = name || '';
    folderBtnEl.textContent = wavs.length
      ? `${currentFolderName || 'folder'} · ${wavs.length} wav`
      : 'no .wav files found';
  }

  if (folderBtnEl) {
    folderBtnEl.addEventListener('click', async () => {
      if ('showDirectoryPicker' in window) {
        let dirHandle;
        try { dirHandle = await window.showDirectoryPicker(); }
        catch (err) { if (err.name !== 'AbortError') throw err; return; }
        const files = [];
        for await (const entry of dirHandle.values()) {
          if (entry.kind === 'file' && /\.wave?$/i.test(entry.name)) files.push(await entry.getFile());
        }
        setFolder(files, dirHandle.name);
        onChange(0);
      } else {
        folderInputEl.click(); // change handler takes over
      }
    });

    folderInputEl.addEventListener('change', () => {
      const files = [...folderInputEl.files];
      if (!files.length) return;
      const name = files[0].webkitRelativePath?.split('/')[0] || 'folder';
      setFolder(files, name);
      onChange(0);
    });
  }

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
    element:         section,
    getType:         () => typeEl.value,
    getFreq:         () => parseFloat(freqEl.value) || 440,
    getFile:         () => currentFile,
    getFileHandle:   () => currentFileHandle,
    getFolder:       () => currentFolder,   // Array<File> | null (folder mode)
    /** Silently restore a file (+ optional handle) without firing callbacks.
     *  Call onFilePicked / onChange separately if needed. */
    setFile(file, handle = null) {
      currentFile       = file;
      currentFileHandle = handle ?? null;
      fileBtnEl.textContent = file.name;
      updateVisibility();
    },
    getLoadFull:   () => loadFullEl.checked,
    getTrimS:      () => parseFloat(trimSEl.value) || 10,
    getVolume:     () => volEl ? parseFloat(volEl.value) : 1,
    isGateEnabled: () => gateEnabled,
    getGateFreq:   () => gateFreqEl ? (parseFloat(gateFreqEl.value) || 4)  : 4,
    getGateDecay:  () => gateDecEl  ? (parseFloat(gateDecEl.value)  || 0)  : 0,
    getState() {
      return {
        type: typeEl.value,
        freq: parseFloat(freqEl.value) || 440,
        file: currentFile,
        handle: currentFileHandle,
        fileName: currentFile ? fileBtnEl.textContent : null,
        folder: currentFolder,
        folderName: currentFolderName,
        folderLabel: folderBtnEl ? folderBtnEl.textContent : null,
        loadFull: loadFullEl.checked,
        trimS:    parseFloat(trimSEl.value) || 10,
        volume: volEl ? parseFloat(volEl.value) : 1,
        gateEnabled,
        gateFreq: gateFreqEl ? (parseFloat(gateFreqEl.value) || 4) : 4,
        gateDecay: gateDecEl  ? (parseFloat(gateDecEl.value)  || 0) : 0,
      };
    },
    setState(s) {
      typeEl.value = s.type;
      freqEl.value = s.freq;
      currentFile       = s.file   ?? null;
      currentFileHandle = s.handle ?? null;   // null if not in saved state (e.g. localStorage restore)
      fileBtnEl.textContent = s.fileName || 'choose file…';
      currentFolder     = s.folder ?? null;   // in-memory File[] (survives swap, not localStorage)
      currentFolderName = s.folderName ?? '';
      if (folderBtnEl) folderBtnEl.textContent = s.folderLabel || 'choose folder…';
      if (s.loadFull !== undefined) loadFullEl.checked = s.loadFull;
      if (s.trimS   !== undefined) trimSEl.value = s.trimS;
      if (volEl) volEl.value = s.volume;
      if (gateTogEl) {
        gateEnabled = s.gateEnabled;
        gateTogEl.textContent = gateEnabled ? 'On' : 'Off';
        gateTogEl.classList.toggle('active', gateEnabled);
        gateRateRow.style.display = gateEnabled ? '' : 'none';
        gateDecRow.style.display  = gateEnabled ? '' : 'none';
      }
      if (gateFreqEl) gateFreqEl.value = s.gateFreq;
      if (gateDecEl)  gateDecEl.value  = s.gateDecay;
      updateVisibility();
    },
    setEnabled(on) {
      [typeEl, freqEl, fileInputEl, folderInputEl, loadFullEl, trimSEl, volEl, gateTogEl, gateFreqEl, gateDecEl]
        .filter(Boolean).forEach(el => { el.disabled = !on; });
      for (const btn of [fileBtnEl, folderBtnEl].filter(Boolean)) {
        btn.style.opacity = on ? '' : '0.4';
        btn.style.pointerEvents = on ? '' : 'none';
      }
    },
  };
}
