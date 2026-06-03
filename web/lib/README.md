# lib — shared modules for EnCodec web experiments

All modules are plain ES modules. Import them with `<script type="module">`.

---

## Typical experiment layout

```js
import { createSingleSourceWidget } from './lib/single-source-widget.js'; // or double
import { createModelPanel }         from './lib/model-panel.js';
import { normalizePeak }            from './lib/audio-utils.js';
import { computeSTFT, stftToOffscreen, drawStrip, PLASMA } from './lib/stft.js';
```

---

## `single-source-widget.js`

One source panel + duration section, bundled into a single element. For experiments with a single audio input.

```js
const srcWidget = createSingleSourceWidget({
  defaultFreq: 440,   // Hz — initial synth frequency
  defaultDur:  2,     // s
  minDur:      0.1,
  maxDur:      10,
  stepDur:     0.1,
  onChange: delay => scheduleEncode(delay),
});
controls.appendChild(srcWidget.element);  // display:contents — sections slot directly into #controls flex column
const srcPanel = srcWidget.panel;         // full source-panel API if needed (getType, getFreq, …)
```

**Returns**

| Member | Description |
|---|---|
| `element` | Wrapper div (`display:contents`) — append to `#controls` |
| `panel` | The underlying `createSourcePanel` object |
| `getDuration()` | Current duration value (s), clamped to `maxDur` |
| `buildAudio(sr)` | `async` — generates or loads the audio, then applies volume and gate. Returns `Float32Array`. Throws if type is `'file'` and no file is selected. |

**Example**

```js
let audio = await srcWidget.buildAudio(sourceSR);
audio = normalizePeak(audio, 0.8);  // optional — useful for synth sources
```

---

## `double-source-widget.js`

Two source panels (A + B) + shared duration section. For experiments that compare or combine two audio inputs.

```js
const srcWidget = createDoubleSourceWidget({
  defaultDur:   4,     // s
  minDur:       0.5,
  maxDur:       10,    // defaults to MAX_S from audio-utils
  stepDur:      0.5,
  defaultFreqA: 440,   // Hz
  defaultFreqB: 880,
  onChange: delay => scheduleEncode(delay),
});
controls.appendChild(srcWidget.element);
const { panelA, panelB } = srcWidget;
```

**Returns**

| Member | Description |
|---|---|
| `element` | Wrapper div (`display:contents`) |
| `panelA` / `panelB` | The underlying `createSourcePanel` objects |
| `getDuration()` | Current duration value (s), clamped to `maxDur` |
| `buildAudio(panel, finalLen, sr)` | `async` — generates or loads audio, trims/loops to `finalLen` samples, then applies volume and gate. Returns `Float32Array`. |
| `buildBothAudios(finalLen, sr)` | `async` convenience — calls `buildAudio` for both panels sequentially. Returns `{ audioA, audioB }`. |

**Example**

```js
const sr       = mp.getSampleRate();
const finalLen = Math.round(srcWidget.getDuration() * sr);

// with status messages between the two loads:
setStatus('Loading A…');
const audioA = await srcWidget.buildAudio(panelA, finalLen, sr);
setStatus('Loading B…');
const audioB = await srcWidget.buildAudio(panelB, finalLen, sr);

// or as a single call when status messages aren't needed:
const { audioA, audioB } = await srcWidget.buildBothAudios(finalLen, sr);
```

`buildAudio` applies volume and gate internally — **do not apply them again** after the call.

---

## `source-panel.js`

Low-level building block used by both widgets. Use directly only if you need a source panel without the duration section, or with non-standard options.

```js
import { createSourcePanel } from './lib/source-panel.js';

const panel = createSourcePanel({
  title:       'Source',
  defaultFreq: 440,
  defaultVol:  0.8,
  showVol:     true,
  showGate:    true,
  showSilence: true,
  onChange:    delay => scheduleEncode(delay),
  onFilePicked: async file => { /* update dependent fields before recalc */ },
});
controls.appendChild(panel.element);
```

**Returns**

| Member | Description |
|---|---|
| `element` | `<div class="section">` to append to `#controls` |
| `getType()` | `'sine'` \| `'triangle'` \| `'sawtooth'` \| `'square'` \| `'sweep'` \| `'silence'` \| `'file'` |
| `getFreq()` | Hz |
| `getFile()` | `File` or `null` |
| `getVolume()` | 0–1 (always 1 if `showVol` is false) |
| `isGateEnabled()` | boolean |
| `getGateFreq()` | Gate repetition rate (Hz) |
| `getGateDecay()` | Gate decay (ms) |
| `setEnabled(bool)` | Enables or disables all controls |

---

## `model-panel.js`

Model and Mode selector. Produces two separate `.section` elements.

```js
const mp = createModelPanel({
  defaultModel:    '24k',   // '24k' | '48k'
  defaultBw:       6,       // kbps — nearest option chosen
  defaultStreaming: false,
  onChange: () => scheduleEncode(0),
});
controls.appendChild(mp.modelSection);
controls.appendChild(mp.modeSection);
```

**Returns**

| Member | Description |
|---|---|
| `modelSection` | `<div class="section">` — model + bandwidth selectors |
| `modeSection` | `<div class="section">` — streaming toggle + frame size |
| `getModelHz()` | `'24k'` or `'48k'` |
| `getSampleRate()` | `24000` or `48000` |
| `getBwKbps()` | number |
| `isStreaming()` | boolean |
| `getFrameSize()` | samples per frame (24k streaming only) |

---

## `audio-utils.js`

Stateless audio helpers. All functions return new `Float32Array`s — inputs are never mutated.

```js
import {
  MAX_S,
  generateSource, applyPulseGate,
  loadFile, getAudioFileDuration,
  normalizePeak, loopToLength,
} from './lib/audio-utils.js';
```

| Export | Description |
|---|---|
| `MAX_S` | Maximum clip duration in seconds (10) |
| `generateSource(type, freq, dur, sr)` | Synthesises a `Float32Array` — sine, triangle, sawtooth, square, sweep, silence |
| `applyPulseGate(audio, sr, freq, decayMs)` | Repeating exponential-decay amplitude gate |
| `loadFile(file, sr, maxS?)` | Decodes an audio `File`, resamples to mono at `sr`, trims to `maxS` seconds |
| `getAudioFileDuration(file)` | Returns file duration in seconds, capped at `MAX_S` |
| `normalizePeak(audio, target?)` | Scales so peak magnitude equals `target` (default 0.8) |
| `loopToLength(audio, targetLen)` | Loops or trims to exactly `targetLen` samples |

---

## `stft.js`

STFT computation and canvas rendering.

```js
import {
  computeSTFT, stftToOffscreen, drawStrip, cmap,
  FFT_N, HOP, PLASMA, OCEAN, HOT,
} from './lib/stft.js';
```

| Export | Description |
|---|---|
| `FFT_N` | FFT window size (2048) |
| `HOP` | Hop size in samples (512) |
| `PLASMA` / `OCEAN` / `HOT` | Built-in colormap stop arrays |
| `computeSTFT(samples)` | Returns a `Float32Array[]` of magnitude frames (log-scaled, 0–1) |
| `stftToOffscreen(stft, stops)` | Renders STFT to an `OffscreenCanvas`; cache and invalidate on resize |
| `drawStrip(canvas, stft, stops, dpr)` | Draws directly to a visible canvas (re-renders each call) |
| `cmap(t, stops)` | Maps a 0–1 value to an `[r,g,b]` colour via a stop array |
