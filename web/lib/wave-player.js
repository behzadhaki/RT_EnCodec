// Self-contained waveform player for one audio buffer.
// onFrameCb(pos) called each animation frame while playing (pos=0..1, pos=-1 when stopped).
// audio: { getAudioCtx, getMasterDest } — injected so players share the app's AudioContext.

export function makeWavePlayer(canvasId, waveColor, { getAudioCtx, getMasterDest }, onFrameCb) {
  const cvs  = document.getElementById(canvasId);
  const c2   = cvs.getContext('2d');
  const pdpr = devicePixelRatio || 1;
  const XFADE_S = 0.04;

  let audio   = null;   // Float32Array
  let sr      = 44100;
  let peaks   = null;   // [min,max] per pixel col
  let looping = false;
  let playing = false;
  let srcNode = null, gainNode = null;
  let playStart  = 0;
  let playOffset = 0;
  let animId  = null;
  let cursorPos = -1;
  let playBtnEl = null, loopBtnEl = null;

  const dur      = () => audio ? audio.length / sr : 0;
  const elapsed  = () => playing ? playOffset + (getAudioCtx().currentTime - playStart) : playOffset;
  const position = () => dur() > 0 ? Math.min(1, elapsed() / dur()) : 0;

  function buildPeaks() {
    if (!audio || !cvs.width) { peaks = null; return; }
    const W = cvs.width;
    peaks = new Float32Array(W * 2);
    for (let px = 0; px < W; px++) {
      const s0 = Math.floor(px / W * audio.length);
      const s1 = Math.max(s0 + 1, Math.floor((px + 1) / W * audio.length));
      let mn = 0, mx = 0;
      for (let s = s0; s < s1 && s < audio.length; s++) {
        const v = audio[s]; if (v < mn) mn = v; if (v > mx) mx = v;
      }
      peaks[px * 2] = mn; peaks[px * 2 + 1] = mx;
    }
  }

  function draw() {
    const W = cvs.width, H = cvs.height;
    c2.clearRect(0, 0, W, H);

    if (!peaks) {
      c2.fillStyle = '#444';
      c2.font = `${9 * pdpr}px system-ui`;
      c2.textAlign = 'center';
      c2.fillText('no audio', W / 2, H / 2 + 3);
      return;
    }

    const cy = H / 2;
    c2.fillStyle = waveColor;
    c2.globalAlpha = 0.5;
    for (let px = 0; px < W; px++) {
      const mn = peaks[px * 2], mx = peaks[px * 2 + 1];
      c2.fillRect(px, cy - mx * cy, 1, Math.max(1, (mx - mn) * cy));
    }
    c2.globalAlpha = 1;

    if (cursorPos >= 0) {
      const cx = cursorPos * W;
      c2.strokeStyle = '#ffffff66';
      c2.lineWidth = 1 * pdpr;
      c2.beginPath(); c2.moveTo(cx, 0); c2.lineTo(cx, H); c2.stroke();
    }

    if (playing || playOffset > 0) {
      const px = position() * W;
      c2.strokeStyle = '#ffffff';
      c2.lineWidth = 1.5 * pdpr;
      c2.beginPath(); c2.moveTo(px, 0); c2.lineTo(px, H); c2.stroke();
    }
  }

  function updatePlayBtn() {
    if (!playBtnEl) return;
    playBtnEl.textContent = playing ? '⏸' : '▶';
    playBtnEl.classList.toggle('active', playing);
  }

  function killSrc() {
    if (srcNode)  { srcNode.onended = null; try { srcNode.stop(); } catch (_) {} srcNode = null; }
    if (gainNode) { try { gainNode.disconnect(); } catch (_) {} gainNode = null; }
  }

  function doPlay(offset, xfade = false) {
    if (!audio) return;
    const ctx = getAudioCtx();
    const now = ctx.currentTime;

    const oldSrc  = srcNode;
    const oldGain = gainNode;
    if (oldSrc) oldSrc.onended = null;

    playOffset = Math.max(0, Math.min(offset, dur() - 0.001));
    playStart  = now;

    const newGain = ctx.createGain();
    newGain.connect(getMasterDest());

    const buf = ctx.createBuffer(1, audio.length, sr);
    buf.copyToChannel(audio, 0);
    const newSrc = ctx.createBufferSource();
    newSrc.buffer = buf;
    newSrc.connect(newGain);

    if (xfade && oldSrc && oldGain) {
      oldGain.gain.cancelScheduledValues(now);
      oldGain.gain.setValueAtTime(oldGain.gain.value, now);
      oldGain.gain.linearRampToValueAtTime(0, now + XFADE_S);
      newGain.gain.setValueAtTime(0, now);
      newGain.gain.linearRampToValueAtTime(1, now + XFADE_S);
      oldSrc.stop(now + XFADE_S + 0.01);
      oldSrc.onended = () => { try { oldGain.disconnect(); } catch (_) {} };
    } else {
      if (oldSrc)  { try { oldSrc.stop();       } catch (_) {} }
      if (oldGain) { try { oldGain.disconnect(); } catch (_) {} }
      newGain.gain.value = 1;
    }

    newSrc.start(now, playOffset);

    newSrc.onended = () => {
      if (srcNode !== newSrc) return;
      playing = false; updatePlayBtn();
      if (looping) {
        playOffset = 0; doPlay(0);
      } else {
        srcNode = null; gainNode = null;
        playOffset = 0; cancelAnimationFrame(animId); draw(); onFrameCb?.(-1);
      }
    };

    srcNode  = newSrc;
    gainNode = newGain;
    playing  = true; updatePlayBtn();
    cancelAnimationFrame(animId);
    (function anim() { draw(); onFrameCb?.(position()); if (playing) animId = requestAnimationFrame(anim); })();
  }

  cvs.addEventListener('click', e => {
    const rect = cvs.getBoundingClientRect();
    const pos  = (e.clientX - rect.left) / rect.width;
    const t    = pos * dur();
    if (playing) doPlay(t, true); else { playOffset = t; draw(); }
  });

  const ro = new ResizeObserver(() => {
    const r = cvs.getBoundingClientRect();
    if (!r.width) return;
    cvs.width = r.width * pdpr; cvs.height = r.height * pdpr;
    buildPeaks(); draw();
  });
  ro.observe(cvs);

  return {
    setAudio(f32, sampleRate) {
      const wasPlaying = playing;
      const wasOffset  = playing ? elapsed() : 0;
      audio = f32; sr = sampleRate;
      buildPeaks();
      if (wasPlaying) {
        doPlay(wasOffset, true);
      } else {
        playOffset = 0; draw();
      }
      updatePlayBtn();
    },
    play() {
      if (playing) {
        playOffset = elapsed();
        killSrc();
        playing = false; cancelAnimationFrame(animId);
        updatePlayBtn(); draw();
      } else {
        doPlay(playOffset);
      }
    },
    stop() {
      killSrc();
      playing = false; playOffset = 0;
      cancelAnimationFrame(animId); updatePlayBtn(); draw(); onFrameCb?.(-1);
    },
    setLoop(on) { looping = on; loopBtnEl?.classList.toggle('active', on); },
    setCursor(pos) { cursorPos = pos; draw(); },
    setPlayBtn(el) { playBtnEl = el; updatePlayBtn(); },
    setLoopBtn(el) { loopBtnEl = el; },
    isPlaying: () => playing,
    getPosition: position,
  };
}
