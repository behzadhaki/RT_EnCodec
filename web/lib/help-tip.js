// Shared hover-help tooltip.
//
// One tooltip element + one delegated listener for the whole page — no
// per-element wiring, and dynamically created nodes (cache entries, library
// rows, level menu) work automatically.
//
// Text resolution, walking up from the hovered element:
//   1. a `data-help` attribute on the element or any ancestor
//   2. the first registry entry whose CSS selector matches the element
// Registry: { '<css-selector>': 'help text', ... } — ids ('#umapWindow') and
// widget classes ('.src-vol') both work.
export function initHelpTips(registry = {}) {
  const entries = Object.entries(registry);

  const tip = document.createElement('div');
  tip.id = 'helpTip';
  document.body.appendChild(tip);

  let showTimer = null;
  let visible   = false;
  let currentEl = null;

  function lookup(start) {
    for (let el = start; el && el !== document.body; el = el.parentElement) {
      if (el.dataset && el.dataset.help) return { el, text: el.dataset.help };
      for (const [sel, text] of entries) {
        try { if (el.matches(sel)) return { el, text }; } catch (_) { /* bad selector */ }
      }
    }
    return null;
  }

  function place(anchor) {
    const pad = 8;
    const a   = anchor.getBoundingClientRect();
    const r   = tip.getBoundingClientRect();
    let left = a.left;
    let top  = a.bottom + 6;
    if (left + r.width > innerWidth - pad) left = innerWidth - r.width - pad;
    if (left < pad) left = pad;
    if (top + r.height > innerHeight - pad) top = a.top - r.height - 6; // flip above
    if (top < pad) top = pad;
    tip.style.left = left + 'px';
    tip.style.top  = top + 'px';
  }

  function hide() {
    clearTimeout(showTimer);
    showTimer = null;
    currentEl = null;
    if (visible) { tip.classList.remove('visible'); visible = false; }
  }

  function show(hit) {
    tip.textContent = hit.text;
    tip.classList.add('visible');
    visible = true;
    place(hit.el);
  }

  document.addEventListener('mouseover', e => {
    const hit = lookup(e.target);
    if (!hit) { hide(); return; }
    if (hit.el === currentEl) return;
    clearTimeout(showTimer);
    currentEl = hit.el;
    // First tip waits; moving between controls while a tip is up is near-instant.
    showTimer = setTimeout(() => show(hit), visible ? 80 : 450);
  });

  document.addEventListener('mouseout', e => {
    if (currentEl && !currentEl.contains(e.relatedTarget)) {
      // Keep `visible` latched briefly so the next control shows fast.
      clearTimeout(showTimer);
      showTimer = setTimeout(() => { if (!currentEl) return; }, 0);
      const wasVisible = visible;
      tip.classList.remove('visible');
      visible = false;
      currentEl = null;
      if (wasVisible) {
        visible = true;                    // latch for fast re-show…
        setTimeout(() => { if (!currentEl) visible = false; }, 350); // …then expire
      }
    }
  });

  // Any interaction dismisses the tip immediately.
  ['mousedown', 'wheel'].forEach(t =>
    document.addEventListener(t, hide, { passive: true, capture: true }));
  window.addEventListener('scroll', hide, { passive: true, capture: true });
  window.addEventListener('blur', hide);
}
