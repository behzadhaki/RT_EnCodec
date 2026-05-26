"""Generate rt_encodec architecture diagrams — one per model."""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

_HERE = os.path.dirname(os.path.abspath(__file__))

FRAME_COLORS = ['#F5C518', '#5B9BD5', '#9B6BB5']
STAGE_FC     = ['#E8F4E8', '#E8EEF8', '#FFF0E8', '#E8F4E8']
STAGE_EC     = ['#2A7A2A', '#1A5C8A', '#8A4010', '#2A7A2A']

STAGE_NAMES = [
    'encode_audio_segment',
    'quantize_encodings',
    'decode_codes',
    'decode_audio',
]

COL_X  = [2.5, 7.5, 12.5]
BOX_W  = 3.4
BOX_H  = 0.70
GAP    = 0.22   # gap between box edge and dashed arrow

Y_HEADER = 18.6
Y_STAGE  = [17.25, 15.0, 12.75, 10.5]
Y_OUTPUT = 9.2
Y_FOOTER = 7.85   # top of overlap-add box (48 kHz only)
FOOTER_H = 0.62


def rbox(ax, cx, cy, w, h, label, fc, ec, fs=9.0, bold=True):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle='round,pad=0.10',
        facecolor=fc, edgecolor=ec, linewidth=1.6, zorder=2))
    ax.text(cx, cy, label, ha='center', va='center',
            fontsize=fs, weight='bold' if bold else 'normal',
            color='#111', zorder=3)


def varr(ax, cx, y_top, y_bot, shape_lbl='', var_lbl=''):
    ax.annotate('', xy=(cx, y_bot + 0.06), xytext=(cx, y_top - 0.06),
                arrowprops=dict(arrowstyle='->', color='#444', lw=1.4), zorder=4)
    if shape_lbl:
        ax.text(cx + 0.15, (y_top + y_bot)/2 + 0.10,
                shape_lbl, ha='left', va='center', fontsize=7.5, color='#222', zorder=5)
        ax.text(cx + 0.15, (y_top + y_bot)/2 - 0.13,
                var_lbl,   ha='left', va='center', fontsize=6.5,
                color='#777', style='italic', zorder=5)


def harr_dashed(ax, x1, x2, y, label):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color='#999', lw=1.1,
                                linestyle='dashed'))
    ax.text((x1 + x2)/2, y + 0.17, label,
            ha='center', va='bottom', fontsize=6.5, color='#999', style='italic')


def draw_model(title, subtitle, arrow_data, show_footer, note_lines, outpath):
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 20.5)
    ax.axis('off')

    # title
    ax.text(7.5, 20.1, title,
            ha='center', va='center', fontsize=12, weight='bold')
    ax.text(7.5, 19.55, subtitle,
            ha='center', va='center', fontsize=8.5, color='#555')

    # column headers
    for ci, (cx, fc) in enumerate(zip(COL_X, FRAME_COLORS)):
        rbox(ax, cx, Y_HEADER, BOX_W, 0.60,
             f'Chunk {["Zero","One","Two"][ci]}',
             fc=fc, ec=fc, fs=10)

    # per-column stages
    for ci, cx in enumerate(COL_X):
        show = (ci == 0)

        # header → stage 0
        s, v = arrow_data[0]
        varr(ax, cx, Y_HEADER - 0.30, Y_STAGE[0] + BOX_H/2,
             s if show else '', v if show else '')

        for si, (sy, sname, sfc, sec) in enumerate(
                zip(Y_STAGE, STAGE_NAMES, STAGE_FC, STAGE_EC)):
            rbox(ax, cx, sy, BOX_W, BOX_H, sname, fc=sfc, ec=sec, fs=8.5)

            s, v = arrow_data[si + 1]
            if si < len(Y_STAGE) - 1:
                varr(ax, cx, sy - BOX_H/2, Y_STAGE[si+1] + BOX_H/2,
                     s if show else '', v if show else '')
            else:
                varr(ax, cx, sy - BOX_H/2, Y_OUTPUT + 0.30,
                     s if show else '', v if show else '')

        # output box
        fc = FRAME_COLORS[ci]
        rbox(ax, cx, Y_OUTPUT, BOX_W, 0.60,
             f'Chunk {["Zero","One","Two"][ci]} Reconstructed',
             fc=fc + '55', ec=fc, fs=9)

        # arrow output → footer or bottom
        y_arrow_end = Y_FOOTER + FOOTER_H + 0.02 if show_footer else Y_OUTPUT - 0.72
        ax.annotate('', xy=(cx, y_arrow_end),
                    xytext=(cx, Y_OUTPUT - 0.30 - 0.04),
                    arrowprops=dict(arrowstyle='->', color='#AA8800' if show_footer else '#444',
                                   lw=1.1), zorder=4)

    # horizontal dashed LSTM state arrows
    for x_from, x_to in [(COL_X[0], COL_X[1]), (COL_X[1], COL_X[2])]:
        x1 = x_from + BOX_W/2 + GAP
        x2 = x_to   - BOX_W/2 - GAP
        harr_dashed(ax, x1, x2, Y_STAGE[0], 'enc_lstm_state  (optional)')
        harr_dashed(ax, x1, x2, Y_STAGE[3], 'dec_lstm_state  (optional)')

    # overlap-add footer (48 kHz only)
    if show_footer:
        ax.add_patch(FancyBboxPatch(
            (0.5, Y_FOOTER), 14.0, FOOTER_H,
            boxstyle='round,pad=0.10',
            facecolor='#FFF8E8', edgecolor='#AA8800', linewidth=1.2, zorder=2))
        ax.text(7.5, Y_FOOTER + FOOTER_H/2,
                '_linear_overlap_add(frames, segment_stride)'
                '   [triangle crossfade, 10 ms overlap, ~1 s latency]',
                ha='center', va='center', fontsize=8, color='#664400', zorder=3)

    # notes
    note_y0 = (Y_FOOTER - 0.55) if show_footer else (Y_OUTPUT - 1.0)
    for i, line in enumerate(note_lines):
        ax.text(0.6, note_y0 - i * 0.45, line,
                ha='left', va='center', fontsize=7.5, color='#333')

    plt.tight_layout(pad=0)
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved {outpath}')


# ── 24 kHz ────────────────────────────────────────────────────────────────────
draw_model(
    title    = 'encodec_24khz — causal · mono · 24 kHz',
    subtitle = 'Inherently mono — stereo encoded via parallel processing (L and R as independent streams)',
    arrow_data = [
        ('[B, 1, T_audio]',   'B=1 mono  or  B=2 stereo-as-batch,  T_audio=samples'),
        ('[B, D, T_frames]',  'D=128 (embedding),  T_frames=T_audio//320'),
        ('[B, K, T_frames]',  'K=active codebooks ≤ 32  (int64)'),
        ('[B, D, T_frames]',  'D=128'),
        ('[B, 1, T_audio]',   ''),
    ],
    show_footer = False,
    note_lines  = [
        'causal=True · C=1 · D=128 · n_q=32 · hop=320 · frame_rate=75 Hz · segment=None',
        'Streaming chunk: 320 samples ≈ 13.3 ms  ·  LSTM state carried across chunks via enc/dec_lstm_state',
        'K (active codebooks) = ⌊bandwidth × 1000 / (frame_rate × 10)⌋',
        '  1.5 kbps→K=2 · 3 kbps→K=4 · 6 kbps→K=8 · 12 kbps→K=16 · 24 kbps→K=32',
    ],
    outpath = os.path.join(_HERE, 'architecture_24khz.png'),
)

# ── 48 kHz ────────────────────────────────────────────────────────────────────
draw_model(
    title    = 'encodec_48khz — non-causal · stereo · 48 kHz',
    subtitle = 'Native stereo: both channels processed jointly; C=2 absorbed into embedding D=128',
    arrow_data = [
        ('[1, 2, T_audio]',   'C=2 stereo,  T_audio=samples'),
        ('[1, D, T_frames]',  'D=128 (embedding, C absorbed),  T_frames=T_audio//320'),
        ('[1, K, T_frames]',  'K=active codebooks ≤ 16  (int64)'),
        ('[1, D, T_frames]',  'D=128'),
        ('[1, 2, T_audio]',   ''),
    ],
    show_footer = True,
    note_lines  = [
        'causal=False · C=2 · D=128 · n_q=16 · hop=320 · frame_rate=150 Hz · segment=1.0 s · stride=0.99 s',
        'Streaming latency: ~1 s (must buffer full 1 s segment before encoding)',
        'K (active codebooks) = ⌊bandwidth × 1000 / (frame_rate × 10)⌋',
        '  3 kbps→K=2 · 6 kbps→K=4 · 12 kbps→K=8 · 24 kbps→K=16',
    ],
    outpath = os.path.join(_HERE, 'architecture_48khz.png'),
)
