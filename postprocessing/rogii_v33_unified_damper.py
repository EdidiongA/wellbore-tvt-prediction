# ===== UNIFIED DAMPER (v32+) -- one cell, preset-driven, bold-search ready =====
# Decomposed damper with independent wiggle (HF) and structural (LO) shrink, flat
# or ramped application, tunable lowpass scale. Flip PRESET per submission.
# Placement: LAST cell of a sealed-v22 copy (or after v29's cells if 8.23 confirmed).
import numpy as np, pandas as pd

PRESET = 'v33_flat25'        # <-- change this string per submission

PRESETS = {
  # name            mode    P0    H_hf  S_lo  WIN   rationale
  'v32_anchor':   ('ramp',  0.40, 0.15, 0.00, 301), # clean decomposition anchor (~8.795 pred)
  'v33_flat25':   ('flat',  0.00, 0.25, 0.00, 301), # STRONG FORM: whole-zone shrink at bolder h
  'v34_flat40':   ('flat',  0.00, 0.40, 0.00, 301), # aggressive: alpha_hidden ~0.4 hypothesis
  'v35_widewin':  ('flat',  0.00, 0.25, 0.00, 601), # coarser smoothing scale (mid-freq too)
  'v36_combo':    ('flat',  0.00, 0.25, 0.08, 301), # + small structural (only if L4 supports)
}
mode, P0, H, S_lo, WIN = PRESETS[PRESET]
MIN_STA = 2 * WIN

def _ma(x, win):
    out = np.convolve(x, np.ones(win) / win, 'same')
    h = win // 2
    out[:h] = np.nan; out[-h:] = np.nan
    return out

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)
print('preset %s: mode=%s P0=%.2f H=%.2f S_lo=%.2f WIN=%d'
      % (PRESET, mode, P0, H, S_lo, WIN))

changed, skipped, mx = 0, 0, []
for w, grp in sub.groupby('well'):
    idx = grp['ridx'].values
    tvt = grp['tvt'].values.astype(float)
    if len(idx) < MIN_STA or not np.isfinite(tvt).all():
        skipped += 1; continue
    o = np.argsort(idx); i_s, v = idx[o], tvt[o]
    pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
    lp = _ma(v, WIN)
    fin = np.isfinite(lp)
    if fin.sum() < 50:
        skipped += 1; continue
    if mode == 'flat':
        w_hf = np.full(len(v), H)
    else:
        w_hf = np.clip((pos - P0) / max(1 - P0, 1e-9), 0, 1) * H
    new = v.copy()
    new[fin] = lp[fin] + (1 - w_hf[fin]) * (v[fin] - lp[fin])
    if S_lo > 0:                              # optional structural shrink of the lowpass
        near = pos <= max(P0, 0.40)
        if near.sum() >= 20:
            a = np.polyfit(pos[near], v[near], 1)
            line = np.polyval(a, pos)
            w_lo = (np.clip((pos - 0.40) / 0.60, 0, 1) * S_lo)
            new[fin] = (1 - w_lo[fin]) * new[fin] + w_lo[fin] * (
                line[fin] + (1 - w_hf[fin]) * 0.0)
    if not np.isfinite(new).all():
        skipped += 1; continue
    back = np.empty_like(v); back[o] = new
    sub.loc[grp.index, 'tvt'] = back
    changed += 1
    mx.append(float(np.abs(new - v).max()))

print('damped %d wells | skipped %d | median max|shift| %.2f ft'
      % (changed, skipped, np.median(mx) if mx else 0.0))
assert sub['tvt'].notna().all()
sub[['id', 'tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (%s)' % PRESET)
