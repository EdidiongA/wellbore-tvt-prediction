# ===== v32: decomposed damper -- HF wiggle shrink, structural shrink zeroed =====
# The v31/v28 ablation showed v28's entire gain came from shrinking toe wiggles
# (structural-only damping HURT: v31 8.936 vs v22 8.913). This applies wiggle
# shrinkage alone: tvt -> lowpass + (1-w)*(tvt - lowpass), w ramped to H at toe.
# Additive prediction from the ablation: ~8.795 on the pure-v22 base.
# Placement: LAST cell of a sealed-v22 copy, after the submission cell.
# If v29=8.23 is CONFIRMED, paste this after v29's damping cell instead (base
# change), and expect the interaction to be unknown -- flag it as such.
import numpy as np, pandas as pd

P0, H, WIN = 0.40, 0.15, 301     # H: max HF shrink at toe; WIN: lowpass window (stations)
MIN_STA = 2 * WIN

def _ma(x, win):
    k = np.ones(win) / win
    out = np.convolve(x, k, 'same')
    half = win // 2
    out[:half] = np.nan; out[-half:] = np.nan
    return out

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)
print('submission: %d rows, %d wells' % (len(sub), sub['well'].nunique()))

changed, skipped, shifts = 0, 0, []
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
    wgt = np.clip((pos - P0) / (1 - P0), 0, 1) * H
    new = v.copy()
    new[fin] = lp[fin] + (1 - wgt[fin]) * (v[fin] - lp[fin])
    if not np.isfinite(new).all():
        skipped += 1; continue
    back = np.empty_like(v); back[o] = new
    sub.loc[grp.index, 'tvt'] = back
    changed += 1
    shifts.append(float(np.abs(new - v).max()))

print('HF-damped %d wells | skipped %d | median max|shift| %.2f ft'
      % (changed, skipped, np.median(shifts) if shifts else 0.0))
assert sub['tvt'].notna().all()
sub[['id', 'tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (v32 = HF-only toe shrink, P0=%.2f H=%.2f WIN=%d)'
      % (P0, H, WIN))
