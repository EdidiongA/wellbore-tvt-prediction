# ===== v32: v28's exact damper + per-well dose allocation (mean dose preserved) =====
# Identical to v28 (tvt-space, P0=0.40, ramp-to-toe) except S varies per well:
# S_w = S0 * m_w with rank-based multipliers in [LO, HI], normalized to mean 1,
# driven by the well's post-hoc toe deviation from its near-zone line.
# SET ALLOC_DIRECTION from L3's local result before submitting:
#   +1 -> more damping where the toe has curved FURTHER from the near trend
#   -1 -> more damping where it has curved LESS
#    0 -> uniform (v32 degenerates to v28 with S0 below)
# Placement: LAST cell of a sealed-v22 copy, after the submission cell.
import numpy as np, pandas as pd

S0, P0 = 0.15, 0.40
ALLOC_DIRECTION = +1        # <-- set from L3 (or 0 for uniform)
LO, HI = 0.6, 1.4           # multiplier range before mean-1 normalization
MIN_STA = 80

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)
print('submission: %d rows, %d wells' % (len(sub), sub['well'].nunique()))

# pass 1: fit lines, collect deviations
fits, devs = {}, {}
for w, grp in sub.groupby('well'):
    idx = grp['ridx'].values; tvt = grp['tvt'].values.astype(float)
    if len(idx) < MIN_STA or not np.isfinite(tvt).all():
        continue
    o = np.argsort(idx); i_s, v_s = idx[o], tvt[o]
    pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
    near = pos <= P0
    if near.sum() < 20:
        continue
    a = np.polyfit(pos[near], v_s[near], 1)
    r = v_s[near] - np.polyval(a, pos[near])
    k = np.abs(r) <= 3 * max(np.std(r), 1e-6)
    if k.sum() >= 10:
        a = np.polyfit(pos[near][k], v_s[near][k], 1)
    line = np.polyval(a, pos)
    fits[w] = (o, pos, line, v_s, grp.index)
    devs[w] = float(np.abs(v_s - line)[pos > P0].mean())

wells = list(fits)
dv = np.array([devs[w] for w in wells])
if ALLOC_DIRECTION == 0 or len(wells) < 10:
    mult = {w: 1.0 for w in wells}
else:
    r = np.argsort(np.argsort(ALLOC_DIRECTION * dv)) / max(len(wells) - 1, 1)
    m = LO + (HI - LO) * r
    m = m / m.mean()
    mult = dict(zip(wells, m))
print('allocation: direction %+d | multiplier range [%.2f, %.2f] | mean %.3f'
      % (ALLOC_DIRECTION, min(mult.values()), max(mult.values()),
         np.mean(list(mult.values()))))

# pass 2: damp
changed, toe = 0, []
for w in wells:
    o, pos, line, v_s, gidx = fits[w]
    wgt = np.clip((pos - P0) / (1 - P0), 0, 1) * S0 * mult[w]
    new = (1 - wgt) * v_s + wgt * line
    back = np.empty_like(v_s); back[o] = new
    sub.loc[gidx, 'tvt'] = back
    changed += 1; toe.append(abs(new[-1] - v_s[-1]))
print('damped %d wells | median |toe shift| %.2f ft' % (changed, np.median(toe)))
assert sub['tvt'].notna().all()
sub[['id', 'tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (v32: adaptive damping, S0=%.2f, dir %+d)'
      % (S0, ALLOC_DIRECTION))
