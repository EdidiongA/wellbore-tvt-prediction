# ===== L1: is the damping gain shift-specific, or does it help locally too? =====
# THE decisive free experiment after v28. Apply the exact v28 damper to v22's
# stored OOF predictions on training wells (truth known) across an (S, P0) grid.
#
# Two possible worlds:
#   A. Local damping HURTS everywhere  -> the +0.095 hidden gain is pure
#      distribution shift. The public LB is the only instrument that can see it,
#      further sweeps = overfitting the public subset. STOP at v28; lock finals.
#   B. Local damping HELPS in some region -> the mechanism is universal and the
#      local surface is an honest optimizer: take its argmin as ONE final probe.
#
# Either answer is worth a section in the paper. WHERE: v22 OOF fork (needs
# v22_oof.pkl pred + competition data). CPU, ~2 min. No submission.

import numpy as np, pickle, glob, os, time
import pandas as pd

_h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
D = pickle.load(open(_h[0], 'rb'))
PRED = D['pred']

DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'train')):
        DATA = c; break
assert DATA

def damp(tvt, idx, P0, S):
    o = np.argsort(idx); idx_s, tvt_s = idx[o], tvt[o]
    pos = (idx_s - idx_s.min()) / max(idx_s.max() - idx_s.min(), 1)
    near = pos <= P0
    if near.sum() < 20:
        return tvt
    a = np.polyfit(pos[near], tvt_s[near], 1)
    r = tvt_s[near] - np.polyval(a, pos[near])
    k = np.abs(r) <= 3 * max(np.std(r), 1e-6)
    if k.sum() >= 10:
        a = np.polyfit(pos[near][k], tvt_s[near][k], 1)
    line = np.polyval(a, pos)
    w = np.clip((pos - P0) / (1 - P0), 0, 1) * S
    new = (1 - w) * tvt_s + w * line
    back = np.empty_like(tvt_s); back[o] = new
    return back

# preload wells once
wells_data, t0 = [], time.time()
for w, pv in PRED.items():
    try:
        h = pd.read_csv(os.path.join(DATA, 'train', '%s__horizontal_well.csv' % w),
                        usecols=['TVT'])
        tr = h['TVT'].values.astype(float)
        idx = np.asarray(pv['blind_idx']); that = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(tr[idx]) & np.isfinite(that)
        if m.sum() < 80:
            continue
        wells_data.append((idx[m], that[m], tr[idx[m]]))
    except Exception:
        pass
print('%d wells loaded [%.0fs]' % (len(wells_data), time.time() - t0))

base = np.mean([np.abs(t_ - tr_).mean() for _, t_, tr_ in wells_data])
print('baseline OOF MAE %.4f\n' % base)

S_GRID  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
P0_GRID = [0.25, 0.40, 0.55]
print('%6s' % 'P0\\S' + ''.join('%9.2f' % s for s in S_GRID))
best = (0, 0, base)
for P0 in P0_GRID:
    row = []
    for S in S_GRID:
        tot = [np.abs(damp(t_.copy(), i_.copy(), P0, S) - tr_).mean()
               for i_, t_, tr_ in wells_data]
        m = np.mean(tot); row.append(m)
        if m < best[2]:
            best = (P0, S, m)
    print('%6.2f' % P0 + ''.join('%+9.4f' % (r - base) for r in row))

print('\nlocal argmin: P0=%.2f S=%.2f  ->  OOF %.4f (delta %+.4f)'
      % (best[0], best[1], best[2], best[2] - base))
print('=' * 64)
if best[1] == 0.0 or best[2] > base - 0.01:
    print('WORLD A: damping does NOT help locally. The +0.095 hidden gain is')
    print('  distribution shift, directly measured — the same perturbation that is')
    print('  neutral-to-harmful on training wells helps hidden wells. Further')
    print('  S/P0 sweeps on the public LB would be subset overfitting. STOP at')
    print('  v28; lock finals v28 + v22; this contrast goes in the paper.')
else:
    print('WORLD B: damping helps locally too (delta %+.3f at P0=%.2f S=%.2f).'
          % (best[2] - base, best[0], best[1]))
    print('  The local surface is an honest optimizer. ONE final probe at the')
    print('  local argmin is justified — then lock finals.')
print('=' * 64)
