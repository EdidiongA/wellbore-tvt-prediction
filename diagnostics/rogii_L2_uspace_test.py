# ===== L2: tvt-space vs u-space damping, local grid =====
# The shipped damper fits/damps in TVT space, which shrinks the exactly-known
# Z wiggles at the toe. The u-space damper (u = tvt + Z) preserves them and
# damps only structural curvature. This compares both over the (P0, S) grid on
# v22's OOF predictions. If u-space dominates (it should, by construction),
# v31 ships the u-space damper at the measured hidden optimum.
# WHERE: fork or gap-probe notebook (needs v22_oof.pkl + competition data). ~2 min.

import numpy as np, pickle, glob, os, time
import pandas as pd

_h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
PRED = pickle.load(open(_h[0], 'rb'))['pred']
DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'train')):
        DATA = c; break
assert DATA

def damp_arr(vals, idx, P0, S):
    """Generic: fit line on near zone of vals vs pos, ramp-shrink toward it."""
    o = np.argsort(idx); i_s, v_s = idx[o], vals[o]
    pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
    near = pos <= P0
    if near.sum() < 20:
        return vals
    a = np.polyfit(pos[near], v_s[near], 1)
    r = v_s[near] - np.polyval(a, pos[near])
    k = np.abs(r) <= 3 * max(np.std(r), 1e-6)
    if k.sum() >= 10:
        a = np.polyfit(pos[near][k], v_s[near][k], 1)
    w = np.clip((pos - P0) / (1 - P0), 0, 1) * S
    new = (1 - w) * v_s + w * np.polyval(a, pos)
    back = np.empty_like(v_s); back[o] = new
    return back

W, t0 = [], time.time()
for w, pv in PRED.items():
    try:
        h = pd.read_csv(os.path.join(DATA, 'train', '%s__horizontal_well.csv' % w),
                        usecols=['TVT', 'Z'])
        tr = h['TVT'].values.astype(float); Z = h['Z'].values.astype(float)
        idx = np.asarray(pv['blind_idx']); that = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(tr[idx]) & np.isfinite(that) & np.isfinite(Z[idx])
        if m.sum() < 80:
            continue
        W.append((idx[m], that[m], tr[idx[m]], Z[idx[m]]))
    except Exception:
        pass
print('%d wells [%.0fs]' % (len(W), time.time() - t0))
base = np.mean([np.abs(t_ - tr_).mean() for _, t_, tr_, _ in W])
print('baseline OOF %.4f\n' % base)

S_GRID  = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35]
P0_GRID = [0.40, 0.55]
for label, use_u in [('TVT-space (shipped)', False), ('u-space (v31)', True)]:
    print('--- %s ---' % label)
    print('%6s' % 'P0\\S' + ''.join('%9.2f' % s for s in S_GRID))
    for P0 in P0_GRID:
        row = []
        for S in S_GRID:
            tot = []
            for i_, t_, tr_, z_ in W:
                if use_u:
                    u = t_ + z_
                    u2 = damp_arr(u.copy(), i_.copy(), P0, S)
                    tot.append(np.abs((u2 - z_) - tr_).mean())
                else:
                    t2 = damp_arr(t_.copy(), i_.copy(), P0, S)
                    tot.append(np.abs(t2 - tr_).mean())
            row.append(np.mean(tot))
        print('%6.2f' % P0 + ''.join('%+9.4f' % (r - base) for r in row))
    print()

print('READ: if u-space deltas <= tvt-space deltas across the grid (they should be,')
print('by construction), the wiggle-damage term is real and v31 = u-space damper at')
print('the hidden-measured S is the mechanism-correct upgrade. The DIFFERENCE between')
print('the two grids at (0.40, 0.15) estimates the free gain v31 adds over v28.')
