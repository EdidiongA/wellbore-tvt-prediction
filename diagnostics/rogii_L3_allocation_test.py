# ===== L3: does ALLOCATING the damping dose per well beat uniform, locally? =====
# All damped submissions used one S for every well. This tests dose allocation at
# MATCHED mean dose: S_w = S0 * m_w, with multipliers m_w (mean 1 across wells)
# driven by (a) branch spread from v22_oof.pkl, (b) the well's own toe deviation
# from its near-zone line (computable post-hoc on test), vs (c) uniform.
# Read the SHAPE, not the level: L1 showed the local optimum sits at lower S than
# hidden, so judge allocations against uniform AT THE SAME S0, not against base.
# WHERE: fork/gap-probe notebook (v22_oof.pkl + competition data). ~3 min.

import numpy as np, pickle, glob, os, time
import pandas as pd

_h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
D = pickle.load(open(_h[0], 'rb'))
PRED, SPREAD = D['pred'], D.get('spread', {})
DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'train')):
        DATA = c; break
assert DATA
P0 = 0.40

def line_and_pos(idx, vals):
    o = np.argsort(idx); i_s, v_s = idx[o], vals[o]
    pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
    near = pos <= P0
    a = np.polyfit(pos[near], v_s[near], 1)
    r = v_s[near] - np.polyval(a, pos[near])
    k = np.abs(r) <= 3 * max(np.std(r), 1e-6)
    if k.sum() >= 10:
        a = np.polyfit(pos[near][k], v_s[near][k], 1)
    return o, pos, np.polyval(a, pos), v_s

W, t0 = [], time.time()
for w, pv in PRED.items():
    try:
        h = pd.read_csv(os.path.join(DATA, 'train', '%s__horizontal_well.csv' % w),
                        usecols=['TVT'])
        tr = h['TVT'].values.astype(float)
        idx = np.asarray(pv['blind_idx']); that = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(tr[idx]) & np.isfinite(that)
        if m.sum() < 80:
            continue
        idx, that, tru = idx[m], that[m], tr[idx[m]]
        o, pos, line, v_s = line_and_pos(idx, that)
        dev = np.abs(v_s - line)[pos > P0].mean()      # post-hoc toe deviation
        W.append((w, o, pos, line, v_s, tru[o], dev))
    except Exception:
        pass
print('%d wells [%.0fs]' % (len(W), time.time() - t0))
base = np.mean([np.abs(v - t).mean() for _, _, _, _, v, t, _ in W])
print('baseline OOF %.4f' % base)

def rank_mult(vals, lo, hi):
    r = np.argsort(np.argsort(vals)) / max(len(vals) - 1, 1)
    m = lo + (hi - lo) * r
    return m / m.mean()                                 # exact mean-1

dev  = np.array([x[6] for x in W])
spr  = np.array([SPREAD.get(x[0], np.nan) for x in W])
spr_ok = np.isfinite(spr)
print('spread available for %.0f%% of wells' % (100 * spr_ok.mean()))

def run(S0, mult):
    tot = []
    for j, (_, o, pos, line, v_s, tru, _) in enumerate(W):
        wgt = np.clip((pos - P0) / (1 - P0), 0, 1) * S0 * mult[j]
        tot.append(np.abs((1 - wgt) * v_s + wgt * line - tru).mean())
    return np.mean(tot)

print('\n%-28s' % 'allocation (mean dose = S0)' + ''.join('%9s' % ('S0=%.2f' % s) for s in [0.05, 0.10, 0.15]))
uni = {}
for S0 in [0.05, 0.10, 0.15]:
    uni[S0] = run(S0, np.ones(len(W)))
print('%-28s' % 'uniform' + ''.join('%+9.4f' % (uni[s] - base) for s in [0.05, 0.10, 0.15]))
for name, sig, direction in [('more S to HIGH deviation', dev, 1),
                             ('more S to LOW deviation',  dev, -1),
                             ('more S to HIGH spread',    np.where(spr_ok, spr, np.nanmedian(spr)), 1),
                             ('more S to LOW spread',     np.where(spr_ok, spr, np.nanmedian(spr)), -1)]:
    mult = rank_mult(direction * sig, 0.5, 1.5)
    row = ''.join('%+9.4f' % (run(S0, mult) - base) for S0 in [0.05, 0.10, 0.15])
    print('%-28s' % name + row)

print('\nREAD: compare each allocation to UNIFORM in the same column. An allocation')
print('that beats uniform at every S0 has shape that plausibly transfers (L1 logic:')
print('levels shift between local and hidden, shapes have transferred). If nothing')
print('beats uniform, allocation is dead and v32 should ship uniform S=0.13-0.15.')
