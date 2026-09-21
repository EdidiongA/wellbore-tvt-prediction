# ===== L3: measure the interpreter-smoothing coefficient alpha, and the local
# =====     dose-response of HF-only wiggle shrinkage =====
# v31 vs v28 decomposed the damping benefit: structural shrink +0.023 (harm),
# wiggle shrink ~ -0.118 (all the gain). Hypothesis: interpreted TVT tracks only
# (1-alpha) of the wellbore's high-frequency undulation, so true u carries
# +alpha*Z_hf and the smooth-u tracker leaves an HF error minimized by shrinking
# prediction wiggles by exactly h = alpha. This cell measures alpha per well on
# training truth, and runs the HF-only damper over an h grid on the OOF preds.
# WHERE: fork or gap-probe notebook (v22_oof.pkl + competition data). ~2-3 min.

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

def _ma(x, win):
    if len(x) < win:
        return np.full_like(x, np.nan)
    k = np.ones(win) / win
    out = np.convolve(x, k, 'same')
    half = win // 2                      # edges are biased; mark invalid
    out[:half] = np.nan; out[-half:] = np.nan
    return out

WIN = 301
alphas, t0 = [], time.time()
wells_d = []
for w, pv in PRED.items():
    try:
        h = pd.read_csv(os.path.join(DATA, 'train', '%s__horizontal_well.csv' % w),
                        usecols=['TVT', 'Z'])
        tr = h['TVT'].values.astype(float); Z = h['Z'].values.astype(float)
        idx = np.asarray(pv['blind_idx']); that = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(tr[idx]) & np.isfinite(that) & np.isfinite(Z[idx])
        if m.sum() < 400:
            continue
        i2, t2, tr2, z2 = idx[m], that[m], tr[idx[m]], Z[idx[m]]
        o = np.argsort(i2); i2, t2, tr2, z2 = i2[o], t2[o], tr2[o], z2[o]
        # toe region: last 40% of the blind zone
        n = len(i2); toe = slice(int(0.6 * n), n)
        hf_T = (tr2 - _ma(tr2, WIN))[toe]
        hf_Z = (z2 - _ma(z2, WIN))[toe]
        ok = np.isfinite(hf_T) & np.isfinite(hf_Z)
        if ok.sum() < 200 or np.var(hf_Z[ok]) < 1e-6:
            continue
        slope = np.cov(hf_T[ok], hf_Z[ok])[0, 1] / np.var(hf_Z[ok])
        alphas.append(1.0 + slope)          # truth_hf = -(1-alpha) Z_hf
        wells_d.append((i2, t2, tr2))
    except Exception:
        pass
al = np.array(alphas)
print('%d wells [%.0fs]' % (len(al), time.time() - t0))
print('\nALPHA (interpreter smoothing of Z wiggles, toe region):')
print('  median %.3f | mean %.3f | p25 %.3f | p75 %.3f | frac>0.10: %.1f%%'
      % (np.median(al), al.mean(), *np.percentile(al, [25, 75]),
         100 * (al > 0.10).mean()))
print('  (alpha ~= optimal HF shrink weight h*. v28 hidden evidence implies'
      ' hidden alpha >= ~0.15.)')

# ---- local dose-response of HF-only shrinkage on OOF preds ------------------
def hf_damp(tvt, idx, P0, h, win=WIN):
    o = np.argsort(idx); i_s, v = idx[o], tvt[o]
    pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
    lp = _ma(v, win)
    fin = np.isfinite(lp)
    w = np.clip((pos - P0) / (1 - P0), 0, 1) * h
    new = v.copy()
    new[fin] = lp[fin] + (1 - w[fin]) * (v[fin] - lp[fin])
    back = np.empty_like(v); back[o] = new
    return back

base = np.mean([np.abs(t_ - tr_).mean() for _, t_, tr_ in wells_d])
print('\nbaseline OOF on these wells %.4f' % base)
print('%6s' % 'P0\\h' + ''.join('%9.2f' % s for s in [0.10, 0.15, 0.25, 0.35, 0.50]))
for P0 in [0.0, 0.40]:
    row = []
    for hh in [0.10, 0.15, 0.25, 0.35, 0.50]:
        tot = [np.abs(hf_damp(t_.copy(), i_.copy(), P0, hh) - tr_).mean()
               for i_, t_, tr_ in wells_d]
        row.append(np.mean(tot))
    print('%6.2f' % P0 + ''.join('%+9.4f' % (r - base) for r in row))
print('\nREAD: if local alpha is small and HF shrink hurts locally while helping')
print('hidden (v28 vs v31), hidden interpretations are smoother than training ones')
print('-- a second measured shift axis. If local alpha ~0.15+ and shrink helps')
print('locally, the h grid above is an honest optimizer for h*.')
