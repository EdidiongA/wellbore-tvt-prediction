# ===== L7: does a well's KNOWN-zone alpha predict its BLIND-zone alpha? =====
# If yes, per-well HF dosing (h_i = f(alpha_known_i)) beats any global h by
# construction, and the mapping f is calibrated here. FREE. ~3 min.
# WHERE: the L1/L3 notebook (v22_oof.pkl + competition data).
import numpy as np, pickle, glob, os, time
import pandas as pd
from scipy.stats import spearmanr

_h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
PRED = pickle.load(open(_h[0], 'rb'))['pred']
DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'train')): DATA = c; break
assert DATA
WIN = 301

def _ma(x, win):
    out = np.convolve(x, np.ones(win)/win, 'same'); h = win//2
    out[:h] = np.nan; out[-h:] = np.nan
    return out

def _alpha(t, z):
    hf_t = t - _ma(t, WIN); hf_z = z - _ma(z, WIN)
    ok = np.isfinite(hf_t) & np.isfinite(hf_z)
    if ok.sum() < 250 or np.var(hf_z[ok]) < 1e-6: return np.nan
    return 1.0 + np.cov(hf_t[ok], hf_z[ok])[0,1] / np.var(hf_z[ok])

rows, t0 = [], time.time()
for w, pv in PRED.items():
    try:
        H = pd.read_csv(os.path.join(DATA,'train','%s__horizontal_well.csv'%w),
                        usecols=['TVT','TVT_input','Z'])
        tr = H['TVT'].values.astype(float); Z = H['Z'].values.astype(float)
        tin = H['TVT_input'].values.astype(float)
        kn = np.isfinite(tin)
        if kn.sum() < 700: continue
        a_known = _alpha(tin[kn], Z[kn])
        idx = np.asarray(pv['blind_idx']); that = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(tr[idx]) & np.isfinite(that)
        if m.sum() < 700: continue
        o = np.argsort(idx[m]); i2 = idx[m][o]
        a_blind = _alpha(tr[i2], Z[i2])
        if np.isfinite(a_known) and np.isfinite(a_blind):
            rows.append((w, a_known, a_blind, i2, that[m][o], tr[i2], Z[i2]))
    except Exception: pass
ak = np.array([r[1] for r in rows]); ab = np.array([r[2] for r in rows])
print('%d wells [%.0fs]' % (len(rows), time.time()-t0))
print('alpha_known:  median %.3f  p25 %.3f  p75 %.3f' % (np.median(ak),*np.percentile(ak,[25,75])))
print('alpha_blind:  median %.3f  p25 %.3f  p75 %.3f' % (np.median(ab),*np.percentile(ab,[25,75])))
rho, p = spearmanr(ak, ab)
print('\nTRANSFER: spearman(alpha_known, alpha_blind) = %+.3f (p=%.1e)' % (rho, p))

# per-well HF dosing on OOF: h_i = clip(A + B*alpha_known_i, 0.10, 0.80), vs global
def hf_damp(v, h):
    lp = _ma(v, 601); fin = np.isfinite(lp)
    out = v.copy(); out[fin] = lp[fin] + (1-h)*(v[fin]-lp[fin]); return out
base = np.mean([np.abs(r[4]-r[5]).mean() for r in rows])
print('\nbaseline OOF %.4f' % base)
print('%-28s %10s' % ('dosing rule', 'delta'))
for lab, hs in [('global h=0.15', [0.15]*len(rows)),
                ('global h=0.46', [0.46]*len(rows)),
                ('h_i = clip(alpha_known,.1,.8)', np.clip(ak, 0.10, 0.80)),
                ('h_i = clip(.2+1.0*ak,.1,.8)',  np.clip(0.2+1.0*ak, 0.10, 0.80)),
                ('h_i = clip(.3+0.8*ak,.1,.8)',  np.clip(0.3+0.8*ak, 0.10, 0.80))]:
    tot = [np.abs(hf_damp(r[4].copy(), h) - r[5]).mean() for r, h in zip(rows, hs)]
    print('%-28s %+10.4f' % (lab, np.mean(tot)-base))
print('\nREAD: if transfer rho is solid (>0.3) and a per-well rule beats the best')
print('global rule locally, v45 ships that rule (hidden side runs hotter: add ~+0.15')
print('to the intercept per the local->hidden dose displacement).')
