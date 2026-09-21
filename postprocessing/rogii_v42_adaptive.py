# ===== v42: per-well ADAPTIVE damping -- dose scaled by typewell disagreement =====
# Wells whose blind-zone GR agrees poorly with the typewell at the predicted TVT
# are the wells v22 errs on most (D6: corr signals ~rho 0.6 with error). This
# scales BOTH damper doses per well by that disagreement rank:
#   S_i = 0.10 + 0.15*u_i   (line-damper toe strength, around measured 0.128)
#   H_i = 0.30 + 0.30*u_i   (flat HF shrink, recentered on measured h*=0.46)
# where u_i = 1 - rank(typewell corr). Self-limiting forms throughout.
# Placement: LAST cell of a sealed-v22 copy, after the submission cell.
import os, glob
import numpy as np, pandas as pd

P0, WIN_HF = 0.40, 601

DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'test')):
        DATA = c; break
if DATA is None:
    _h = glob.glob('/kaggle/input/*/test/*__horizontal_well.csv')
    DATA = os.path.dirname(os.path.dirname(_h[0])) if _h else None
assert DATA

def _ma_nan(x, win):
    out = np.convolve(x, np.ones(win)/win, 'same'); h = win // 2
    out[:h] = np.nan; out[-h:] = np.nan
    return out

def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100: return np.nan
    a, b = a[m]-a[m].mean(), b[m]-b[m].mean()
    d = np.sqrt((a*a).sum()*(b*b).sum())
    return float((a*b).sum()/d) if d > 0 else np.nan

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)

# ---- pass 1: per-well typewell agreement at the current prediction -----------
agree = {}
for w, grp in sub.groupby('well'):
    try:
        hf = os.path.join(DATA, 'test', '%s__horizontal_well.csv' % w)
        tf = os.path.join(DATA, 'test', '%s__typewell.csv' % w)
        if not (os.path.exists(hf) and os.path.exists(tf)): continue
        H = pd.read_csv(hf, usecols=['GR']); T = pd.read_csv(tf)
        gr = H['GR'].values.astype(float)
        idx = grp['ridx'].values; tvt = grp['tvt'].values.astype(float)
        if idx.max() >= len(gr) or idx.min() < 0: continue
        tw_t = T['TVT'].values.astype(float); tw_g = T['GR'].values.astype(float)
        ok = np.isfinite(tw_t) & np.isfinite(tw_g)
        tw_t, tw_g = tw_t[ok], tw_g[ok]
        o = np.argsort(tw_t); tw_t, tw_g = tw_t[o], tw_g[o]
        if len(tw_t) < 50: continue
        ref = np.interp(tvt, tw_t, tw_g, left=np.nan, right=np.nan)
        ref[(tvt < tw_t[0]) | (tvt > tw_t[-1])] = np.nan
        agree[w] = _corr(gr[idx], ref)
    except Exception:
        pass
vals = np.array([v for v in agree.values() if np.isfinite(v)])
if len(vals) >= 10:
    print('typewell agreement: %d wells | median %.3f  p25 %.3f  p75 %.3f'
          % (len(vals), np.median(vals), *np.percentile(vals, [25, 75])))
    ranks = {w: (np.searchsorted(np.sort(vals), v) / max(len(vals)-1, 1))
             for w, v in agree.items() if np.isfinite(v)}
else:
    print('WARNING: only %d finite agreements -- all wells get fallback doses '
          '(S=0.175, H=0.45, the global optima); adaptivity disabled, run continues.'
          % len(vals))
    ranks = {}

# ---- pass 2: adaptive line-damp + adaptive HF shrink -------------------------
changed, s_used, h_used = 0, [], []
for w, grp in sub.groupby('well'):
    idx = grp['ridx'].values
    tvt = grp['tvt'].values.astype(float)
    if len(idx) < 400 or not np.isfinite(tvt).all(): continue
    u_i = 1.0 - ranks.get(w, 0.5)              # unknown wells -> mid dose
    S_i = 0.10 + 0.15 * u_i
    H_i = 0.30 + 0.30 * u_i   # recentered on measured h*=0.46
    o = np.argsort(idx); i_s = idx[o]; v = tvt[o]
    pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
    near = pos <= P0
    if near.sum() < 20: continue
    a = np.polyfit(pos[near], v[near], 1)
    r = v[near] - np.polyval(a, pos[near])
    k = np.abs(r) <= 3 * max(np.std(r), 1e-6)
    if k.sum() >= 10: a = np.polyfit(pos[near][k], v[near][k], 1)
    wgt = np.clip((pos - P0) / (1 - P0), 0, 1) * S_i
    new = (1 - wgt) * v + wgt * np.polyval(a, pos)
    if len(new) >= 2 * WIN_HF:
        lp = _ma_nan(new, WIN_HF); fin = np.isfinite(lp)
        if fin.sum() >= 50:
            new[fin] = lp[fin] + (1 - H_i) * (new[fin] - lp[fin])
    if not np.isfinite(new).all(): continue
    back = np.empty_like(v); back[o] = new
    sub.loc[grp.index, 'tvt'] = back
    changed += 1; s_used.append(S_i); h_used.append(H_i)

print('adaptively damped %d wells | S median %.3f [%.2f-%.2f] | H median %.3f [%.2f-%.2f]'
      % (changed, np.median(s_used), min(s_used), max(s_used),
         np.median(h_used), min(h_used), max(h_used)))
assert sub['tvt'].notna().all()
sub[['id', 'tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (v42 adaptive damping)')
