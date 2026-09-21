# ===== v45: ALPHA-ADAPTIVE damper -- per-well HF dose measured from the well's
# ===== own known zone, plus the standard line-damp (S=0.13) underneath =====
# h_i = clip(A + B*alpha_known_i, HMIN, HMAX); defaults calibrated for the
# local->hidden displacement (hidden runs ~+0.15-0.3 hotter than local optimum).
# EDIT A/B after L7 reports. Placement: LAST cell of a sealed-v22 copy.
import os, glob
import numpy as np, pandas as pd

S_LINE, P0 = 0.15, 0.40
A, B, HMIN, HMAX = 0.30, 0.80, 0.15, 0.70
H_FALLBACK = 0.46          # unmeasurable wells get the measured global optimum
WIN_A, WIN_HF = 301, 601

DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'test')): DATA = c; break
if DATA is None:
    _h = glob.glob('/kaggle/input/*/test/*__horizontal_well.csv')
    DATA = os.path.dirname(os.path.dirname(_h[0])) if _h else None
assert DATA

def _ma(x, win):
    out = np.convolve(x, np.ones(win)/win, 'same'); h = win//2
    out[:h] = np.nan; out[-h:] = np.nan
    return out

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)

changed, hs = 0, []
for w, grp in sub.groupby('well'):
    try:
        hf = os.path.join(DATA, 'test', '%s__horizontal_well.csv' % w)
        if not os.path.exists(hf): continue
        H = pd.read_csv(hf, usecols=['TVT_input', 'Z'])
        tin = H['TVT_input'].values.astype(float); Z = H['Z'].values.astype(float)
        kn = np.isfinite(tin) & np.isfinite(Z)
        # --- per-well alpha from the known zone -------------------------------
        h_i = H_FALLBACK                           # fallback: measured global h*
        if kn.sum() >= 700:
            t_k, z_k = tin[kn], Z[kn]
            hf_t = t_k - _ma(t_k, WIN_A); hf_z = z_k - _ma(z_k, WIN_A)
            ok = np.isfinite(hf_t) & np.isfinite(hf_z)
            if ok.sum() >= 250 and np.var(hf_z[ok]) > 1e-6:
                a_kn = 1.0 + np.cov(hf_t[ok], hf_z[ok])[0,1] / np.var(hf_z[ok])
                h_i = float(np.clip(A + B * a_kn, HMIN, HMAX))
        idx = grp['ridx'].values
        tvt = grp['tvt'].values.astype(float)
        if len(idx) < 400 or not np.isfinite(tvt).all(): continue
        o = np.argsort(idx); i_s = idx[o]; v = tvt[o]
        pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
        near = pos <= P0
        if near.sum() < 20: continue
        a = np.polyfit(pos[near], v[near], 1)
        r = v[near] - np.polyval(a, pos[near])
        k = np.abs(r) <= 3 * max(np.std(r), 1e-6)
        if k.sum() >= 10: a = np.polyfit(pos[near][k], v[near][k], 1)
        wgt = np.clip((pos - P0) / (1 - P0), 0, 1) * S_LINE
        new = (1 - wgt) * v + wgt * np.polyval(a, pos)
        if len(new) >= 2 * WIN_HF:
            lp = _ma(new, WIN_HF); fin = np.isfinite(lp)
            if fin.sum() >= 50:
                new[fin] = lp[fin] + (1 - h_i) * (new[fin] - lp[fin])
        if not np.isfinite(new).all(): continue
        back = np.empty_like(v); back[o] = new
        sub.loc[grp.index, 'tvt'] = back
        changed += 1; hs.append(h_i)
    except Exception:
        pass
print('alpha-adaptive: %d wells | h median %.3f [%.2f-%.2f]'
      % (changed, np.median(hs), min(hs), max(hs)))
assert sub['tvt'].notna().all()
sub[['id','tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (v45 alpha-adaptive, A=%.2f B=%.2f)' % (A, B))
