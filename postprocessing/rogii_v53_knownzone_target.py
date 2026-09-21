# ===== v53: KNOWN-ZONE TARGET damper + full verified stack =====
# Stage 1 variant: damp toward the line fitted on the well's OWN INTERPRETED
# known zone (u = TVT_input + Z, robust line, extrapolated across the blind
# zone, converted back via known Z) -- a ground-truth structural target,
# independent of model drift -- instead of the near-blind prediction line.
# Stages 2-3 unchanged (HF h=0.40 W601; flat ramp -1.5). Single bottom cell.
import os, glob
import numpy as np, pandas as pd

S_LINE, RAMP_P0 = 0.15, 0.40
KN_TAIL = 1500            # fit on the last ~1500 known stations
H_HF, WIN_HF = 0.40, 601
DELTA = -1.5

DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'test')): DATA = c; break
if DATA is None:
    _h = glob.glob('/kaggle/input/*/test/*__horizontal_well.csv')
    DATA = os.path.dirname(os.path.dirname(_h[0])) if _h else None
assert DATA, 'competition data not found'

def _ma_nan(x, win):
    out = np.convolve(x, np.ones(win)/win, 'same'); h = win//2
    out[:h] = np.nan; out[-h:] = np.nan
    return out

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)
print('v53: known-zone-target damper | S=%.2f KN_TAIL=%d | HF h=%.2f W%d | DELTA=%+.1f'
      % (S_LINE, KN_TAIL, H_HF, WIN_HF, DELTA))

n_kn, n_fb, n_skip = 0, 0, 0
for w, grp in sub.groupby('well'):
    try:
        hf = os.path.join(DATA, 'test', '%s__horizontal_well.csv' % w)
        if not os.path.exists(hf): n_skip += 1; continue
        H = pd.read_csv(hf, usecols=['TVT_input', 'Z'])
        tin = H['TVT_input'].values.astype(float); Z = H['Z'].values.astype(float)
        idx = grp['ridx'].values
        tvt = grp['tvt'].values.astype(float)
        if len(idx) < 400 or not np.isfinite(tvt).all(): n_skip += 1; continue
        if idx.min() < 0 or idx.max() >= len(Z) or not np.isfinite(Z[idx]).all():
            n_skip += 1; continue
        o = np.argsort(idx); i_s = idx[o]; v = tvt[o]; z = Z[i_s]
        pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)

        # ---- target: known-zone u-trend, extrapolated ------------------------
        kn = np.where(np.isfinite(tin) & np.isfinite(Z))[0]
        kn = kn[kn < i_s.min()]                      # strictly before the blind zone
        use_known = len(kn) >= 300
        if use_known:
            kn = kn[-KN_TAIL:]
            u_k = tin[kn] + Z[kn]
            a = np.polyfit(kn.astype(float), u_k, 1)
            r = u_k - np.polyval(a, kn.astype(float))
            keep = np.abs(r) <= 3 * max(np.std(r), 1e-6)
            if keep.sum() >= 100:
                a = np.polyfit(kn[keep].astype(float), u_k[keep], 1)
            target = np.polyval(a, i_s.astype(float)) - z    # u_line - Z
            n_kn += 1
        else:                                          # fallback: near-blind line
            near = pos <= 0.40
            a = np.polyfit(pos[near], v[near], 1)
            r = v[near] - np.polyval(a, pos[near])
            keep = np.abs(r) <= 3 * max(np.std(r), 1e-6)
            if keep.sum() >= 10:
                a = np.polyfit(pos[near][keep], v[near][keep], 1)
            target = np.polyval(a, pos)
            n_fb += 1

        wgt = np.clip((pos - RAMP_P0) / (1 - RAMP_P0), 0, 1) * S_LINE
        new = (1 - wgt) * v + wgt * target
        # ---- stages 2-3: unchanged ------------------------------------------
        if len(new) >= 2 * WIN_HF:
            lp = _ma_nan(new, WIN_HF); fin = np.isfinite(lp)
            if fin.sum() >= 50:
                new[fin] = lp[fin] + (1 - H_HF) * (new[fin] - lp[fin])
        new = new + DELTA * pos
        if not np.isfinite(new).all(): n_skip += 1; continue
        back = np.empty_like(v); back[o] = new
        sub.loc[grp.index, 'tvt'] = back
    except Exception:
        n_skip += 1

print('known-zone target: %d wells | fallback near-blind: %d | skipped: %d'
      % (n_kn, n_fb, n_skip))
assert sub['tvt'].notna().all()
sub[['id', 'tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (v53 known-zone-target + full stack)')
