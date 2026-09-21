# ===== v38: COMBO -- steering term + flat HF shrink (both measured positives) =====
W_TOE, RAMP_P0, CAP = 0.20, 0.40, 30.0
H_HF, WIN_HF = 0.25, 601
TAG='v38 steering w=0.20 full-band + HF h=0.25 W601'

import os, glob
import numpy as np, pandas as pd

FIT_P0 = 0.40          # line-fit window is ALWAYS pos<=0.40 (decoupled from ramp)

DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'test')):
        DATA = c; break
if DATA is None:
    _h = glob.glob('/kaggle/input/*/test/*__horizontal_well.csv')
    DATA = os.path.dirname(os.path.dirname(_h[0])) if _h else None
assert DATA, 'competition data not found'

def _ma_nan(x, win):
    out = np.convolve(x, np.ones(win) / win, 'same')
    h = win // 2
    out[:h] = np.nan; out[-h:] = np.nan
    return out

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)
print('CONFIG: W_TOE=%.2f RAMP_P0=%.2f CAP=%s | HF: h=%.2f WIN=%d | sign: +w*(Z - L_Z)'
      % (W_TOE, RAMP_P0, str(CAP), H_HF, WIN_HF))
print('submission: %d rows, %d wells' % (len(sub), sub['well'].nunique()))

changed, skipped, cmax_log, capped_wells = 0, 0, [], 0
for w, grp in sub.groupby('well'):
    try:
        hf_path = os.path.join(DATA, 'test', '%s__horizontal_well.csv' % w)
        if not os.path.exists(hf_path):
            skipped += 1; continue
        Z_all = pd.read_csv(hf_path, usecols=['Z'])['Z'].values.astype(float)
        idx = grp['ridx'].values
        tvt = grp['tvt'].values.astype(float)
        if len(idx) < 400 or not np.isfinite(tvt).all():
            skipped += 1; continue
        if idx.min() < 0 or idx.max() >= len(Z_all):
            skipped += 1; continue
        o = np.argsort(idx); i_s = idx[o]
        v = tvt[o]; z = Z_all[i_s]
        if not np.isfinite(z).all():
            skipped += 1; continue
        pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)

        # ---- steering map: low-frequency deviation of Z from its near-zone line
        near = pos <= FIT_P0
        if near.sum() < 20:
            skipped += 1; continue
        a = np.polyfit(pos[near], z[near], 1)
        r = z[near] - np.polyval(a, pos[near])
        k = np.abs(r) <= 3 * max(np.std(r), 1e-6)
        if k.sum() >= 10:
            a = np.polyfit(pos[near][k], z[near][k], 1)
        dev = z - np.polyval(a, pos)          # FULL-BAND: exact v28 Z-part

        wgt = np.clip((pos - RAMP_P0) / max(1 - RAMP_P0, 1e-9), 0, 1) * W_TOE
        corr = wgt * dev                    # CORRECT SIGN: += w*(Z - L_Z)_low
        if CAP is not None:
            over = np.abs(corr) > CAP
            if over.any():
                corr = np.clip(corr, -CAP, CAP)
                capped_wells += 1
        new = v + corr

        # ---- optional flat HF shrink on top (v35-faithful edge behavior) ------
        if H_HF > 0:
            lp = _ma_nan(new, WIN_HF); fin = np.isfinite(lp)
            new[fin] = lp[fin] + (1 - H_HF) * (new[fin] - lp[fin])

        if not np.isfinite(new).all():
            skipped += 1; continue
        back = np.empty_like(v); back[o] = new
        sub.loc[grp.index, 'tvt'] = back
        changed += 1
        cmax_log.append(float(np.abs(corr).max()))
    except Exception:
        skipped += 1

print('corrected %d wells | skipped %d | capped %d | max|corr|: median %.2f ft, p90 %.2f ft'
      % (changed, skipped, capped_wells,
         np.median(cmax_log) if cmax_log else 0.0,
         np.percentile(cmax_log, 90) if cmax_log else 0.0))
assert sub['tvt'].notna().all()
sub[['id', 'tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (%s)' % TAG)
