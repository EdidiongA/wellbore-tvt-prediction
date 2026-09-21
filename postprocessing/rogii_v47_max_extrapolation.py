# ===== v47: DISPLACEMENT-EXTRAPOLATION stack -- line-damp + HF shrink RAMPED along the well
# ===== (0.30 heel -> 0.65 toe, W601) + extra fine-band shrink (0.20, W151) =====
# Tests two shape hypotheses in one attributable arm vs flat v40: (1) alpha grows
# along the blind zone (schematic interpretation deepens), (2) the fine band
# carries its own measured increment. Placement: LAST cell of sealed-v22 copy.
import os, glob
import numpy as np, pandas as pd

S_LINE, P0 = 0.15, 0.40
H_HEEL, H_TOE, WIN_MID = 0.45, 0.80, 901
H_FINE, WIN_FINE = 0.25, 301

def _ma(x, win):
    # rolling mean defined to the very last station (no toe dead zone)
    return pd.Series(x).rolling(win, center=True, min_periods=max(win//4, 50)).mean().values

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)
n = 0
for w, grp in sub.groupby('well'):
    idx = grp['ridx'].values
    tvt = grp['tvt'].values.astype(float)
    if len(idx) < 400 or not np.isfinite(tvt).all(): continue
    o = np.argsort(idx); i_s = idx[o]; v = tvt[o]
    pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
    near = pos <= P0
    if near.sum() < 20: continue
    a = np.polyfit(pos[near], v[near], 1)
    r = v[near] - np.polyval(a, pos[near])
    k = np.abs(r) <= 3*max(np.std(r),1e-6)
    if k.sum() >= 10: a = np.polyfit(pos[near][k], v[near][k], 1)
    wgt = np.clip((pos-P0)/(1-P0),0,1)*S_LINE
    new = (1-wgt)*v + wgt*np.polyval(a,pos)
    if len(new) >= 2*WIN_MID:                     # HF stages only where windows fit
        lp = _ma(new, WIN_MID); fin = np.isfinite(lp)
        h_pos = H_HEEL + (H_TOE-H_HEEL)*pos
        new[fin] = lp[fin] + (1-h_pos[fin])*(new[fin]-lp[fin])
    if len(new) >= 2*WIN_FINE:
        lp2 = _ma(new, WIN_FINE); fin2 = np.isfinite(lp2)
        new[fin2] = lp2[fin2] + (1-H_FINE)*(new[fin2]-lp2[fin2])
    if not np.isfinite(new).all(): continue
    back = np.empty_like(v); back[o] = new
    sub.loc[grp.index,'tvt'] = back
    n += 1
print('shape-combo damped %d wells' % n)
assert sub['tvt'].notna().all()
sub[['id','tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (v47 displacement-extrapolation stack)')
