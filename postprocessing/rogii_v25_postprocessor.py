# ===== v25: typewell slope post-processor (append AFTER the submission cell) =====
# Applies the D6 correction to submission.csv: per test well, scan candidate slope
# corrections db against the typewell GR, and where the correlation peak clearly
# beats db=0 (lift >= GATE), tilt the blind trajectory by LAM * db_hat * pos.
# Parameters are EXACTLY the OOF-validated ones from D6. Needs no labels.
#
# WHERE: your REAL v22 notebook (copy it -> v25), as a new final cell AFTER
# submission.csv is written. It edits the file in place and reports every change.

import os, glob
import numpy as np, pandas as pd

LAM, GATE = 0.2, 0.10                 # OOF-validated on 773 training wells (D6)
DB_GRID = np.arange(-30.0, 30.0 + 1e-9, 1.0)
MIN_STA, EDGE = 60, 29.5

DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'test')):
        DATA = c; break
if DATA is None:
    _h = glob.glob('/kaggle/input/*/test/*__horizontal_well.csv')
    DATA = os.path.dirname(os.path.dirname(_h[0])) if _h else None
assert DATA, 'competition data not found'

def _tw_interp(tw_tvt, tw_gr, q):
    out = np.interp(q, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    out[(q < tw_tvt[0]) | (q > tw_tvt[-1])] = np.nan
    return out

def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)
print('submission: %d rows, %d wells' % (len(sub), sub['well'].nunique()))

changed, skipped = 0, []
for w, grp in sub.groupby('well'):
    try:
        hf = os.path.join(DATA, 'test', '%s__horizontal_well.csv' % w)
        tf = os.path.join(DATA, 'test', '%s__typewell.csv' % w)
        if not (os.path.exists(hf) and os.path.exists(tf)):
            skipped.append((w, 'files missing')); continue
        h = pd.read_csv(hf); t = pd.read_csv(tf)
        gr_all = h['GR'].values.astype(float)
        idx = grp['ridx'].values
        tvt_hat = grp['tvt'].values.astype(float)
        ok = (idx >= 0) & (idx < len(gr_all))
        if ok.sum() < MIN_STA:
            skipped.append((w, 'too short')); continue
        idx, tvt_hat = idx[ok], tvt_hat[ok]
        g = gr_all[idx]
        m = np.isfinite(g) & np.isfinite(tvt_hat)
        if m.sum() < MIN_STA:
            skipped.append((w, 'too few finite')); continue
        pos_full = (idx - idx.min()) / max(idx.max() - idx.min(), 1)

        tw_tvt = t['TVT'].values.astype(float)
        tw_gr = t['GR'].values.astype(float)
        tok = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
        tw_tvt, tw_gr = tw_tvt[tok], tw_gr[tok]
        o = np.argsort(tw_tvt); tw_tvt, tw_gr = tw_tvt[o], tw_gr[o]
        if len(tw_tvt) < 50:
            skipped.append((w, 'typewell short')); continue

        cs = np.array([_corr(g[m], _tw_interp(tw_tvt, tw_gr,
                                              tvt_hat[m] + db * pos_full[m]))
                       for db in DB_GRID])
        if not np.isfinite(cs).any():
            skipped.append((w, 'no finite corr')); continue
        j = int(np.nanargmax(cs))
        db_hat, peak = float(DB_GRID[j]), float(cs[j])
        c0 = float(cs[np.argmin(np.abs(DB_GRID))])
        lift = peak - c0 if np.isfinite(c0) else 0.0

        if abs(db_hat) >= EDGE:
            skipped.append((w, 'edge hit db=%.0f' % db_hat)); continue
        if lift < GATE:
            continue                       # gate not met -> leave v22 untouched

        newv = tvt_hat + LAM * db_hat * pos_full
        sub.loc[grp.index[ok], 'tvt'] = newv
        changed += 1
        print('  %s: db_hat %+5.0f  lift %.3f  -> tilted %d stations by %+.1f ft at toe'
              % (w, db_hat, lift, ok.sum(), LAM * db_hat))
    except Exception as e:
        skipped.append((w, repr(e)[:60]))

print('\ncorrected %d wells | untouched %d | skipped %d'
      % (changed, sub['well'].nunique() - changed - len(skipped), len(skipped)))
if skipped[:5]:
    print('skips:', skipped[:5])

sub[['id', 'tvt']].to_csv('submission.csv', index=False)
print('\nsubmission.csv rewritten (v25 = v22 + typewell slope correction,'
      ' lam=%.1f gate=%.2f)' % (LAM, GATE))
