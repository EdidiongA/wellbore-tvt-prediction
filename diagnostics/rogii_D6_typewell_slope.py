# ===== D6: can the TYPEWELL predict the drift slope that internal signals cannot? =====
# D3 proved 0.8 ft of OOF error is slope-shaped (oracle), but internal drift signals
# cannot predict the slope (spearman 0.095). This tests the one untried input: the
# typewell. Mechanism: if v22's blind trajectory has a residual slope error db, then
# sampling the typewell GR at (tvt_hat + db*pos) should MATCH the well's actual GR
# better at the true db than at db=0. Scan db, take the correlation-maximising value
# as a per-well slope estimate, and test it exactly like D3: correlation with the
# true slope, then shrunk-correction OOF gain, 0.20 gate.
#
# This needs NO model re-runs -- it uses stored v22 predictions (v22_pred) plus the
# raw well + typewell CSVs. WHERE: v22 OOF fork, after D3. CPU, ~10-15 min.

import numpy as np, time, pickle, glob
from scipy.stats import spearmanr

if 'v22_pred' in globals() and v22_pred:
    PRED = v22_pred
else:
    _h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
    PRED = pickle.load(open(_h[0], 'rb'))['pred']

DB_GRID = np.arange(-30.0, 30.0 + 1e-9, 1.0)   # candidate slope errors, ft per blind zone
MIN_STA = 60                                    # min blind stations to attempt

def _tw_interp(tw_tvt, tw_gr, q):
    """Sample typewell GR at TVT positions q (NaN outside the log)."""
    out = np.interp(q, tw_tvt, tw_gr, left=np.nan, right=np.nan)
    out[(q < tw_tvt[0]) | (q > tw_tvt[-1])] = np.nan
    return out

def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return np.nan
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a*a).sum() * (b*b).sum())
    return float((a*b).sum() / d) if d > 0 else np.nan

W = [w for w in wells('train') if w in PRED]
print('typewell slope scan over %d wells...' % len(W))

rows = []          # (well, db_hat, peak_corr, corr_at_0, b_true, resid, pos)
t0 = time.time()
for n, w in enumerate(W):
    try:
        h, t = load_well('train', w)
        truth = h['TVT'].values.astype(float)
        gr = h['GR'].values.astype(float)
        tin = h['TVT_input'].values.astype(float)
        blind = ~np.isfinite(tin)
        pv = PRED[w]
        idx = np.asarray(pv['blind_idx'])
        tvt_hat = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(truth[idx]) & np.isfinite(gr[idx]) & np.isfinite(tvt_hat)
        if m.sum() < MIN_STA:
            continue
        idx, tvt_hat = idx[m], tvt_hat[m]
        g = gr[idx]
        pos = (idx - idx.min()) / max(idx.max() - idx.min(), 1)
        tw_tvt = t['TVT'].values.astype(float)
        tw_gr  = t['GR'].values.astype(float)
        ok = np.isfinite(tw_tvt) & np.isfinite(tw_gr)
        tw_tvt, tw_gr = tw_tvt[ok], tw_gr[ok]
        o = np.argsort(tw_tvt); tw_tvt, tw_gr = tw_tvt[o], tw_gr[o]
        if len(tw_tvt) < 50:
            continue
        # scan candidate slope corrections
        cs = np.array([_corr(g, _tw_interp(tw_tvt, tw_gr, tvt_hat + db * pos))
                       for db in DB_GRID])
        if not np.isfinite(cs).any():
            continue
        j = int(np.nanargmax(cs))
        db_hat, peak = float(DB_GRID[j]), float(cs[j])
        c0 = float(cs[np.argmin(np.abs(DB_GRID))]) if np.isfinite(cs[np.argmin(np.abs(DB_GRID))]) else np.nan
        resid = tvt_hat - truth[idx]
        b_true = float(np.polyfit(pos, resid, 1)[0])
        rows.append((w, db_hat, peak, c0, b_true, resid, pos))
    except Exception as e:
        if len(rows) < 3:
            print('skip %s: %r' % (w, e))
    if (n+1) % 150 == 0:
        print('  %d/%d [%.0fs]' % (n+1, len(W), time.time()-t0), flush=True)

nW = len(rows)
db_hat = np.array([r[1] for r in rows]); peak = np.array([r[2] for r in rows])
c0 = np.array([r[3] for r in rows]);     b_true = np.array([r[4] for r in rows])
print('\n%d wells scanned | median peak corr %.3f | median corr at db=0 %.3f'
      % (nW, np.nanmedian(peak), np.nanmedian(c0)))
print('scan hit grid edge (|db_hat|=30) on %.1f%% of wells  (edge hits = untrustworthy)'
      % (100 * (np.abs(db_hat) >= 29.5).mean()))

# note: db_hat estimates -b_true (correcting the prediction toward truth), so the
# expected relationship is NEGATIVE correlation between db_hat and b_true.
rho, p = spearmanr(db_hat, b_true)
print('\nspearman(db_hat, true slope b) = %+.3f (p=%.1e)   [expect NEGATIVE if real]'
      % (rho, p))

# confidence-gated version: only trust wells where the peak clearly beats db=0
lift = peak - c0
for thr in [0.0, 0.02, 0.05, 0.10]:
    s = lift >= thr
    if s.sum() < 40:
        continue
    r2, p2 = spearmanr(db_hat[s], b_true[s])
    print('  wells with corr-lift >= %.2f: n=%3d  spearman %+.3f (p=%.1e)'
          % (thr, s.sum(), r2, p2))

# --- correction sweep: subtract lam * db_hat * pos, gated by lift -------------
base = np.mean([np.abs(r[5]).mean() for r in rows])
print('\nbaseline MAE on scanned wells: %.4f' % base)
print('%6s %8s %12s %12s' % ('lam', 'gate', 'MAE', 'vs base'))
print('-' * 42)
rng = np.random.default_rng(0)
best = (0.0, 0.0, base)
for gate in [0.0, 0.05, 0.10]:
    for lam in [0.2, 0.4, 0.6, 1.0]:
        tot = []
        for i, r in enumerate(rows):
            resid, pos = r[5], r[6]
            if lift[i] >= gate:
                tot.append(np.abs(resid + lam * db_hat[i] * pos).mean())
            else:
                tot.append(np.abs(resid).mean())
        m = np.mean(tot)
        flag = '  <-- best' if m < best[2] - 1e-9 else ''
        print('%6.1f %8.2f %12.4f %+12.4f%s' % (lam, gate, m, m - base, flag))
        if m < best[2]:
            best = (lam, gate, m)

lam, gate, bm = best
gain = base - bm
pw0 = np.array([np.abs(r[5]).mean() for r in rows])
pw1 = np.array([np.abs(r[5] + (lam * db_hat[i] * r[6] if lift[i] >= gate else 0)).mean()
                for i, r in enumerate(rows)])
g = pw0 - pw1
boot = np.array([g[rng.integers(0, nW, nW)].mean() for _ in range(2000)])
lo, hi = np.percentile(boot, [2.5, 97.5])

print('\n' + '=' * 62)
print('best lam %.1f gate %.2f -> MAE %.4f | gain %.4f | 95%% CI [%.4f, %.4f]'
      % (lam, gate, bm, gain, lo, hi))
print('VERDICT')
if gain < 0.05:
    print('  DEAD END. The typewell cannot see the slope either. That closes the')
    print('  last input family: the drift is unpredictable from everything available.')
    print('  Seventh confirmation of the information ceiling — finish the writeup.')
elif gain < 0.20 or lo <= 0:
    print('  MARGINAL. Real signal but under the 0.20 floor. Not a submission on')
    print('  its own; note it in the writeup as partial typewell recoverability.')
else:
    print('  BUILD v25. Typewell slope scan gains %.4f with CI excluding zero.' % gain)
    print('  Wire into v22: after prediction, scan db on the typewell exactly as')
    print('  here, correct blind trajectory by lam*db_hat*pos where lift >= gate.')
    print('  Works identically on test wells (no labels needed). Gate through your')
    print('  full-200 / pad-holdout / corrupted-suite checks before submitting.')
print('=' * 62)
