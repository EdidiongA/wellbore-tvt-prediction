# ===== V1: do v22's own drift-divergence signals predict where it goes wrong? =====
# G3 found three public teams gained from rolling-std / first-diff of their track
# (pf_*_std, pf_*_delta). v22 ALREADY computes the same family as diag scalars --
# em_div, dc_div, sp_div, le_div, tw_div, drift_span, pred_dev. This tests whether
# those predict per-well OOF error. If they do, a light residual corrector fed by
# them (or a confidence taper keyed on them) is the concrete build.
#
# WHERE: append to the v22 OOF fork AFTER P6 (needs predict_well_diag, load_well,
# wells, patched _field_query live, and v22_err in memory or v22_oof.pkl attached).
# CPU only, ~6-8 min for 150 wells.

import numpy as np, time
from scipy.stats import spearmanr

N = 150
rng = np.random.default_rng(0)

if 'v22_err' in globals() and v22_err:
    ERR = dict(v22_err)
else:
    import pickle, glob
    _h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
    ERR = pickle.load(open(_h[0], 'rb'))['err']
print('per-well error for %d wells' % len(ERR))

_all = [w for w in wells('train') if w in ERR]
samp = [_all[i] for i in sorted(rng.choice(len(_all), min(N, len(_all)), replace=False))]

# v22 signals matching the public teams' drift-derivative family.
# -1 / -9 are v22's "unavailable" sentinels -> treat as NaN.
SIGS = ['em_div', 'dc_div', 'sp_div', 'le_div', 'tw_div', 'dctw_div',
        'drift_span', 'pred_dev', 'anchor_margin', 'corr', 'nn_dist']
SENTINEL = {-1.0, -9.0, -1, -9}

rows, errs, t0 = [], [], time.time()
first = True
for n, w in enumerate(samp):
    try:
        h, t = load_well('train', w)
        h.attrs['well'] = w
        _, _, diag = predict_well_diag(h, t)
        if first:
            print('diag keys exposed:', sorted(diag)); first = False
        f = {k: float(diag[k]) for k in SIGS
             if k in diag and diag[k] not in SENTINEL and np.isfinite(diag.get(k, np.nan))}
        if f:
            rows.append(f); errs.append(ERR[w])
    except Exception as e:
        if len(rows) < 3:
            print('skip %s: %r' % (w, e))
    if (n + 1) % 30 == 0:
        print('%3d/%d  [%.0fs]' % (n + 1, len(samp), time.time() - t0), flush=True)

errs = np.array(errs, float)
keys = sorted({k for r in rows for k in r})
print('\n%d wells | error mean %.2f\n' % (len(rows), errs.mean()))
print('%-14s %6s %9s %9s   %s' % ('signal', 'n', 'spearman', 'p', 'reading'))
print('-' * 58)
res = []
for k in keys:
    x = np.array([r.get(k, np.nan) for r in rows], float)
    ok = np.isfinite(x) & np.isfinite(errs)
    if ok.sum() < 30:
        continue
    rho, p = spearmanr(x[ok], errs[ok])
    res.append((abs(rho), rho, p, k))
    tag = ('STRONG' if abs(rho) > 0.30 and p < 0.01 else
           'weak' if abs(rho) > 0.15 else 'none')
    print('%-14s %6d %+9.3f %9.1e   %s' % (k, ok.sum(), rho, p, tag))

# multivariate: how much error variance do the significant ones explain TOGETHER?
if len(res) >= 2 and len(rows) > 40:
    from numpy.linalg import lstsq
    strong = [k for _, _, p, k in res if p < 0.05]
    if len(strong) >= 2:
        def _rank(a):
            o = a.argsort(); r = np.empty_like(o, float); r[o] = np.arange(len(a)); return r
        M = np.column_stack([_rank(np.array([r.get(k, np.nan) for r in rows]))
                             for k in strong])
        y = _rank(errs)
        keep = np.isfinite(M).all(1)
        A = np.column_stack([M[keep], np.ones(keep.sum())])
        coef, *_ = lstsq(A, y[keep], rcond=None)
        yhat = A @ coef
        r2 = 1 - ((y[keep] - yhat) ** 2).sum() / ((y[keep] - y[keep].mean()) ** 2).sum()
        print('\njoint rank-R^2 of {%s}: %.3f' % (', '.join(strong), r2))

print('\n' + '=' * 58)
if res:
    best = max(res)
    if best[0] > 0.30 and best[2] < 0.01:
        print('VERDICT: %s tracks v22 error (spearman %+.3f, p=%.1e).'
              % (best[3], best[1], best[2]))
        print('  The drift-derivative family is REAL on your data. Build one of:')
        print('   (a) residual corrector: LightGBM on these signals -> predicts and')
        print('       subtracts v22 per-well error. Test in OOF before shipping.')
        print('   (b) confidence taper: down-weight the structural prior where the')
        print('       high-error signals fire. Cheaper, no new model.')
        print('  Either is a v25 candidate. Gate through your 3 checks + 0.20 floor.')
    elif best[0] > 0.15:
        print('VERDICT: weak (best %s %+.3f). The public teams stacked MANY such'
              % (best[3], best[1]))
        print('  signals; individually weak but jointly useful. Check joint R^2')
        print('  above -- if > 0.10, a corrector on the full set may still pay.')
    else:
        print('VERDICT: no correlation. These signals do not predict v22 failure')
        print('  on your data; G3 summed gains did not transfer. Drop this lead.')
else:
    print('No signals computed. Print sorted(diag) from one well and tell me which')
    print('keys exist -- the SIGS list may need adjusting to your v22 build.')
print('=' * 58)
