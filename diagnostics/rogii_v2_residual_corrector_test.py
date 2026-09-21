# ===== V2: does a residual corrector actually REDUCE v22 error out-of-fold? =====
# V1 showed dc_div / sp_div / em_div CORRELATE with error. That is necessary but
# NOT sufficient: a signal can mark where error lives without the error being
# reducible. This fits a corrector in cross-validation and asks the only question
# that matters -- does subtracting its (shrunk) prediction LOWER held-out MAE?
#
# WHERE: append to the v22 OOF fork AFTER the V1 cell (needs predict_well_diag,
# load_well, wells, patched _field_query live, and v22_err/v22_pred in memory or
# v22_oof.pkl attached). CPU only, ~30-40 min over all 773 wells (it needs the
# full set for a trustworthy OOF estimate; do not subsample this one).
#
# WHY OOF AND SHRINKAGE: fitting error in-fold then subtracting it always lowers
# in-fold MAE -- that is the exact trap that inverted your four LB submissions.
# The corrector is trained on 4 folds and applied to the 5th, never on wells it
# saw. And the correction is SHRUNK by a factor lambda<1, because a corrector
# fit on noisy per-station targets overshoots; the sweep finds the lambda that
# helps out-of-fold, or shows that none does.

import numpy as np, time, pickle, glob
from sklearn.ensemble import HistGradientBoostingRegressor

# ---- 1. gather per-station drift features + the SIGNED residual target -------
# Target is per-station signed error (pred - truth) on blind stations, because a
# corrector must know direction, not just magnitude. Features are the well-level
# drift diagnostics broadcast to each of that well's blind stations, plus the
# station's own position in the blind zone (drift accumulates with distance).

if 'v22_pred' in globals() and v22_pred:
    PRED = v22_pred
else:
    _h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
    PRED = pickle.load(open(_h[0], 'rb'))['pred']

SIGS = ['dc_div', 'sp_div', 'em_div', 'drift_span', 'pred_dev', 'nn_dist',
        'branch_spread_mean', 'field_conf', 'corr']
SENT = {-1.0, -9.0, -1, -9}

W = [w for w in wells('train') if w in PRED]
print('building features over %d wells...' % len(W))

Xrows, yrows, wid, t0 = [], [], [], time.time()
for n, w in enumerate(W):
    try:
        h, t = load_well('train', w)
        h.attrs['well'] = w
        pred, status, diag = predict_well_diag(h, t)
        pred = np.asarray(pred, float)
        truth = h['TVT'].values.astype(float)
        blind = ~np.isfinite(h['TVT_input'].values.astype(float))
        m = blind & np.isfinite(truth) & np.isfinite(pred)
        if m.sum() < 1:
            continue
        idx = np.where(m)[0]
        # well-level drift features, constant across the well's blind stations
        feat = [float(diag[k]) if (k in diag and diag[k] not in SENT
                                   and np.isfinite(diag.get(k, np.nan))) else np.nan
                for k in SIGS]
        # per-station position within the blind zone (0 at heel .. 1 at toe)
        pos = (idx - idx.min()) / max(idx.max() - idx.min(), 1)
        for j, s in enumerate(idx):
            Xrows.append(feat + [pos[j]])
            yrows.append(pred[s] - truth[s])      # SIGNED residual
            wid.append(n)
    except Exception as e:
        if len(set(wid)) < 3:
            print('skip %s: %r' % (w, e))
    if (n + 1) % 100 == 0:
        print('  %d/%d  [%.0fs]' % (n + 1, len(W), time.time() - t0), flush=True)

X = np.array(Xrows, float); y = np.array(yrows, float); wid = np.array(wid)
nW = len(np.unique(wid))
print('\n%d stations over %d wells | feature dim %d' % (len(y), nW, X.shape[1]))

# ---- 2. per-well 5-fold split (a well is entirely in one fold) ---------------
rng = np.random.default_rng(0)
perm = rng.permutation(nW)
fold_of_well = np.zeros(nW, int)
for f, chunk in enumerate(np.array_split(perm, 5)):
    fold_of_well[chunk] = f
fold = fold_of_well[wid]

# ---- 3. baseline per-well MAE (no correction) --------------------------------
def well_mae(resid_after):
    e = np.abs(resid_after)
    return np.array([e[wid == u].mean() for u in range(nW)])

base_well = well_mae(y)
print('baseline OOF MAE: %.4f' % base_well.mean())

# ---- 4. out-of-fold corrector predictions ------------------------------------
yhat = np.full(len(y), np.nan)
for f in range(5):
    tr = fold != f; te = fold == f
    ok = tr & np.isfinite(X).all(1) & np.isfinite(y)
    mdl = HistGradientBoostingRegressor(max_iter=300, max_depth=3,
                                        learning_rate=0.05, l2_regularization=1.0,
                                        min_samples_leaf=200, random_state=0)
    mdl.fit(X[ok], y[ok])
    pe = te & np.isfinite(X).all(1)
    yhat[pe] = mdl.predict(X[pe])
valid = np.isfinite(yhat)
print('OOF corrector predictions on %.1f%% of stations' % (100 * valid.mean()))

# ---- 5. shrinkage sweep: does subtracting lambda*yhat help OUT OF FOLD? -------
print('\n%6s %12s %12s' % ('lambda', 'OOF MAE', 'vs base'))
print('-' * 32)
best_l, best_m = 0.0, base_well.mean()
for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    corr = np.where(valid, y - lam * yhat, y)
    m = well_mae(corr).mean()
    flag = '  <-- best' if m < best_m - 1e-9 else ''
    print('%6.1f %12.4f %+12.4f%s' % (lam, m, m - base_well.mean(), flag))
    if m < best_m:
        best_l, best_m = lam, m

gain = base_well.mean() - best_m

# bootstrap CI on the per-well gain at best lambda
pw_base = base_well
pw_best = well_mae(np.where(valid, y - best_l * yhat, y))
pw_gain = pw_base - pw_best
boot = np.array([pw_gain[rng.integers(0, nW, nW)].mean() for _ in range(2000)])
lo, hi = np.percentile(boot, [2.5, 97.5])

print('\n' + '=' * 60)
print('best lambda %.1f -> OOF MAE %.4f | gain %.4f | 95%% CI [%.4f, %.4f]'
      % (best_l, best_m, gain, lo, hi))
print('VERDICT')
if best_l == 0.0 or gain < 0.05:
    print('  DEAD END. No shrinkage reduces OOF error. The drift signals mark')
    print('  where v22 fails but that error is IRREDUCIBLE from these features --')
    print('  exactly the "correlates but cannot fix" case. Do not build v25 on it.')
elif gain < 0.20 or lo <= 0:
    print('  MARGINAL. Gain %.4f is real but under the 0.20 ft floor (or CI' % gain)
    print('  crosses zero). Not worth a submission slot alone. Could combine with')
    print('  another lead, but do not ship as-is.')
else:
    print('  BUILD v25. Gain %.4f clears 0.20 ft with CI excluding zero, OUT OF' % gain)
    print('  FOLD. Apply the corrector at lambda %.1f inside your real v22:' % best_l)
    print('  compute these diag signals per test well, predict the residual, and')
    print('  subtract %.1f x it before writing submission.csv. Save as v25.' % best_l)
    print('  Then gate through your full-200 + pad-holdout + corrupted-suite checks')
    print('  before selecting it as a final.')
print('=' * 60)

# feature importance, for interpretability
try:
    imp = mdl.feature_importances_ if hasattr(mdl, 'feature_importances_') else None
except Exception:
    imp = None
print('\nfeatures used:', SIGS + ['blind_pos'])
