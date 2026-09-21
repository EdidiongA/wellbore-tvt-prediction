# ===== D3: is the tail error a correctable per-well SLOPE? =====
# V2 tested per-station residual correction and failed. But your tail error is a
# 3.6->33 ft RAMP from anchor to toe -- accumulated SLOPE drift, one wrong tilt per
# well, not independent station noise. A residual corrector cannot fix a slope. This
# fits ONE drift-rate per well, tests whether it is predictable from drift signals
# OUT OF FOLD, and whether correcting the tilt lowers OOF MAE.
#
# WHERE: v22 OOF fork, after V1/V2. Needs predict_well_diag, load_well, wells,
# patched _field_query, and v22_pred/v22_err (or v22_oof.pkl). CPU, ~30 min.

import numpy as np, time, pickle, glob
from sklearn.ensemble import HistGradientBoostingRegressor

if 'v22_pred' in globals() and v22_pred:
    PRED = v22_pred
else:
    _h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
    PRED = pickle.load(open(_h[0], 'rb'))['pred']

SIGS = ['dc_div', 'sp_div', 'em_div', 'drift_span', 'pred_dev', 'nn_dist',
        'branch_spread_mean', 'field_conf', 'corr', 'blind_len', 'known_len']
SENT = {-1.0, -9.0, -1, -9}
W = [w for w in wells('train') if w in PRED]
print('fitting per-well drift slopes over %d wells...' % len(W))

# per-well: signed error vs blind position -> (intercept a, slope b)
feats, slopes, inter, errs, per_well_resid = [], [], [], [], {}
t0 = time.time()
for n, w in enumerate(W):
    try:
        h, t = load_well('train', w)
        h.attrs['well'] = w
        pred, status, diag = predict_well_diag(h, t)
        pred = np.asarray(pred, float)
        truth = h['TVT'].values.astype(float)
        blind = ~np.isfinite(h['TVT_input'].values.astype(float))
        m = blind & np.isfinite(truth) & np.isfinite(pred)
        if m.sum() < 20:
            continue
        idx = np.where(m)[0]
        pos = (idx - idx.min()) / max(idx.max() - idx.min(), 1)   # 0 heel .. 1 toe
        resid = pred[idx] - truth[idx]                            # signed error
        b, a = np.polyfit(pos, resid, 1)                          # slope, intercept
        f = [float(diag[k]) if (k in diag and diag[k] not in SENT
             and np.isfinite(diag.get(k, np.nan))) else np.nan for k in SIGS]
        feats.append(f); slopes.append(b); inter.append(a)
        errs.append(np.abs(resid).mean())
        per_well_resid[w] = (idx, pos, resid, np.abs(resid).mean())
    except Exception as e:
        if len(feats) < 3: print('skip %s: %r' % (w, e))
    if (n+1) % 150 == 0:
        print('  %d/%d [%.0fs]' % (n+1, len(W), time.time()-t0), flush=True)

X = np.array(feats); b_true = np.array(slopes); nW = len(b_true)
print('\n%d wells | mean |slope| %.3f ft/blindzone | baseline MAE %.4f'
      % (nW, np.abs(b_true).mean(), np.mean(errs)))

# how much of total error IS the slope? (vs intercept + curvature)
wkeys = list(per_well_resid)
base_mae = np.mean([per_well_resid[w][3] for w in wkeys])
# oracle: subtract each well's OWN fitted slope (upper bound on slope-only fix)
orc = []
for w in wkeys:
    idx, pos, resid, _ = per_well_resid[w]
    b, a = np.polyfit(pos, resid, 1)
    orc.append(np.abs(resid - b*pos).mean())        # remove slope, keep intercept+noise
print('ORACLE slope removal (subtract true slope): %.4f (gain %.4f)'
      % (np.mean(orc), base_mae - np.mean(orc)))
print('  ^ upper bound if slope were perfectly known. If gain < 0.2, error is')
print('    error is NOT mainly slope and D3 cannot help regardless of prediction.')

# --- OOF: is the slope PREDICTABLE from signals? --------------------------
rng = np.random.default_rng(0); perm = rng.permutation(nW)
fold = np.zeros(nW, int)
for f, ch in enumerate(np.array_split(perm, 5)): fold[ch] = f
bhat = np.full(nW, np.nan)
for f in range(5):
    tr = (fold != f) & np.isfinite(X).all(1) & np.isfinite(b_true)
    te = (fold == f) & np.isfinite(X).all(1)
    if tr.sum() < 50 or te.sum() < 1: continue
    mdl = HistGradientBoostingRegressor(max_iter=200, max_depth=3, learning_rate=0.05,
              l2_regularization=1.0, min_samples_leaf=20, random_state=0)
    mdl.fit(X[tr], b_true[tr]); bhat[te] = mdl.predict(X[te])
from scipy.stats import spearmanr
ok = np.isfinite(bhat) & np.isfinite(b_true)
rho, p = spearmanr(b_true[ok], bhat[ok])
print('\nOOF slope predictability: spearman(true b, pred b) = %+.3f (p=%.1e)' % (rho, p))

# --- does OOF-predicted slope correction lower MAE? -----------------------
print('\n%6s %12s %12s' % ('lambda', 'OOF MAE', 'vs base'))
print('-'*32)
best_l, best_m = 0.0, base_mae
for lam in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    tot = []
    for i, w in enumerate(wkeys):
        if not np.isfinite(bhat[i]): tot.append(per_well_resid[w][3]); continue
        idx, pos, resid, _ = per_well_resid[w]
        tot.append(np.abs(resid - lam*bhat[i]*pos).mean())
    m = np.mean(tot)
    flag = '  <-- best' if m < best_m - 1e-9 else ''
    print('%6.1f %12.4f %+12.4f%s' % (lam, m, m-base_mae, flag))
    if m < best_m: best_l, best_m = lam, m

gain = base_mae - best_m
pw = np.array([per_well_resid[w][3] for w in wkeys])
pwb = np.array([np.abs(per_well_resid[w][2] - (best_l*bhat[i]*per_well_resid[w][1]
                if np.isfinite(bhat[i]) else 0)).mean() for i, w in enumerate(wkeys)])
g = pw - pwb
boot = np.array([g[rng.integers(0, nW, nW)].mean() for _ in range(2000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print('\n' + '='*60)
print('best lambda %.1f -> MAE %.4f | gain %.4f | 95%% CI [%.4f, %.4f]'
      % (best_l, best_m, gain, lo, hi))
print('VERDICT')
if best_l == 0 or gain < 0.05:
    print('  DEAD END. Slope not predictable OOF, or error is not slope-shaped.')
elif gain < 0.20 or lo <= 0:
    print('  MARGINAL. Real but under 0.20 floor / CI crosses zero.')
else:
    print('  BUILD v25. OOF slope correction gains %.4f. Apply lambda %.1f x' % (gain, best_l))
    print('  predicted-slope x position to each blind station before submission.')
print('='*60)
