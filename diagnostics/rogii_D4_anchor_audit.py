# ===== D4: is the anchor mis-estimated, and does that drive error? =====
# v22's anchor is anchor_tvt = tvt_in[k0-1] -- a SINGLE station value at the
# known/blind boundary. If that one point is noisy, the entire blind trajectory
# inherits the offset. Recentered MAE being 9-15 vs raw 6.7-7.7 across all BiGRU
# folds is the fingerprint of exactly this. D4 audits anchor quality on training
# wells (where truth is known) and tests whether a smoothed anchor reduces error.
#
# WHERE: v22 OOF fork, after V1/V2/D3. CPU, ~25 min. Needs the usual globals.

import numpy as np, time, pickle, glob
from scipy.stats import spearmanr

if 'v22_err' in globals() and v22_err:
    ERR = dict(v22_err)
else:
    _h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
    ERR = pickle.load(open(_h[0], 'rb'))['err']

W = [w for w in wells('train') if w in ERR]
print('auditing anchor over %d wells...' % len(W))

anchor_err, well_err, sig_rows = [], [], []
per = {}
t0 = time.time()
for n, w in enumerate(W):
    try:
        h, t = load_well('train', w)
        h.attrs['well'] = w
        tin = h['TVT_input'].values.astype(float)
        truth = h['TVT'].values.astype(float)
        blind = ~np.isfinite(tin)
        k0 = int(np.argmax(blind))
        if k0 < 6 or not blind.any():
            continue
        # v22's point anchor: the single station before blind
        pt_anchor = tin[k0 - 1]
        # a robust alternative: median of the last few KNOWN TVT_input values
        tail = tin[max(0, k0 - 6):k0]
        tail = tail[np.isfinite(tail)]
        if len(tail) < 3 or not np.isfinite(pt_anchor):
            continue
        smooth_anchor = float(np.median(tail))
        # "true" anchor: the actual TVT at the boundary station
        true_anchor = truth[k0 - 1]
        if not np.isfinite(true_anchor):
            continue
        pt_ae = abs(pt_anchor - true_anchor)          # v22 point-anchor error
        sm_ae = abs(smooth_anchor - true_anchor)      # smoothed-anchor error
        anchor_err.append(pt_ae); well_err.append(ERR[w])
        per[w] = (pt_ae, sm_ae, smooth_anchor - pt_anchor)   # how much smoothing moves it
    except Exception as e:
        if len(anchor_err) < 3: print('skip %s: %r' % (w, e))
    if (n+1) % 150 == 0:
        print('  %d/%d [%.0fs]' % (n+1, len(W), time.time()-t0), flush=True)

ae = np.array(anchor_err); we = np.array(well_err)
wk = list(per)
pt = np.array([per[w][0] for w in wk]); sm = np.array([per[w][1] for w in wk])

print('\n%d wells audited' % len(ae))
print('v22 point-anchor error : median %.3f  p90 %.3f  max %.3f'
      % (np.median(pt), np.percentile(pt, 90), pt.max()))
print('smoothed-anchor error  : median %.3f  p90 %.3f' % (np.median(sm), np.percentile(sm, 90)))
print('smoothing reduces anchor error on %.1f%% of wells' % (100*(sm < pt).mean()))

rho, p = spearmanr(ae, we)
print('\nspearman(anchor_error, v22 well_error) = %+.3f (p=%.1e)' % (rho, p))
if abs(rho) > 0.3 and p < 0.01:
    tag = 'STRONG — anchor error drives model error'
elif abs(rho) > 0.15:
    tag = 'weak — anchor is a minor contributor'
else:
    tag = 'none — anchor quality is not the problem'
print('  ->', tag)

# how much would perfect anchoring help? proxy: wells in worst anchor quartile
q = np.quantile(pt, 0.75)
bad = pt >= q
print('\nworst-anchor quartile (>= %.3f ft error): mean well err %.3f' % (q, we[[wk.index(w) for w in wk if per[w][0] >= q]].mean() if False else np.array([ERR[w] for w in wk])[bad].mean()))
print('best  3 quartiles:                        mean well err %.3f' % np.array([ERR[w] for w in wk])[~bad].mean())

print('\n' + '='*62)
print('VERDICT')
gap = np.array([ERR[w] for w in wk])[bad].mean() - np.array([ERR[w] for w in wk])[~bad].mean()
if abs(rho) > 0.3 and p < 0.01 and (sm < pt).mean() > 0.55:
    print('  BUILD candidate. Anchor error correlates with model error (%.2f) AND' % rho)
    print('  a median-smoothed anchor beats the point anchor on %.0f%% of wells.'
          % (100*(sm < pt).mean()))
    print('  Next: replace anchor_tvt = tvt_in[k0-1] with the median of the last')
    print('  ~6 known values, re-run OOF, gate at 0.20 ft. Worst-anchor wells carry')
    print('  %.2f ft more error, so the ceiling is real.' % gap)
elif abs(rho) > 0.15:
    print('  PARTIAL. Anchor matters somewhat (rho %.2f) but smoothing may not be' % rho)
    print('  the fix. Worth trying the median anchor, expect modest gain.')
else:
    print('  NOT THE PROBLEM. Anchor error does not track model error (rho %.2f).' % rho)
    print('  The recentered-MAE gap in BiGRU came from something else. Drop D4.')
print('='*62)
