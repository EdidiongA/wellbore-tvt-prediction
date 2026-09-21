# ===== S1: what actually differs between train and test? (runs inside submission) =====
# THE key unmeasured quantity. Your held-out OOF is 5.5 but LB is 8.9 -- a 3.4 ft
# train->test gap that no model change touches. This probe runs during a real
# scoring pass, where TEST well coordinates and logs ARE visible, and diffs the
# test distribution against train on the axes most likely to carry the shift.
#
# WHERE: paste into your REAL v22 submission notebook (rogii-tvt-main), right
# AFTER the wells are loaded but it does NOT need to alter the submission. It
# prints a report and writes shift_report.csv. Set WRITE_SUBMISSION appropriately
# -- this can ride along on a normal submission run at zero extra cost.
#
# Nothing here changes your predictions. It is pure measurement, and it is the
# only way to see the test distribution.

import os, glob
import numpy as np, pandas as pd

DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'train')):
        DATA = c; break
assert DATA, 'competition data not found'

def _well_stats(split):
    """Per-well summary of the axes most likely to differ train vs test."""
    rows = []
    for f in sorted(glob.glob(os.path.join(DATA, split, '*__horizontal_well.csv'))):
        w = os.path.basename(f).split('__')[0]
        try:
            h = pd.read_csv(f)
        except Exception:
            continue
        tin = h['TVT_input'].values.astype(float) if 'TVT_input' in h else np.full(len(h), np.nan)
        blind = ~np.isfinite(tin)
        known = np.isfinite(tin)
        X, Y = h['X'].values.astype(float), h['Y'].values.astype(float)
        Z = h['Z'].values.astype(float)
        GR = h['GR'].values.astype(float) if 'GR' in h else np.full(len(h), np.nan)
        MD = h['MD'].values.astype(float) if 'MD' in h else np.arange(len(h)).astype(float)
        r = dict(
            well=w, split=split, n=len(h),
            blind_len=int(blind.sum()), known_len=int(known.sum()),
            blind_frac=float(blind.mean()),
            md_span=float(np.nanmax(MD) - np.nanmin(MD)) if len(MD) else np.nan,
            blind_md_span=float(np.nanmax(MD[blind]) - np.nanmin(MD[blind])) if blind.sum() > 1 else np.nan,
            z_mean=float(np.nanmean(Z)), z_std=float(np.nanstd(Z)),
            gr_mean=float(np.nanmean(GR)), gr_std=float(np.nanstd(GR)),
            gr_known_mean=float(np.nanmean(GR[known])) if known.sum() else np.nan,
            gr_blind_mean=float(np.nanmean(GR[blind])) if blind.sum() else np.nan,
            # GR calibration drift: does GR level shift between known and blind zones?
            gr_kb_shift=(float(np.nanmean(GR[blind]) - np.nanmean(GR[known]))
                         if known.sum() and blind.sum() else np.nan),
        )
        rows.append(r)
    return pd.DataFrame(rows)

tr = _well_stats('train')
te = _well_stats('test')
print('train wells: %d | test wells: %d' % (len(tr), len(te)))
if len(te) < 20:
    print('\n*** Only %d test wells visible -- this is the EXAMPLE test folder, not the'
          ' hidden set. On a real scored rerun this fills with ~200 real test wells'
          ' and the report below becomes meaningful. Submit it to get the real diff. ***'
          % len(te))

# --- distribution diff on each axis, with a standardized effect size ----------
AXES = ['blind_len', 'known_len', 'blind_frac', 'md_span', 'blind_md_span',
        'z_mean', 'z_std', 'gr_mean', 'gr_std', 'gr_kb_shift']
print('\n%-16s %12s %12s %10s %8s' % ('axis', 'train med', 'test med', 'shift(SD)', 'flag'))
print('-' * 62)
flags = []
for a in AXES:
    tv = tr[a].dropna().values; ev = te[a].dropna().values
    if len(tv) < 5 or len(ev) < 1:
        continue
    pooled_sd = np.sqrt((tv.var() + ev.var()) / 2) or 1.0
    d = (np.median(ev) - np.median(tv)) / pooled_sd     # standardized shift
    flag = '<<<' if abs(d) > 0.5 else '<' if abs(d) > 0.25 else ''
    if abs(d) > 0.25:
        flags.append((abs(d), a, d))
    print('%-16s %12.2f %12.2f %10.2f %8s' % (a, np.median(tv), np.median(ev), d, flag))

print('\n' + '=' * 62)
if not flags:
    print('No axis shifts > 0.25 SD. Either test ~ train (gap is elsewhere), or')
    print('this is the example folder (too few test wells to tell).')
else:
    print('SHIFTED AXES (test differs from train), largest first:')
    for _, a, d in sorted(flags, reverse=True):
        direction = 'HIGHER' if d > 0 else 'LOWER'
        print('  %-16s test is %.2f SD %s than train' % (a, abs(d), direction))
    print('\nThis is the mechanism behind your 3.4 ft train->test gap. The model is')
    print('fine (OOF 5.5); it is being applied off-distribution. Next step: condition')
    print('v22 on the shifted axis, or reweight training to match the test marginal.')
print('=' * 62)

# --- the single most actionable cut: error vs blind_len, if we have OOF -------
try:
    import pickle
    _h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
    if _h:
        E = pickle.load(open(_h[0], 'rb'))['err']
        m = tr[tr.well.isin(E)].copy()
        m['err'] = m.well.map(E)
        from scipy.stats import spearmanr
        for a in ['blind_len', 'blind_frac', 'gr_kb_shift', 'blind_md_span']:
            ok = m[a].notna() & m['err'].notna()
            if ok.sum() > 30:
                rho, p = spearmanr(m.loc[ok, a], m.loc[ok, 'err'])
                tag = 'STRONG' if abs(rho) > 0.3 and p < 0.01 else 'weak' if abs(rho) > 0.15 else ''
                print('train: spearman(%s, v22 err) = %+.3f (p=%.1e) %s' % (a, rho, p, tag))
        print('\nIf an axis is BOTH shifted test-vs-train AND correlated with error on')
        print('train, that is your highest-leverage target: test wells sit where the')
        print('model is known to be weak, and you can see it coming.')
except Exception as e:
    print('(OOF cross-tab skipped: %r)' % e)

te.to_csv('shift_report_test.csv', index=False)
tr.to_csv('shift_report_train.csv', index=False)
print('\nwrote shift_report_{train,test}.csv')
