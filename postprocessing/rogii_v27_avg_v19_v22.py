# ===== v27: average v19 + v22 predictions in one scored run =====
# This must run BOTH models inside one notebook, because the hidden test set only
# exists during the scored rerun -- you cannot average two old CSVs (they only
# cover the 3 example wells).
#
# SETUP (one-time):
#   1. Copy your REAL v22 notebook -> rename v27.
#   2. In Kaggle, open your notebook's Version 19 (the one that scored 8.926).
#      Copy the FULL SOURCE of its model-code cell (cell 3) into V19_MODEL_SRC
#      below, and the full source of its field-build cell (cell 7) into
#      V19_FIELD_SRC. Keep the r''' ... ''' quoting exactly.
#   3. Paste this as the LAST cell, AFTER the submission cell.
#   4. Save & Run All -> Submit.
#
# The cell runs v19 in an isolated namespace (no collision with v22's globals),
# predicts every test well again, and averages 50/50 with the v22 submission.

V19_MODEL_SRC = r'''
# <<< PASTE v19 CELL 3 (model code) HERE >>>
'''

V19_FIELD_SRC = r'''
# <<< PASTE v19 CELL 7 (spatial map + structural field build) HERE >>>
'''

import os, glob, time
import numpy as np, pandas as pd

assert 'PASTE v19' not in V19_MODEL_SRC, 'paste v19 cell 3 into V19_MODEL_SRC first'

NS = {'__name__': 'v19_iso'}
exec(V19_MODEL_SRC, NS)
if 'PASTE v19' not in V19_FIELD_SRC:
    exec(V19_FIELD_SRC, NS)          # builds v19's field/map inside the namespace
print('v19 namespace loaded:',
      [k for k in ('predict_well', 'predict_well_diag', 'load_well') if k in NS])

_pred19 = NS.get('predict_well_diag') or NS.get('predict_well')
_load19 = NS['load_well']

sub = pd.read_csv('submission.csv')
sub['well'] = sub['id'].astype(str).str[:8]
sub['ridx'] = sub['id'].astype(str).str.split('_').str[-1].astype(int)
print('v22 submission: %d rows, %d wells' % (len(sub), sub['well'].nunique()))

t0, done, failed = time.time(), 0, 0
for w, grp in sub.groupby('well'):
    try:
        h, t = _load19('test', w)
        out = _pred19(h, t)
        p19 = np.asarray(out[0] if isinstance(out, tuple) else out, float)
        idx = grp['ridx'].values
        ok = (idx >= 0) & (idx < len(p19)) & np.isfinite(p19[np.clip(idx, 0, len(p19)-1)])
        merged = grp['tvt'].values.astype(float)
        merged[ok] = 0.5 * merged[ok] + 0.5 * p19[idx[ok]]
        sub.loc[grp.index, 'tvt'] = merged
        done += 1
    except Exception as e:
        failed += 1
        if failed <= 3:
            print('  v19 failed on %s (%r) -- keeping pure v22 there' % (w, e))
    if done and done % 25 == 0:
        print('  %d wells averaged [%.0fs]' % (done, time.time() - t0), flush=True)

print('\naveraged %d wells | v19-failed (kept v22): %d' % (done, failed))
sub[['id', 'tvt']].to_csv('submission.csv', index=False)
print('submission.csv rewritten (v27 = 0.5*v19 + 0.5*v22)')
