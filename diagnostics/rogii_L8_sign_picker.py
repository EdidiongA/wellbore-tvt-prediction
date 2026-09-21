# ===== L8: pick v49's first DELTA sign from free local data (30 seconds) =====
# Median SIGNED toe residual of v22's OOF: if predictions run HIGH on training
# toes (median > 0), set DELTA negative first; if LOW, positive. WHERE: L1/L3 notebook.
import numpy as np, pickle, glob, os
import pandas as pd
_h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
PRED = pickle.load(open(_h[0], 'rb'))['pred']
DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'train')): DATA = c; break
meds = []
for w, pv in PRED.items():
    try:
        tr = pd.read_csv(os.path.join(DATA, 'train', '%s__horizontal_well.csv' % w),
                         usecols=['TVT'])['TVT'].values.astype(float)
        idx = np.asarray(pv['blind_idx']); that = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(tr[idx]) & np.isfinite(that)
        if m.sum() < 300: continue
        o = np.argsort(idx[m]); i2 = idx[m][o]
        r = that[m][o] - tr[i2]
        meds.append(np.median(r[int(0.7*len(r)):]))     # toe 30%
    except Exception: pass
meds = np.array(meds)
print('%d wells | median of per-well toe signed residual: %+.3f ft' % (len(meds), np.median(meds)))
print('fraction of wells with positive toe residual: %.1f%%' % (100*(meds > 0).mean()))
print('\nRULE: median > +0.3 -> predictions run HIGH -> set DELTA = -1.5 first')
print('      median < -0.3 -> predictions run LOW  -> keep DELTA = +1.5')
print('      |median| <= 0.3 -> no strong local prior; keep +1.5 (still informative)')
