# ===== L6: FREE empirical sign check before spending any slot =====
# (1) per-station correlation between v36's correction and v28's correction on
#     training wells -- must be strongly POSITIVE (v36 is v28's Z-part).
# (2) local MAE of the v36 damper at BOTH signs -- plus should be ~neutral
#     locally (hidden amplifies ~54x on this family), minus clearly harmful.
# WHERE: L1/L3 notebook (v22_oof.pkl + competition data). ~2 min.
import numpy as np, pickle, glob, os, time
import pandas as pd

_h = glob.glob('/kaggle/input/**/v22_oof.pkl', recursive=True) + glob.glob('v22_oof.pkl')
PRED = pickle.load(open(_h[0], 'rb'))['pred']
DATA = None
for c in ['/kaggle/input/competitions/rogii-wellbore-geology-prediction',
          '/kaggle/input/rogii-wellbore-geology-prediction']:
    if os.path.isdir(os.path.join(c, 'train')):
        DATA = c; break
assert DATA

W_TOE, RAMP_P0, FIT_P0 = 0.20, 0.40, 0.40
cors, t0 = [], time.time()
mae = {'base': [], 'plus': [], 'minus': [], 'v28': []}
for w, pv in PRED.items():
    try:
        h = pd.read_csv(os.path.join(DATA, 'train', '%s__horizontal_well.csv' % w),
                        usecols=['TVT', 'Z'])
        tr = h['TVT'].values.astype(float); Z = h['Z'].values.astype(float)
        idx = np.asarray(pv['blind_idx']); that = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(tr[idx]) & np.isfinite(that) & np.isfinite(Z[idx])
        if m.sum() < 600: continue
        o = np.argsort(idx[m])
        i_s = idx[m][o]; v = that[m][o]; z = Z[idx[m]][o]; t_true = tr[idx[m]][o]
        pos = (i_s - i_s.min()) / max(i_s.max() - i_s.min(), 1)
        near = pos <= FIT_P0
        a = np.polyfit(pos[near], z[near], 1)
        r = z[near] - np.polyval(a, pos[near]); k = np.abs(r) <= 3*max(np.std(r),1e-6)
        if k.sum() >= 10: a = np.polyfit(pos[near][k], z[near][k], 1)
        dev = z - np.polyval(a, pos)
        wgt = np.clip((pos - RAMP_P0)/(1-RAMP_P0), 0, 1) * W_TOE
        c36 = wgt * dev
        # v28's correction on the same well
        a2 = np.polyfit(pos[near], v[near], 1)
        r2 = v[near] - np.polyval(a2, pos[near]); k2 = np.abs(r2) <= 3*max(np.std(r2),1e-6)
        if k2.sum() >= 10: a2 = np.polyfit(pos[near][k2], v[near][k2], 1)
        w28 = np.clip((pos - 0.40)/0.60, 0, 1) * 0.15
        c28 = w28 * (np.polyval(a2, pos) - v)
        toe = pos > 0.40
        if toe.sum() > 100 and np.std(c36[toe]) > 1e-9 and np.std(c28[toe]) > 1e-9:
            cors.append(np.corrcoef(c36[toe], c28[toe])[0, 1])
        mae['base'].append(np.abs(v - t_true).mean())
        mae['plus'].append(np.abs(v + c36 - t_true).mean())
        mae['minus'].append(np.abs(v - c36 - t_true).mean())
        mae['v28'].append(np.abs(v + c28 - t_true).mean())
    except Exception:
        pass
print('%d wells [%.0fs]' % (len(mae['base']), time.time()-t0))
print('\nper-station corr(c36, c28) on toe: median %.3f  p25 %.3f'
      % (np.median(cors), np.percentile(cors, 25)))
b = np.mean(mae['base'])
print('\nlocal MAE  base   %.4f' % b)
print('local MAE  v28    %.4f  (%+.4f)  [L1 said ~+0.05 at this dose]' % (np.mean(mae['v28']), np.mean(mae['v28'])-b))
print('local MAE  PLUS   %.4f  (%+.4f)  <- ship sign' % (np.mean(mae['plus']), np.mean(mae['plus'])-b))
print('local MAE  MINUS  %.4f  (%+.4f)' % (np.mean(mae['minus']), np.mean(mae['minus'])-b))
print('\nGO if: corr strongly positive AND plus <= minus. Otherwise STOP and report.')
