# ===== L4: rank ALL damper presets on OOF locally, one free run =====
# Local ordering is displaced toward LESS regularization than hidden (measured:
# local S*=0.05 vs hidden S*=0.13). Rule of thumb from that displacement: take
# the local argmin and go ~2-3x bolder hidden-side. WHERE: L1/L3 notebook. ~3 min.
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

def _ma(x, win):
    out = np.convolve(x, np.ones(win)/win, 'same')
    h = win // 2; out[:h] = np.nan; out[-h:] = np.nan
    return out

W, t0 = [], time.time()
for w, pv in PRED.items():
    try:
        h = pd.read_csv(os.path.join(DATA, 'train', '%s__horizontal_well.csv' % w),
                        usecols=['TVT'])
        tr = h['TVT'].values.astype(float)
        idx = np.asarray(pv['blind_idx']); that = np.asarray(pv['tvt_hat'], float)
        m = np.isfinite(tr[idx]) & np.isfinite(that)
        if m.sum() < 700: continue
        o = np.argsort(idx[m])
        W.append((idx[m][o], that[m][o], tr[idx[m]][o]))
    except Exception: pass
print('%d wells [%.0fs]' % (len(W), time.time()-t0))
base = np.mean([np.abs(t_-tr_).mean() for _,t_,tr_ in W])
print('baseline OOF %.4f\n' % base)

def apply(v, idx, mode, P0, H, WIN):
    pos = (idx-idx.min())/max(idx.max()-idx.min(),1)
    lp = _ma(v, WIN); fin = np.isfinite(lp)
    wh = np.full(len(v),H) if mode=='flat' else np.clip((pos-P0)/max(1-P0,1e-9),0,1)*H
    new = v.copy(); new[fin] = lp[fin] + (1-wh[fin])*(v[fin]-lp[fin])
    return new

GRID = [('ramp',0.40,0.15,301,'v32_anchor'),
        ('flat',0.00,0.10,301,''), ('flat',0.00,0.15,301,''),
        ('flat',0.00,0.25,301,'v33'), ('flat',0.00,0.40,301,'v34'),
        ('flat',0.00,0.60,301,''),
        ('flat',0.00,0.25,151,''), ('flat',0.00,0.25,601,'v35'),
        ('ramp',0.00,0.50,301,'')]
print('%-6s %5s %5s %5s %10s %10s  %s' % ('mode','P0','H','WIN','OOF','delta','tag'))
for mode,P0,H,WIN,tag in GRID:
    tot=[np.abs(apply(t_.copy(),i_.copy(),mode,P0,H,WIN)-tr_).mean() for i_,t_,tr_ in W]
    m=np.mean(tot)
    print('%-6s %5.2f %5.2f %5d %10.4f %+10.4f  %s' % (mode,P0,H,WIN,m,m-base,tag))
print('\nREAD with the displacement rule: hidden optimum has sat at ~2-3x the local')
print('optimum on this mechanism. Local argmin at H_loc -> probe hidden at ~2*H_loc.')
