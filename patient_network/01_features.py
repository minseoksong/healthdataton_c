"""Synthetic K-MIMIC: 입실 후 0-24h 관점별 환자 피처 생성 (view별 CSV)"""
import pandas as pd, numpy as np, sys, os
SRC = sys.argv[1]   # pkl 폴더 (xlsx -> pkl 변환본)
OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
W = 24  # 관찰 시간창 (h)
r = lambda n: pd.read_pickle(f"{SRC}/{n}.pkl")

icu = r('syn_icustays'); adm = r('syn_admissions'); pat = r('syn_patients')
icu['intime'] = pd.to_datetime(icu.intime); icu['outtime'] = pd.to_datetime(icu.outtime)
base = icu[['subject_id','hadm_id','stay_id','intime','outtime','first_careunit','op_flag']].merge(
    adm[['hadm_id','admission_type','hospital_expire_flag','icu_expire_flag','admittime','dischtime']], on='hadm_id').merge(
    pat[['subject_id','sex','anchor_age']], on='subject_id')
base['icu_los_h'] = (base.outtime - base.intime).dt.total_seconds()/3600
base['hosp_los_d'] = (pd.to_datetime(base.dischtime) - pd.to_datetime(base.admittime)).dt.total_seconds()/86400
base = base.set_index('stay_id').sort_index()
ids = base.index

def window(df, tcol, key='stay_id'):
    df = df.rename(columns={key: 'stay_id'}).merge(base[['intime']], left_on='stay_id', right_index=True)
    df['rel'] = (pd.to_datetime(df[tcol]) - df['intime']).dt.total_seconds()/3600
    return df[(df.rel >= 0) & (df.rel < W)]

def summarize(df, item, val, names, stats=('mean','min','max','last')):
    df = df[df[item].isin(names)].sort_values(['stay_id','rel'])
    g = df.groupby(['stay_id', item])[val]
    parts = {s: (g.last() if s=='last' else g.agg(s)) for s in stats}
    wide = pd.concat({s: p.unstack() for s, p in parts.items()}, axis=1)
    wide.columns = [f"{names[c]}_{s}" for s, c in wide.columns]
    return wide.reindex(ids)

# ---------- View A: 인구학·입원·진단 ----------
A = pd.DataFrame(index=ids)
A['age'] = base.anchor_age; A['sex_M'] = (base.sex=='M').astype(int)
A = A.join(pd.get_dummies(base.first_careunit.fillna('Unknown'), prefix='unit').astype(int))
A = A.join(pd.get_dummies(base.admission_type, prefix='adm').astype(int))
A = A.join(pd.get_dummies(base.op_flag, prefix='op').astype(int))
dx = r('syn_diagnoses_icd'); dx = dx[dx.icd_code!='Others'].merge(base.reset_index()[['hadm_id','stay_id']], on='hadm_id')
A = A.join(pd.crosstab(dx.stay_id, dx.icd_code).clip(upper=1).add_prefix('dx_').reindex(ids).fillna(0))

# ---------- View B: 활력징후·의식 ----------
ce = r('syn_chartevents'); ce['v'] = pd.to_numeric(ce.valuenum, errors='coerce'); ce = window(ce, 'charttime')
VIT = {'001C_1021_25105':'HR','001C_1023_27310':'RR','001C_1012_24795':'SBP','001C_1013_25640':'DBP',
       '001C_1014_27095':'MBP','001C_1015_24795':'ASBP','001C_1017_27095':'AMBP','001C_1026_26520':'BT',
       '001C_1003_24480':'SpO2','001C_1187_20765':'GCS_E','001C_1187_20770':'GCS_M','001C_1187_20775':'GCS_V'}
B = summarize(ce, 'itemid', 'v', VIT, ('mean','min','max','last','std'))
B = B[[c for c in B if not (c.startswith('GCS') and c.endswith(('_std','_mean')))]]

# ---------- View C: 검사 ----------
lab = window(r('syn_labevents'), 'charttime')
LAB = {'001L30821':'pH','001L30822':'pCO2','001L30823':'pO2','001L30824':'HCO3','001L3442':'Lactate',
       '001L3005':'Glucose','001L3043':'Na','001L3044':'K','001L3041':'Cr','001L3006':'BUN','001L30314':'eGFR',
       '001L2001':'WBC','001L2003':'Hb','001L2009':'PLT','001L22031':'INR','001L2204':'aPTT','001L3011':'Bil',
       '001L3010':'Alb','001L3014':'AST','001L3092':'CRP','001L3085':'TnI','001L3084':'CKMB'}
C = summarize(lab, 'itemid', 'valuenum', LAB, ('min','max','last'))

# ---------- View D: 치료·호흡 ----------
D = pd.DataFrame(index=ids)
em = window(r('syn_emar'), 'charttime')
MED = {'Norpin':'norepi','Vasopressin':'vasopressin','Epinephrine':'epi','Humulin':'insulin','Fresofol':'propofol',
       'Remiva':'remifentanil','Precedex':'dexmed','Lasix':'furosemide'}
for k, v in MED.items():
    D[f'med_{v}'] = em[em.medication.str.contains(k)].groupby('stay_id').size().reindex(ids).fillna(0).clip(upper=1)
D['n_vasoactive'] = D[['med_norepi','med_vasopressin','med_epi']].sum(axis=1)
ie = window(r('syn_inputevents'), 'starttime', 'icustay_id')
INP = {'001I_1315_26175':'in_iv','001I_1301_23155':'in_oral','001I_1302_21090':'in_ltube'}
D = D.join(ie[ie.itemid.isin(INP)].assign(k=lambda d: d.itemid.map(INP)).pivot_table(index='stay_id', columns='k', values='amount', aggfunc='sum').reindex(ids).fillna(0))
oe = window(r('syn_outputevents'), 'charttime', 'icustay_id')
URINE = ['001O_1479_25880','001O_1481_25470','001O_1496_23625']
D['out_urine'] = oe[oe.itemid.isin(URINE)].groupby('stay_id').value.sum().reindex(ids).fillna(0)
D['out_drain'] = oe[~oe.itemid.isin(URINE + ['001O_1804_26960'])].groupby('stay_id').value.sum().reindex(ids).fillna(0)
D['crrt'] = (oe[oe.itemid=='001O_1804_26960'].groupby('stay_id').size().reindex(ids).fillna(0) > 0).astype(int)
D['fluid_balance'] = D[['in_iv','in_oral','in_ltube']].sum(axis=1) - D.out_urine - D.out_drain
mode = ce[ce.itemid=='001C_1810_22385'].copy()
mode['m'] = mode.value.astype(str).replace({'SIMV(PC)':'SIMV-PC','PC(AC)':'PC','A/C':'AC','PS/CPAP':'PS','PSV':'PS'})
mode = mode[~mode.m.isin(['STANDBY','Others'])]
D = D.join(pd.crosstab(mode.stay_id, mode.m, normalize='index').add_prefix('vent_').reindex(ids).fillna(0))
D['vent_any'] = D.filter(like='vent_').sum(axis=1).gt(0).astype(int)
VENT = {'001C_1894_20720':'FiO2','001C_1567_21330':'PEEP','001C_1728_22200':'PIP'}
D = D.join(summarize(ce, 'itemid', 'v', VENT, ('max',)))

# ---------- 저장 ----------
for name, df in zip('ABCD', [A, B, C, D]):
    df.to_csv(f"{OUT}/view_{name}.csv"); print(name, df.shape, 'missing% =', round(df.isna().mean().mean()*100,1))
base[['subject_id','hadm_id','first_careunit','op_flag','admission_type','sex','anchor_age',
      'hospital_expire_flag','icu_expire_flag','icu_los_h','hosp_los_d']].to_csv(f"{OUT}/outcomes.csv")
