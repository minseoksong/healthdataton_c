"""군집별 특성·결과 요약 + 군집을 가장 잘 구분하는 피처(표준화 평균차)"""
import pandas as pd, numpy as np, sys
from scipy.stats import chi2_contingency, kruskal
D_ = sys.argv[1]
nodes = pd.read_csv(f"{D_}/nodes.csv", index_col=0)
X = pd.concat([pd.read_csv(f"{D_}/view_{v}.csv", index_col=0).add_prefix(f"{v}:") for v in 'ABCD'], axis=1).loc[nodes.index]
cl = nodes.cluster
prof = nodes.groupby('cluster').agg(n=('x','size'), age=('anchor_age','median'), male=('sex', lambda s: (s=='M').mean()),
    hosp_mort=('hospital_expire_flag','mean'), icu_mort=('icu_expire_flag','mean'), icu_los_h=('icu_los_h','median'),
    surgery=('op_flag', lambda s: (s>0).mean()), ed_adm=('admission_type', lambda s: (s=='Emergency department').mean()),
    top_unit=('first_careunit', lambda s: ', '.join(f"{k} {v:.0%}" for k, v in s.value_counts(normalize=True).head(2).items())))
key = {'B:HR_mean':'HR', 'B:MBP_mean':'MBP', 'B:GCS_M_min':'GCS_M_min', 'C:Lactate_max':'Lactate', 'C:Cr_max':'Cr', 'C:BUN_max':'BUN',
       'C:aPTT_max':'aPTT', 'C:PLT_min':'PLT', 'D:fluid_balance':'fluid_bal'}
for c, nm in key.items(): prof[nm] = X[c].groupby(cl).median()
for c, nm in {'D:n_vasoactive':'vasoactive', 'D:vent_any':'vent', 'D:crrt':'CRRT', 'A:dx_N179':'AKI_dx', 'A:dx_I509':'HF_dx', 'A:dx_I4890':'AF_dx'}.items():
    prof[nm] = (X[c] > 0).groupby(cl).mean()
prof.round(3).to_csv(f"{D_}/cluster_profiles.csv"); print(prof.round(2).T.to_string())
p = chi2_contingency(pd.crosstab(cl, nodes.hospital_expire_flag))[1]; print('mortality chi2 p =', p)
Z = (X - X.mean()) / X.std().replace(0, 1)
rows = []
for c in sorted(cl.unique()):
    d = Z[cl == c].mean().dropna().sort_values()
    rows.append({'cluster': c, 'high': ', '.join(f"{k.split(':')[1]}(+{v:.2f})" for k, v in d.tail(6)[::-1].items()),
                 'low': ', '.join(f"{k.split(':')[1]}({v:.2f})" for k, v in d.head(4).items())})
pd.DataFrame(rows).to_csv(f"{D_}/cluster_markers.csv", index=False); print(pd.DataFrame(rows).to_string())
