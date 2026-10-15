"""view별 유사도 -> SNF 융합 -> kNN 환자 네트워크 -> Leiden 커뮤니티 -> 결과 비교"""
import pandas as pd, numpy as np, sys, os, json
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import igraph as ig, leidenalg, networkx as nx
D_ = sys.argv[1]; K = 20; MU = 0.5; T = 20; KG = 15
views = {v: pd.read_csv(f"{D_}/view_{v}.csv", index_col=0) for v in 'ABCD'}
NAMES = {'A':'인구학·진단','B':'활력징후','C':'검사','D':'치료·호흡'}
out = pd.read_csv(f"{D_}/outcomes.csv", index_col=0); y = out.hospital_expire_flag.values
n = len(out)

def prep(df):
    X = df.copy()
    X = X.clip(X.quantile(.01), X.quantile(.99), axis=1)
    X = X.fillna(X.median())
    X = (X - X.mean()) / X.std().replace(0, 1)
    return X.fillna(0).values

def affinity(X):  # scaled exponential kernel (Wang et al. 2014)
    Dm = cdist(X, X); Ds = np.sort(Dm, axis=1)[:, 1:K+1].mean(1)
    eps = (Ds[:, None] + Ds[None, :] + Dm) / 3
    W = np.exp(-Dm**2 / (MU * eps)); return (W + W.T) / 2

def normP(W):
    W = W.copy(); np.fill_diagonal(W, 0); P = W / (2 * W.sum(1, keepdims=True)); np.fill_diagonal(P, .5); return (P + P.T) / 2
def knnS(W):
    W = W.copy(); np.fill_diagonal(W, 0); idx = np.argsort(-W, axis=1)[:, :K]
    S = np.zeros_like(W); rows = np.arange(len(W))[:, None]; S[rows, idx] = W[rows, idx]
    return S / S.sum(1, keepdims=True)

def snf(Ws):
    Ps = [normP(W) for W in Ws]; Ss = [knnS(W) for W in Ws]
    for _ in range(T):
        Ps = [normP(Ss[i] @ (sum(Ps[j] for j in range(len(Ps)) if j != i) / (len(Ps)-1)) @ Ss[i].T) for i in range(len(Ps))]
    return sum(Ps) / len(Ps)

Ws = {v: affinity(prep(df)) for v, df in views.items()}
F = snf(list(Ws.values())); np.fill_diagonal(F, 0)

def nbr_auc(W, k=KG):  # leave-one-out: 이웃 k명의 사망 가중평균으로 본인 사망 예측
    W = W.copy(); np.fill_diagonal(W, 0); idx = np.argsort(-W, axis=1)[:, :k]
    w = np.take_along_axis(W, idx, 1); s = (w * y[idx]).sum(1) / w.sum(1); return roc_auc_score(y, s)

res = {f"{NAMES[v]}": round(nbr_auc(W), 3) for v, W in Ws.items()}
res['SNF 융합'] = round(nbr_auc(F), 3)
Xall = np.hstack([prep(df) for df in views.values()])
lr = cross_val_predict(LogisticRegression(C=.05, max_iter=3000), Xall, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method='predict_proba')[:, 1]
res['(참고) 로지스틱, 전체 피처 5-fold'] = round(roc_auc_score(y, lr), 3)
print('mortality AUC', res)

# kNN 그래프 (mutual 아님: 상호 합집합, 가중치=융합 유사도)
idx = np.argsort(-F, axis=1)[:, :KG]; E = {}
for i in range(n):
    for j in idx[i]: a, b = min(i, j), max(i, j); E[(a, b)] = F[a, b]
edges = pd.DataFrame([(a, b, w) for (a, b), w in E.items()], columns=['s', 't', 'w']); edges['w'] /= edges.w.max()
g = ig.Graph(n=n, edges=list(zip(edges.s, edges.t))); g.es['weight'] = edges.w.tolist()

def leiden(res_, seed): return np.array(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=res_, seed=seed).membership)
scan = []
for rp in [0.5, 0.6, 0.7, 0.8, 0.9]:
    ms = [leiden(rp, s) for s in range(10)]
    ari = np.mean([adjusted_rand_score(ms[0], m) for m in ms[1:]])
    big = (np.bincount(ms[0]) >= 20).sum()
    scan.append((rp, int(len(set(ms[0]))), int(big), float(round(ari, 3)))); print('res', rp, 'clusters', len(set(ms[0])), 'size>=20', big, 'ARI', round(ari, 3))
rp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
# 합의(consensus) 군집: 50개 seed에서 같은 군집에 묶인 비율로 edge 가중치를 재조정한 뒤 다시 Leiden
runs = np.array([leiden(rp, s) for s in range(50)])
co = np.array([(runs[:, a] == runs[:, b]).mean() for a, b in zip(edges.s, edges.t)])
g2 = ig.Graph(n=n, edges=list(zip(edges.s, edges.t))); g2.es['weight'] = (edges.w.values * co).tolist()
memb = np.array(leidenalg.find_partition(g2, leidenalg.RBConfigurationVertexPartition, weights='weight', resolution_parameter=rp, seed=0).membership)
stab = float(np.mean([adjusted_rand_score(memb, r_) for r_ in runs])); print('consensus vs runs ARI', round(stab, 3))
# 20명 미만 소군집은 가장 강하게 연결된 큰 군집으로 병합
sizes = np.bincount(memb)
for c in np.where(sizes < 20)[0]:
    for i in np.where(memb == c)[0]:
        nb = edges[(edges.s == i) | (edges.t == i)]; o = np.where(nb.s == i, nb.t, nb.s)
        cand = pd.Series(nb.w.values, index=memb[o]); cand = cand[cand.index.map(lambda x: sizes[x] >= 20)]
        if len(cand): memb[i] = cand.groupby(level=0).sum().idxmax()
_, memb = np.unique(memb, return_inverse=True)
# 사망률 순으로 군집 번호 재정렬 (C1 = 가장 낮음)
mr = pd.Series(y).groupby(memb).mean().sort_values(); remap = {c: i+1 for i, c in enumerate(mr.index)}; memb = np.array([remap[c] for c in memb])
out['cluster'] = memb
print('modularity', round(g.modularity(memb - 1, weights='weight'), 3))

# 2D 배치: 융합 유사도를 거리로 바꿔 UMAP (precomputed)
import umap
Dist = 1 - F / F.max(); np.fill_diagonal(Dist, 0)
emb = umap.UMAP(n_neighbors=KG, min_dist=0.25, metric='precomputed', random_state=7).fit_transform(Dist)
out['x'] = emb[:, 0]; out['y'] = emb[:, 1]
np.save(f"{D_}/fused_similarity.npy", F)
out.to_csv(f"{D_}/nodes.csv"); edges.to_csv(f"{D_}/edges.csv", index=False)
json.dump({'auc': res, 'scan': scan, 'resolution': rp, 'stability_ari': stab, 'modularity': float(g.modularity(memb - 1, weights='weight'))}, open(f"{D_}/metrics.json", 'w'), ensure_ascii=False, indent=1)
