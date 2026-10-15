# 환자 유사도 네트워크 (Synthetic K-MIMIC)

ICU 입실 후 0–24시간 데이터를 4개 관점(인구학·진단 / 활력징후 / 검사 / 치료·호흡)으로 요약하고,
관점별 유사도를 Similarity Network Fusion(SNF)으로 합친 뒤 kNN(k=15) 그래프와 Leiden 합의 군집으로 환자군을 찾습니다.

## 실행
    pip install pandas python-calamine scikit-learn scipy igraph leidenalg umap-learn
    python 00_convert.py ../Synthetic_K-MIMIC pkl
    python 01_features.py pkl out
    python 02_snf_network.py out 0.6      # 0.6 = Leiden resolution
    python 03_profiles.py out

## 결과 파일 (out/)
- view_A~D.csv: 관점별 환자 피처 (stay_id 기준)
- outcomes.csv: 사망·재원기간 등 결과 변수
- fused_similarity.npy: SNF 융합 유사도 행렬 (1328×1328, stay_id 오름차순)
- nodes.csv: 환자별 군집 번호와 2D 좌표(UMAP) / edges.csv: kNN 엣지 (s,t는 nodes.csv의 행 순서)
- cluster_profiles.csv, cluster_markers.csv: 군집별 특성과 구분 피처
- metrics.json: 관점별 이웃 기반 사망 AUC, resolution 스캔, 안정성(ARI), modularity

## 주요 설정
- SNF: K=20, mu=0.5, 20회 반복 (Wang et al., Nat Methods 2014)
- 결측: 1–99% 절단 후 중앙값 대체, 표준화
- 20명 미만 소군집은 가장 강하게 연결된 큰 군집으로 병합, 군집 번호는 사망률 오름차순
