"""Synthetic_K-MIMIC xlsx -> pkl 변환 (이후 스크립트 입력). 사용: python 00_convert.py <Synthetic_K-MIMIC 폴더> <pkl 폴더>"""
import pandas as pd, glob, os, sys
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
for f in sorted(glob.glob(f"{src}/syn_*.xlsx")):
    n = os.path.basename(f)[:-5]; pd.read_excel(f, engine="calamine").to_pickle(f"{out}/{n}.pkl"); print(n)
