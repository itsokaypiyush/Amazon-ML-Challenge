import pandas as pd, numpy as np, time, sys
from src.preprocessing.normalize import normalize_company_name, normalize_address
t0 = time.time()
STOP = {"incorporated", "corporation", "company", "limited", "liability", "private", "partnership",
        "public", "the", "and", "of", "societe", "a", "responsabilite", "limitee", "sociedad", "anonima"}
def load(path, n=None):
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"]).fillna("")
    if n: df = df.sample(n=n, random_state=0)
    ks, ns = [], []
    for a, c, nm in zip(df.business_address, df.country, df.business_name):
        toks = sorted(set(normalize_address(a).split()))
        ks.append((" ".join(toks) + "|" + c) if len(toks) >= 3 else "")
        ns.append(frozenset(t for t in normalize_company_name(nm).split() if t not in STOP))
    df["k"], df["nt"] = ks, ns
    return df[df.k != ""][["entity_id", "k", "nt"]]
def join(s1, t):
    cnt = s1.k.value_counts(); u = s1[s1.k.isin(cnt.index[cnt == 1])]
    m = t.merge(u.rename(columns={"entity_id": "s1", "nt": "s1nt"}), on="k")
    return m[[bool(a & b) for a, b in zip(m.nt, m.s1nt)]][["s1", "entity_id"]]
D = "data/student_resource/dataset/train"
s1 = load(f"{D}/train_source1.tsv")
t = pd.concat([load(f"{D}/train_source2.tsv", 1_000_000), load(f"{D}/train_source3.tsv", 1_000_000)])
m = join(s1, t)
gt = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
g = gt.assign(t=gt.matched_entity_ids.str.split(",")).explode("t"); g = g[g.t != ""]
own = pd.Series(g.source1_entity_id.values, index=g.t.values)
prec = (m.entity_id.map(own) == m.s1).mean()
print(f"[train] address-key rule: pairs={len(m)} precision={prec:.4f} ({time.time()-t0:.0f}s)", flush=True)
if prec < 0.95: sys.exit("precision too low -> NOT applying")
D = "data/student_resource/dataset/test"
s1 = load(f"{D}/test_source1.tsv")
t = pd.concat([load(f"{D}/test_source2.tsv"), load(f"{D}/test_source3.tsv")])
m = join(s1, t)
cur = pd.read_csv("output/matching_results_v4.tsv", sep="\t", dtype=str).fillna("")
p = cur.assign(t=cur.matched_entity_ids.str.split(",")).explode("t"); p = p[p.t != ""]
new = m[~m.entity_id.isin(set(p.t))].drop_duplicates("entity_id")
print(f"[test] new pairs={len(new)} S1 gaining={new.s1.nunique()} ({time.time()-t0:.0f}s)", flush=True)
allp = pd.concat([p.rename(columns={"source1_entity_id": "s1"})[["s1", "t"]], new.rename(columns={"entity_id": "t"})[["s1", "t"]]]).drop_duplicates()
lists = allp.sort_values("t").groupby("s1").t.agg(",".join)
out = pd.DataFrame({"source1_entity_id": cur.source1_entity_id}); out["matched_entity_ids"] = out.source1_entity_id.map(lists).fillna("")
out.to_csv("output/matching_results_v5.tsv", sep="\t", index=False)
cand = pd.read_csv("output/candidate_pairs_v4.tsv", sep="\t", dtype=str).fillna("")
add = new.groupby("s1").entity_id.agg(",".join); extra = cand.source1_entity_id.map(add).fillna(""); rows = (extra != "").to_numpy()
def mi(a, b):
    s = a.split(",") if a else []; seen = set(s)
    for x in b.split(","):
        if x and x not in seen: s.append(x); seen.add(x)
    return ",".join(s)
cand.loc[rows, "candidate_entity_ids"] = [mi(a, b) for a, b in zip(cand.candidate_entity_ids[rows], extra[rows])]
cand.to_csv("output/candidate_pairs_v5.tsv", sep="\t", index=False)
print(f"[test] wrote v5 ({time.time()-t0:.0f}s)", flush=True)
