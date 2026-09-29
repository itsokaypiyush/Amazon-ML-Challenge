import pandas as pd, numpy as np, time, sys
from src.preprocessing.normalize import normalize_company_name
t0 = time.time()
STOP = {"incorporated", "corporation", "company", "limited", "liability", "private", "partnership",
        "public", "the", "and", "of", "societe", "a", "responsabilite", "limitee", "sociedad", "anonima"}

def keys(names, countries):
    out = []
    for n, c in zip(names, countries):
        toks = sorted(set(t for t in normalize_company_name(n).split() if t not in STOP))
        out.append((" ".join(toks) + "|" + c) if toks else "")
    return out

def load(path, n=None):
    df = pd.read_csv(path, sep="\t", dtype=str, usecols=["entity_id", "business_name", "business_address", "country"]).fillna("")
    if n:
        df = df.sample(n=n, random_state=0)
    df["k"] = keys(df.business_name.tolist(), df.country.tolist())
    df["nums"] = df.business_address.str.findall(r"\d+").map(frozenset)
    return df[df.k != ""][["entity_id", "k", "nums"]]

def unique_join(s1, t):
    cnt = s1.k.value_counts()
    u = s1[s1.k.isin(cnt.index[(cnt >= 2) & (cnt <= 20)])]
    t = t[t.nums.map(len) > 0]
    m = t.merge(u.rename(columns={"entity_id": "s1", "nums": "s1nums"}), on="k")
    m = m[[bool(a & b) for a, b in zip(m.nums, m.s1nums)]]
    m = m[m.groupby("entity_id").s1.transform("size") == 1]
    share = np.array([bool(a & b) for a, b in zip(m.nums, m.s1nums)])
    t_no_num = np.array([len(a) == 0 for a in m.nums])
    m["ruleA"] = share
    m["ruleB"] = share | t_no_num
    return m[["s1", "entity_id", "ruleA", "ruleB"]]

# ---- 1) precision on TRAINING (full source1, 2M sampled targets) ----
D = "data/student_resource/dataset/train"
s1 = load(f"{D}/train_source1.tsv")
t = pd.concat([load(f"{D}/train_source2.tsv", 1_000_000), load(f"{D}/train_source3.tsv", 1_000_000)])
m = unique_join(s1, t)
gt = pd.read_csv(f"{D}/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
g = gt.assign(t=gt.matched_entity_ids.str.split(",")).explode("t")
g = g[g.t != ""]
own = pd.Series(g.source1_entity_id.values, index=g.t.values)
correct = (m.entity_id.map(own) == m.s1)
n_owned = t.entity_id.isin(own.index).sum()
RULE = None
for r in ("ruleA",):
    pr = correct[m[r]].mean()
    print(f"[train] {r}: pairs={int(m[r].sum())} precision={pr:.4f} recall-of-owned={correct[m[r]].sum()/n_owned:.3f}", flush=True)
    if RULE is None and pr >= 0.95:
        RULE = r
print(f"[train] chosen rule: {RULE} ({time.time()-t0:.0f}s)", flush=True)
if RULE is None:
    sys.exit("precision too low -> NOT applying to test")

# ---- 2) apply to TEST: add pairs for targets not already assigned ----
D = "data/student_resource/dataset/test"
s1 = load(f"{D}/test_source1.tsv")
t = pd.concat([load(f"{D}/test_source2.tsv"), load(f"{D}/test_source3.tsv")])
m = unique_join(s1, t)
m = m[m[RULE]]
cur = pd.read_csv("output/matching_results_v3.tsv", sep="\t", dtype=str).fillna("")
p = cur.assign(t=cur.matched_entity_ids.str.split(",")).explode("t")
p = p[p.t != ""]
new = m[~m.entity_id.isin(set(p.t))]
empty_ids = set(cur.source1_entity_id[cur.matched_entity_ids == ""])
print(f"[test] key pairs={len(m)} new pairs={len(new)} S1 gaining={new.s1.nunique()} "
      f"(currently empty: {new.s1[new.s1.isin(empty_ids)].nunique()}) ({time.time()-t0:.0f}s)", flush=True)

allp = pd.concat([p.rename(columns={"source1_entity_id": "s1"})[["s1", "t"]],
                  new.rename(columns={"entity_id": "t"})[["s1", "t"]]]).drop_duplicates()
lists = allp.sort_values("t").groupby("s1").t.agg(",".join)
out = pd.DataFrame({"source1_entity_id": cur.source1_entity_id})
out["matched_entity_ids"] = out.source1_entity_id.map(lists).fillna("")
out.to_csv("output/matching_results_v4.tsv", sep="\t", index=False)

cand = pd.read_csv("output/candidate_pairs_v3.tsv", sep="\t", dtype=str).fillna("")
add = new.groupby("s1").entity_id.agg(",".join)
extra = cand.source1_entity_id.map(add).fillna("")
rows = (extra != "").to_numpy()
def merge_ids(a, b):
    s = a.split(",") if a else []
    seen = set(s)
    for x in b.split(","):
        if x and x not in seen:
            s.append(x); seen.add(x)
    return ",".join(s)
cand.loc[rows, "candidate_entity_ids"] = [merge_ids(a, b) for a, b in zip(cand.candidate_entity_ids[rows], extra[rows])]
cand.to_csv("output/candidate_pairs_v4.tsv", sep="\t", index=False)
print(f"[test] wrote output/matching_results_v3.tsv + output/candidate_pairs_v3.tsv ({time.time()-t0:.0f}s)", flush=True)
