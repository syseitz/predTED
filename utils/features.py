from typing import List, Tuple, Dict
import numpy as np
import RNA
from collections import Counter, deque

def count_bulges(structure: str) -> int:
    stack = []
    bulges = 0
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')' and stack:
            stack.pop()
        elif char == '.' and stack:
            if i + 1 < len(structure) and structure[i + 1] == ')':
                bulges += 1
    return bulges

def count_internal_loops(structure: str) -> int:
    internal_loops = 0
    i = 0
    while i < len(structure) - 1:
        if structure[i] == ')' and structure[i + 1] == '.':
            j = i + 1
            while j < len(structure) and structure[j] == '.':
                j += 1
            if j < len(structure) and structure[j] == '(':
                internal_loops += 1
            i = j
        else:
            i += 1
    return internal_loops

def max_loop_size(structure: str) -> int:
    max_size = 0
    current = 0
    for c in structure:
        if c == '.':
            current += 1
        else:
            max_size = max(max_size, current)
            current = 0
    return max(max_size, current)

def count_multiloops(structure: str) -> int:
    stack = []
    multiloops = 0
    for i, c in enumerate(structure):
        if c == '(':
            stack.append(i)
        elif c == ')' and stack:
            stack.pop()
            if stack and i + 1 < len(structure) and structure[i + 1] == '(':
                multiloops += 1
    return multiloops

def graph_centrality(structure: str) -> float:
    pairs = structure.count('(')
    return pairs / len(structure) if structure else 0.0

def tree_depth(structure: str) -> int:
    depth = 0
    max_d = 0
    for c in structure:
        if c == '(':
            depth += 1
            max_d = max(max_d, depth)
        elif c == ')':
            depth -= 1
    return max_d

def compute_depth_profile(structure: str) -> List[int]:
    depth = 0
    profile = []
    for c in structure:
        if c == '(':
            depth += 1
        elif c == ')':
            depth -= 1
        profile.append(depth)
    return profile

def get_depth_features(structure: str) -> Tuple[float, float, int, float, float, float, float]:
    profile = compute_depth_profile(structure)
    paired = [profile[i] for i, c in enumerate(structure) if c in '()']
    unpaired = [profile[i] for i, c in enumerate(structure) if c == '.']
    mean_d = np.mean(profile)
    var_d = np.var(profile)
    mean_pd = np.mean(paired) if paired else 0
    var_pd = np.var(paired) if paired else 0
    mean_ud = np.mean(unpaired) if unpaired else 0
    var_ud = np.var(unpaired) if unpaired else 0
    peaks = sum(
        1 for i in range(1, len(profile)-1)
        if profile[i] > profile[i-1] and profile[i] > profile[i+1]
    )
    return mean_d, var_d, peaks, mean_pd, var_pd, mean_ud, var_ud

def find_stems(structure: str) -> List[int]:
    pt = RNA.ptable(structure)
    stems = []
    visited = set()
    i = 1
    while i < len(pt)-1:
        if pt[i] > i and i not in visited and pt[i] not in visited:
            j = pt[i]; length = 1
            while i+1 < len(pt) and pt[i+1] == j-1:
                i += 1; j -= 1; length += 1
                visited.add(i); visited.add(j)
            if length > 1:
                stems.append(length)
        i += 1
    return stems

def get_stem_features(structure: str) -> Tuple[int, float, float, float]:
    stems = find_stems(structure)
    return (
        len(stems),
        max(stems) if stems else 0,
        np.mean(stems) if stems else 0,
        np.var(stems) if stems else 0
    )

def get_loop_features(structure: str) -> Tuple[float, float]:
    sizes = []
    curr = 0
    for c in structure:
        if c == '.':
            curr += 1
        else:
            if curr > 0:
                sizes.append(curr)
                curr = 0
    if curr > 0:
        sizes.append(curr)
    return (
        np.mean(sizes) if sizes else 0,
        np.var(sizes) if sizes else 0
    )

def get_ngram_features(structure: str, n: int=2) -> List[float]:
    ngrams = [structure[i:i+n] for i in range(len(structure)-n+1)]
    cnt = Counter(ngrams)
    total = sum(cnt.values())
    possible = [a+b for a in '().' for b in '().']
    return [cnt.get(ng, 0)/total for ng in possible]

def count_hairpin_loops(structure: str) -> int:
    pt = RNA.ptable(structure)
    count = 0
    for i in range(1, len(pt)):
        if pt[i] > i and all(pt[k]==0 for k in range(i+1, pt[i])):
            count += 1
    return count

def count_stacked_pairs(structure: str) -> int:
    pt = RNA.ptable(structure)
    return sum(
        1 for i in range(1, len(pt)-1)
        if pt[i]>i and pt[i+1]==pt[i]-1
    )

def get_base_pair_distances(structure: str) -> Tuple[float,int]:
    pt = RNA.ptable(structure)
    dists = [pt[i]-i for i in range(1,len(pt)) if pt[i]>i]
    return (np.mean(dists), max(dists)) if dists else (0,0)

def num_paired_bases(structure: str) -> int:
    return 2*structure.count('(')

def num_unpaired_bases(structure: str) -> int:
    return structure.count('.')

def get_hairpin_loop_sizes(structure: str) -> List[int]:
    pt = RNA.ptable(structure)
    sizes = []
    for i in range(1, len(pt)):
        if pt[i]>i and all(pt[k]==0 for k in range(i+1, pt[i])):
            sizes.append(pt[i]-i-1)
    return sizes

def get_internal_loop_sizes(structure: str) -> List[int]:
    pt = RNA.ptable(structure)
    pairs = [(i,pt[i]) for i in range(1,len(pt)) if pt[i]>i]
    sizes = []
    for idx,(i,j) in enumerate(pairs):
        for k,l in pairs[idx+1:]:
            if i<k<l<j and all(pt[m]==0 for m in range(i+1,k)) and all(pt[m]==0 for m in range(l+1,j)):
                sizes.append((k-i-1)+(j-l-1))
    return sizes

def get_bulge_sizes(structure: str) -> List[int]:
    pt = RNA.ptable(structure)
    pairs = [(i,pt[i]) for i in range(1,len(pt)) if pt[i]>i]
    sizes = []
    for idx,(i,j) in enumerate(pairs):
        for k,l in pairs[idx+1:]:
            if k==i+1 and l<j and all(pt[m]==0 for m in range(i+1,k)):
                sizes.append(k-i-1)
            if i==k+1 and j<l and all(pt[m]==0 for m in range(k+1,i)):
                sizes.append(i-k-1)
    return sizes

def depth_correlation(s1: str, s2: str) -> float:
    """
    Pearson correlation of the depth‐profiles of two structures.
    """
    p1 = compute_depth_profile(s1)
    p2 = compute_depth_profile(s2)
    L = min(len(p1), len(p2))
    if L < 2:
        return 0.0

    # Convert to floats so we can subtract the mean
    v1 = np.array(p1[:L], dtype=np.float64)
    v2 = np.array(p2[:L], dtype=np.float64)

    v1 -= v1.mean()
    v2 -= v2.mean()
    denom = np.sqrt((v1 * v1).sum() * (v2 * v2).sum())
    return float((v1 * v2).sum() / denom) if denom > 0 else 0.0


def compute_wiener_index(structure: str) -> int:
    L = len(structure)
    pt = RNA.ptable(structure)
    adj: Dict[int,List[int]] = {i:[] for i in range(L)}
    for i in range(L-1):
        adj[i].append(i+1); adj[i+1].append(i)
    for i in range(1,L+1):
        j=pt[i]
        if j>i:
            adj[i-1].append(j-1); adj[j-1].append(i-1)
    w = 0
    for start in range(L):
        dist=[-1]*L; dist[start]=0
        dq=deque([start])
        while dq:
            u=dq.popleft()
            for v in adj[u]:
                if dist[v]<0:
                    dist[v]=dist[u]+1; dq.append(v)
        w+=sum(dist)
    return w//2

def wiener_index_difference(s1:str,s2:str)->int:
    return abs(compute_wiener_index(s1)-compute_wiener_index(s2))

def get_loop_sizes(structure:str)->List[int]:
    sizes=[]; curr=0
    for c in structure:
        if c=='.': curr+=1
        else:
            if curr>0: sizes.append(curr); curr=0
    if curr>0: sizes.append(curr)
    return sizes

def loop_size_quantiles(structure:str, qs:List[float]=[0.1,0.9])->List[float]:
    sizes=get_loop_sizes(structure)
    return list(np.quantile(sizes,qs)) if sizes else [0.0]*len(qs)

def stem_size_quantiles(structure:str, qs:List[float]=[0.1,0.9])->List[float]:
    stems=find_stems(structure)
    return list(np.quantile(stems,qs)) if stems else [0.0]*len(qs)

def max_node_degree(structure:str)->int:
    L=len(structure); pt=RNA.ptable(structure)
    adj={i:set() for i in range(L)}
    for i in range(L-1):
        adj[i].add(i+1); adj[i+1].add(i)
    for i in range(1,L+1):
        j=pt[i]
        if j>i:
            adj[i-1].add(j-1); adj[j-1].add(i-1)
    return max(len(v) for v in adj.values())

def subtree_pattern_counts(structure:str, patterns:List[str])->Dict[str,int]:
    return {p: structure.count(p) for p in patterns}
