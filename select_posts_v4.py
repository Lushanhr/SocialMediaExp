import pickle, json, pandas as pd, numpy as np, os, shutil

# Load data
with open('/data/Lushanhr/popularity/CopyGRPO/data/ICIP/merged.pkl','rb') as f:
    merged = pickle.load(f)
with open('/data/Lushanhr/popularity/CopyGRPO/data/ICIP/split_811_seed2026.json') as f:
    split = json.load(f)
earlystop_df = pd.read_csv('/data/Lushanhr/popularity/CopyGRPO/output/eval_earlystop_newcban_test_fixed.csv')

merged_dict = {}
for item in merged:
    iid = str(item['item_id'])
    cap = item.get('cban_text','') or item.get('skapp_text','') or item.get('image_caption','')
    merged_dict[iid] = {'label': item['label'], 'caption': cap}

test_ids = set(str(x) for x in split['test'])
earlystop_dict = dict(zip(earlystop_df['image_id'].astype(str), earlystop_df['rewritten_caption']))

# Build target candidates
target_candidates = []
for iid in test_ids:
    oc = merged_dict[iid]['caption']
    opc = earlystop_dict.get(iid,'')
    if not oc or not opc: continue
    ol, opl = len(oc), len(opc)
    target_candidates.append({
        'item_id': iid, 'orig_len': ol, 'opt_len': opl,
        'label': merged_dict[iid]['label'],
        'orig_caption': oc, 'opt_caption': opc,
        'len_diff': abs(ol - opl),
    })

df_target = pd.DataFrame(target_candidates)

# Strategy: fix a length window [lo, hi], require:
#   - target: orig_len in [lo,hi] AND opt_len in [lo,hi] AND popularity within 30%
#   - bg: caption_len in [lo,hi] AND popularity consistent with targets
# Try VERY narrow windows (width 10-40) with finer steps

results = []
for lo in range(10, 150, 2):
    for width in [10, 15, 20, 25, 30, 35, 40]:
        hi = lo + width
        
        mask = (df_target['orig_len'] >= lo) & (df_target['orig_len'] <= hi) & \
               (df_target['opt_len'] >= lo) & (df_target['opt_len'] <= hi)
        df_cand = df_target[mask]
        
        if len(df_cand) < 40:
            continue
        
        # Check popularity window
        df_cand = df_cand.sort_values('label').reset_index(drop=True)
        labels = df_cand['label'].values
        
        left = 0
        for right in range(len(labels)):
            while left < right and (labels[right] - labels[left]) / max(labels[left], 1e-9) > 0.30:
                left += 1
            if right - left + 1 >= 40:
                df_sel = df_cand.iloc[left:right+1].head(40)
                tl_min = df_sel['label'].min()
                tl_max = df_sel['label'].max()
                tids = set(df_sel['item_id'].values)
                
                # Count backgrounds
                bg_ok = 0
                for iid, info in merged_dict.items():
                    if iid in tids: continue
                    cap = info['caption']
                    if not cap: continue
                    cl = len(cap)
                    if cl < lo or cl > hi: continue
                    label = info['label']
                    alo = min(tl_min, label)
                    ahi = max(tl_max, label)
                    if (ahi - alo) / max(alo, 1e-9) > 0.30: continue
                    bg_ok += 1
                
                if bg_ok >= 36:
                    results.append({
                        'lo': lo, 'hi': hi, 'width': width,
                        'targets': len(df_cand), 'bgs': bg_ok,
                        'pop_lo': tl_min, 'pop_hi': tl_max,
                    })
                break  # only need first valid window for this lo/hi

df_results = pd.DataFrame(results)
if len(df_results) > 0:
    df_results = df_results.sort_values('width')
    print(f'Found {len(df_results)} valid configurations')
    print('\nTop 20 (narrowest width):')
    print(df_results.head(20).to_string())
else:
    print('No valid narrow configuration found!')
    # Try relaxed: len_diff <= 30 and see what width is achievable
    print('\nTrying relaxed approach: len_diff <= 30, then find tight length cluster')
    df_relax = df_target[df_target['len_diff'] <= 30].copy()
    print(f'Candidates with len_diff<=30: {len(df_relax)}')
    # Use the "representative length" = (orig_len + opt_len)/2
    df_relax['rep_len'] = (df_relax['orig_len'] + df_relax['opt_len']) / 2
    # For each candidate, the length span is [min(orig,opt), max(orig,opt)]
    # We want to find a window that covers ALL of [min, max] for 40 items
    for w in [20, 25, 30, 35, 40, 50, 60]:
        count = 0
        for center in range(20, 200, 2):
            lo2 = center - w//2
            hi2 = center + w//2
            mask2 = (df_relax['orig_len'] >= lo2) & (df_relax['orig_len'] <= hi2) & \
                    (df_relax['opt_len'] >= lo2) & (df_relax['opt_len'] <= hi2)
            c = mask2.sum()
            if c > count:
                count = c
        print(f'  Width {w}: max targets = {count}')
