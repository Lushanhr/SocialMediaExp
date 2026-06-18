import pickle, json, pandas as pd, numpy as np, os, shutil

# 1. Load data
with open('/data/Lushanhr/popularity/CopyGRPO/data/ICIP/merged.pkl','rb') as f:
    merged = pickle.load(f)

with open('/data/Lushanhr/popularity/CopyGRPO/data/ICIP/split_811_seed2026.json') as f:
    split = json.load(f)

earlystop_df = pd.read_csv('/data/Lushanhr/popularity/CopyGRPO/output/eval_earlystop_newcban_test_fixed.csv')

# 2. Build lookup
merged_dict = {}
for item in merged:
    iid = str(item['item_id'])
    cap = item.get('cban_text','') or item.get('skapp_text','') or item.get('image_caption','')
    merged_dict[iid] = {'label': item['label'], 'caption': cap}

test_ids = set(str(x) for x in split['test'])
earlystop_dict = dict(zip(earlystop_df['image_id'].astype(str), earlystop_df['rewritten_caption']))

# 3. Build target candidates
target_candidates = []
for iid in test_ids:
    oc = merged_dict[iid]['caption']
    opc = earlystop_dict.get(iid,'')
    if not oc or not opc:
        continue
    ol, opl = len(oc), len(opc)
    target_candidates.append({
        'item_id': iid,
        'orig_len': ol,
        'opt_len': opl,
        'label': merged_dict[iid]['label'],
        'orig_caption': oc,
        'opt_caption': opc,
    })

df_target = pd.DataFrame(target_candidates)

# 4. Finer grid search: try narrower windows
results = []
for center in range(20, 201, 2):
    for half_width in [5, 8, 10, 12, 15, 18, 20, 25, 30]:
        lo = center - half_width
        hi = center + half_width
        if lo < 3:
            continue
        
        mask = (df_target['orig_len'] >= lo) & (df_target['orig_len'] <= hi) & \
               (df_target['opt_len'] >= lo) & (df_target['opt_len'] <= hi)
        df_cand = df_target[mask]
        
        if len(df_cand) < 40:
            continue
        
        df_cand = df_cand.sort_values('label').reset_index(drop=True)
        labels = df_cand['label'].values
        
        left = 0
        best_target_count = 0
        best_target_range = None
        for right in range(len(labels)):
            while left < right and (labels[right] - labels[left]) / max(labels[left], 1e-9) > 0.30:
                left += 1
            if right - left + 1 >= 40 and right - left + 1 > best_target_count:
                best_target_count = right - left + 1
                best_target_range = (left, right)
        
        if best_target_count < 40:
            continue
        
        l, r = best_target_range
        df_sel = df_cand.iloc[l:r+1].head(40)
        target_label_min = df_sel['label'].min()
        target_label_max = df_sel['label'].max()
        target_ids = set(df_sel['item_id'].values)
        
        bg_count = 0
        for iid, info in merged_dict.items():
            if iid in target_ids:
                continue
            cap = info['caption']
            if not cap:
                continue
            cl = len(cap)
            if cl < lo or cl > hi:
                continue
            label = info['label']
            all_lo = min(target_label_min, label)
            all_hi = max(target_label_max, label)
            fluct = (all_hi - all_lo) / max(all_lo, 1e-9)
            if fluct > 0.30:
                continue
            bg_count += 1
        
        if bg_count >= 36:
            results.append({
                'center': center,
                'lo': lo,
                'hi': hi,
                'width': hi - lo,
                'target_count': best_target_count,
                'bg_count': bg_count,
                'label_min': target_label_min,
                'label_max': target_label_max,
            })

df_results = pd.DataFrame(results)
if len(df_results) > 0:
    df_results = df_results.sort_values('width')
    print(f'Found {len(df_results)} valid configurations')
    print('\nTop 20 (narrowest width):')
    print(df_results.head(20).to_string())
    
    best = df_results.iloc[0]
    print(f'\n=== Best (narrowest) config ===')
    print(f'Length window: [{best["lo"]}, {best["hi"]}] (width={best["width"]})')
    print(f'Center: {best["center"]}')
    print(f'Target candidates: {best["target_count"]}, Background candidates: {best["bg_count"]}')
    print(f'Popularity: {best["label_min"]:.4f} ~ {best["label_max"]:.4f}')
else:
    print('No valid configuration found!')
