import pickle, json, pandas as pd, numpy as np

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

target_candidates = []
for iid in test_ids:
    oc = merged_dict[iid]['caption']
    opc = earlystop_dict.get(iid,'')
    if not oc or not opc: continue
    ol, opl = len(oc), len(opc)
    target_candidates.append({
        'item_id': iid, 'orig_len': ol, 'opt_len': opl,
        'label': merged_dict[iid]['label'],
    })

df_target = pd.DataFrame(target_candidates)

# Search width 30-60, find best
results = []
for lo in range(5, 180, 1):
    for width in range(30, 65, 2):
        hi = lo + width
        mask = (df_target['orig_len'] >= lo) & (df_target['orig_len'] <= hi) & \
               (df_target['opt_len'] >= lo) & (df_target['opt_len'] <= hi)
        df_cand = df_target[mask].copy()
        if len(df_cand) < 40:
            continue
        
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
                        'n_targets': right-left+1, 'n_bgs': bg_ok,
                        'pop_lo': tl_min, 'pop_hi': tl_max,
                    })
                break

df_results = pd.DataFrame(results)
if len(df_results) > 0:
    df_results = df_results.sort_values('width')
    print(f'Found {len(df_results)} valid configurations')
    print('\nTop 20 (narrowest width):')
    print(df_results.head(20).to_string())
else:
    print('No valid config found!')
