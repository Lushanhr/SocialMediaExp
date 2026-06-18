import pickle, json, pandas as pd
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
rows = []
for iid in test_ids:
    oc = merged_dict[iid]['caption']
    opc = earlystop_dict.get(iid,'')
    if not oc or not opc: continue
    rows.append({'item_id':iid,'orig_len':len(oc),'opt_len':len(opc),'len_diff':abs(len(oc)-len(opc))})
df = pd.DataFrame(rows)
print('Len diff (orig-opt) stats:')
print(df['len_diff'].describe())
print()
for thresh in [5, 10, 15, 20, 25, 30, 40, 50]:
    n = (df['len_diff'] <= thresh).sum()
    print(f'len_diff <= {thresh}: {n} items')
