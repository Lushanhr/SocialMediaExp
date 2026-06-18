import pickle, json, pandas as pd, numpy as np, os, shutil, re

# ========== 1. Load data ==========
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

# ========== 2. Filter out meaningless captions ==========
def is_meaningful_caption(text):
    """Check if a caption has meaningful content, not just numbers/IDs/short junk"""
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    # Too short
    if len(text) < 10:
        return False
    # Remove URLs, hashtags, mentions
    cleaned = re.sub(r'https?://\S+', '', text)
    cleaned = re.sub(r'#\S+', '', cleaned)
    cleaned = re.sub(r'@\S+', '', cleaned)
    cleaned = cleaned.strip()
    # After removing URLs/tags, too short
    if len(cleaned) < 8:
        return False
    # Check if it has at least some real words (letters)
    words = re.findall(r'[a-zA-Z]+', cleaned)
    if len(words) < 3:
        return False
    # Check if it's mostly just numbers/IDs
    alpha_count = sum(c.isalpha() for c in cleaned)
    digit_count = sum(c.isdigit() for c in cleaned)
    if alpha_count < 5:
        return False
    if digit_count > alpha_count * 3:
        return False
    return True

# ========== 3. Build target candidates (test ∩ earlystop ∩ meaningful) ==========
target_candidates = []
skipped_meaningless = 0
for iid in test_ids:
    oc = merged_dict[iid]['caption']
    opc = earlystop_dict.get(iid,'')
    if not oc or not opc:
        continue
    # Both orig and optimized must be meaningful
    if not is_meaningful_caption(oc) or not is_meaningful_caption(opc):
        skipped_meaningless += 1
        continue
    ol, opl = len(oc), len(opc)
    target_candidates.append({
        'item_id': iid, 'orig_len': ol, 'opt_len': opl,
        'label': merged_dict[iid]['label'],
        'orig_caption': oc, 'opt_caption': opc,
        'len_diff': abs(ol - opl),
    })

df_target = pd.DataFrame(target_candidates)
print(f'After meaningful filter: {len(df_target)} target candidates (skipped {skipped_meaningless})')
print(f'Len_diff stats:')
print(df_target['len_diff'].describe())

# ========== 4. Build background candidates (all ∩ meaningful) ==========
bg_candidates = []
for iid, info in merged_dict.items():
    cap = info['caption']
    if not is_meaningful_caption(cap):
        continue
    bg_candidates.append({
        'item_id': iid, 'label': info['label'],
        'caption_len': len(cap), 'caption': cap,
    })

df_bg_all = pd.DataFrame(bg_candidates)
print(f'\nBackground candidates (meaningful): {len(df_bg_all)}')

# ========== 5. Select 40 targets: prefer small len_diff, popularity within 30% ==========
# Sort by len_diff, then greedily pick while maintaining popularity constraint
df_target_sorted = df_target.sort_values('len_diff').reset_index(drop=True)

# Greedy selection: pick items with smallest len_diff that fit within 30% popularity window
selected_indices = []
selected_labels = []

for idx, row in df_target_sorted.iterrows():
    label = row['label']
    # Check if adding this item keeps fluctuation <= 30%
    candidate_labels = selected_labels + [label]
    lo, hi = min(candidate_labels), max(candidate_labels)
    fluct = (hi - lo) / max(lo, 1e-9)
    if fluct <= 0.30:
        selected_indices.append(idx)
        selected_labels.append(label)
    if len(selected_indices) >= 40:
        break

# If we couldn't get 40 with greedy, try sliding window approach on top candidates
if len(selected_indices) < 40:
    print(f'\nGreedy only got {len(selected_indices)}, trying sliding window...')
    # Take top candidates by len_diff (e.g. top 200)
    df_top = df_target_sorted.head(200).sort_values('label').reset_index(drop=True)
    labels = df_top['label'].values
    
    best_left, best_right = 0, 0
    left = 0
    for right in range(len(labels)):
        while left < right and (labels[right] - labels[left]) / max(labels[left], 1e-9) > 0.30:
            left += 1
        if right - left + 1 >= 40 and (right - left) > (best_right - best_left):
            best_left, best_right = left, right
    
    if best_right - best_left + 1 >= 40:
        df_selected_targets = df_top.iloc[best_left:best_right+1].head(40).copy()
        df_selected_targets['len_diff'] = abs(df_selected_targets['orig_len'] - df_selected_targets['opt_len'])
        print(f'Sliding window: got {len(df_selected_targets)} targets')
    else:
        print('ERROR: Cannot find 40 targets!')
        exit(1)
else:
    df_selected_targets = df_target_sorted.loc[selected_indices].copy()
    df_selected_targets['len_diff'] = abs(df_selected_targets['orig_len'] - df_selected_targets['opt_len'])

print(f'\n=== 40 TARGET POSTS ===')
print(f'Popularity: {df_selected_targets["label"].min():.4f} ~ {df_selected_targets["label"].max():.4f}')
fluct_t = (df_selected_targets["label"].max() - df_selected_targets["label"].min()) / df_selected_targets["label"].min()
print(f'Popularity fluctuation: {fluct_t*100:.1f}%')
print(f'Orig len: {df_selected_targets["orig_len"].min()} ~ {df_selected_targets["orig_len"].max()}, mean={df_selected_targets["orig_len"].mean():.1f}')
print(f'Opt len: {df_selected_targets["opt_len"].min()} ~ {df_selected_targets["opt_len"].max()}, mean={df_selected_targets["opt_len"].mean():.1f}')
print(f'Len diff: {df_selected_targets["len_diff"].min()} ~ {df_selected_targets["len_diff"].max()}, mean={df_selected_targets["len_diff"].mean():.1f}')

# ========== 6. Select 36 background posts ==========
target_ids = set(df_selected_targets['item_id'].values)
tl_min = df_selected_targets['label'].min()
tl_max = df_selected_targets['label'].max()
target_avg_len = df_selected_targets[['orig_len','opt_len']].values.flatten().mean()

# Filter backgrounds: popularity consistent with targets
bg_filtered = []
for _, row in df_bg_all.iterrows():
    if row['item_id'] in target_ids:
        continue
    label = row['label']
    alo = min(tl_min, label)
    ahi = max(tl_max, label)
    if (ahi - alo) / max(alo, 1e-9) > 0.30:
        continue
    bg_filtered.append({
        'item_id': row['item_id'], 'label': row['label'],
        'caption_len': row['caption_len'], 'caption': row['caption'],
        'len_diff_from_avg': abs(row['caption_len'] - target_avg_len),
    })

df_bg = pd.DataFrame(bg_filtered)
print(f'\nBackground candidates (popularity ok): {len(df_bg)}')

# Select 36 closest to target average length
df_selected_bg = df_bg.nsmallest(36, 'len_diff_from_avg').reset_index(drop=True)

# ========== 7. Final summary ==========
all_labels = list(df_selected_targets['label'].values) + list(df_selected_bg['label'].values)
all_lo, all_hi = min(all_labels), max(all_labels)
total_fluct = (all_hi - all_lo) / all_lo

all_lens = (list(df_selected_targets['orig_len'].values) + 
            list(df_selected_targets['opt_len'].values) + 
            list(df_selected_bg['caption_len'].values))

print(f'\n=== 36 BACKGROUND POSTS ===')
print(f'Popularity: {df_selected_bg["label"].min():.4f} ~ {df_selected_bg["label"].max():.4f}')
print(f'Caption len: {df_selected_bg["caption_len"].min()} ~ {df_selected_bg["caption_len"].max()}, mean={df_selected_bg["caption_len"].mean():.1f}')

print(f'\n===== FINAL SUMMARY (76 posts) =====')
print(f'Popularity: {all_lo:.4f} ~ {all_hi:.4f}, fluctuation={total_fluct*100:.1f}%')
print(f'All text lengths: min={min(all_lens)}, max={max(all_lens)}, mean={np.mean(all_lens):.1f}, std={np.std(all_lens):.1f}')
print(f'Target orig len: {df_selected_targets["orig_len"].min()}~{df_selected_targets["orig_len"].max()} mean={df_selected_targets["orig_len"].mean():.1f}')
print(f'Target opt len: {df_selected_targets["opt_len"].min()}~{df_selected_targets["opt_len"].max()} mean={df_selected_targets["opt_len"].mean():.1f}')
print(f'BG caption len: {df_selected_bg["caption_len"].min()}~{df_selected_bg["caption_len"].max()} mean={df_selected_bg["caption_len"].mean():.1f}')

# ========== 8. Save ==========
df_selected_targets[['item_id','label','orig_len','opt_len','len_diff','orig_caption','opt_caption']].to_csv(
    '/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/selected_targets.csv', index=False)
df_selected_bg[['item_id','label','caption_len','caption']].to_csv(
    '/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/selected_backgrounds.csv', index=False)
print('\nSaved: selected_targets.csv, selected_backgrounds.csv')

# ========== 9. Copy images ==========
img_src_dir = '/data/Lushanhr/popularity/data/ICIP/train_imgs'
img_dst_dir = '/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/test_images_811'
os.makedirs(img_dst_dir, exist_ok=True)
for f in os.listdir(img_dst_dir):
    os.remove(os.path.join(img_dst_dir, f))

count = 0
all_ids = list(df_selected_targets['item_id'].values) + list(df_selected_bg['item_id'].values)
for iid in all_ids:
    for ext in ['.jpg','.jpeg','.png']:
        src = os.path.join(img_src_dir, iid + ext)
        if os.path.exists(src):
            shutil.copy2(src, img_dst_dir)
            count += 1
            break

print(f'Copied {count} images to test_images_811/')

# Print target details
print('\n=== TARGET DETAILS ===')
for i, row in df_selected_targets.iterrows():
    print(f'  [{i}] id={row["item_id"]} pop={row["label"]:.4f} orig={row["orig_len"]} opt={row["opt_len"]} diff={row["len_diff"]} orig="{row["orig_caption"][:60]}" opt="{row["opt_caption"][:60]}"')
