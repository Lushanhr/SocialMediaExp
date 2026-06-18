import pickle, json, pandas as pd, numpy as np, os, shutil

# 1. Load data
with open('/data/Lushanhr/popularity/CopyGRPO/data/ICIP/merged.pkl','rb') as f:
    merged = pickle.load(f)

with open('/data/Lushanhr/popularity/CopyGRPO/data/ICIP/split_811_seed2026.json') as f:
    split = json.load(f)

earlystop_df = pd.read_csv('/data/Lushanhr/popularity/CopyGRPO/output/eval_earlystop_newcban_test_fixed.csv')

# 2. Build lookup: item_id -> {label, caption}
merged_dict = {}
for item in merged:
    iid = str(item['item_id'])
    cap = item.get('cban_text','') or item.get('skapp_text','') or item.get('image_caption','')
    merged_dict[iid] = {'label': item['label'], 'caption': cap}

# 3. Get candidate sets
test_ids = set(str(x) for x in split['test'])
earlystop_ids = set(str(x) for x in earlystop_df['image_id'].values)
target_candidates = test_ids & earlystop_ids

print(f'Test set: {len(test_ids)}')
print(f'Earlystop results: {len(earlystop_ids)}')
print(f'Target candidates (test ∩ earlystop): {len(target_candidates)}')

# 4. Build earlystop lookup
earlystop_dict = dict(zip(earlystop_df['image_id'].astype(str), earlystop_df['rewritten_caption']))

# 5. Filter target candidates: must have both original and optimized caption
target_info = []
for iid in target_candidates:
    oc = merged_dict[iid]['caption']
    opc = earlystop_dict.get(iid,'')
    if not oc or not opc:
        continue
    ol, opl = len(oc), len(opc)
    target_info.append({
        'item_id': iid,
        'orig_len': ol,
        'opt_len': opl,
        'len_ratio': max(ol, opl) / max(min(ol, opl), 1),
        'label': merged_dict[iid]['label'],
        'orig_caption': oc,
        'opt_caption': opc
    })

df_target = pd.DataFrame(target_info)
print(f'\nTarget candidates with both captions: {len(df_target)}')
print(f'\n=== Label (popularity) stats ===')
print(df_target['label'].describe())
print(f'\n=== Original caption length stats ===')
print(df_target['orig_len'].describe())
print(f'\n=== Optimized caption length stats ===')
print(df_target['opt_len'].describe())
print(f'\n=== Length ratio (max/min) stats ===')
print(df_target['len_ratio'].describe())

# 6. Strategy: select 40 target posts + 36 background posts
# Constraints:
#   - All posts (target+background) have similar popularity, fluctuation <= 30%
#   - Target posts: orig_len ≈ opt_len (len_ratio not too large)
#   - Background posts: length near target posts' length L

# Step 1: Filter target candidates by len_ratio (orig vs opt not too different)
# Use len_ratio <= 2.0 as initial filter
df_target_filtered = df_target[df_target['len_ratio'] <= 2.0].copy()
print(f'\nAfter len_ratio <= 2.0 filter: {len(df_target_filtered)} target candidates')

# Step 2: Find a popularity window that can contain 40+ targets and 36+ backgrounds
# Sort by label and try to find a window where fluctuation <= 30%
df_target_filtered = df_target_filtered.sort_values('label').reset_index(drop=True)

# Try different windows: pick a center and expand with 30% constraint
def find_best_window(df, n_needed=40, max_fluctuation=0.30):
    """Find the largest window with fluctuation <= max_fluctuation that contains >= n_needed items"""
    best = None
    labels = df['label'].values
    n = len(labels)
    
    # Sliding window approach
    left = 0
    for right in range(n):
        # Ensure window fluctuation <= 30%
        while left < right and (labels[right] - labels[left]) / max(labels[left], 1e-9) > max_fluctuation:
            left += 1
        window_size = right - left + 1
        if window_size >= n_needed:
            fluct = (labels[right] - labels[left]) / max(labels[left], 1e-9)
            if best is None or window_size > best[2]:
                best = (left, right, window_size, fluct, labels[left], labels[right])
    
    return best

result = find_best_window(df_target_filtered, n_needed=40, max_fluctuation=0.30)
if result:
    left, right, size, fluct, lo, hi = result
    print(f'\nBest window for targets: [{lo:.4f}, {hi:.4f}], fluctuation={fluct:.4f}, size={size}')
    
    df_in_window = df_target_filtered.iloc[left:right+1].copy()
    print(f'Items in window: {len(df_in_window)}')
    
    # Select 40 targets from this window
    if len(df_in_window) >= 40:
        # Pick 40 with smallest len_ratio (most balanced orig/opt lengths)
        df_selected_targets = df_in_window.nsmallest(40, 'len_ratio')
        print(f'\n=== Selected 40 target posts ===')
        print(f'Label range: {df_selected_targets["label"].min():.4f} ~ {df_selected_targets["label"].max():.4f}')
        fluct_targets = (df_selected_targets["label"].max() - df_selected_targets["label"].min()) / max(df_selected_targets["label"].min(), 1e-9)
        print(f'Fluctuation: {fluct_targets:.4f} ({fluct_targets*100:.1f}%)')
        print(f'Orig len range: {df_selected_targets["orig_len"].min()} ~ {df_selected_targets["orig_len"].max()}')
        print(f'Opt len range: {df_selected_targets["opt_len"].min()} ~ {df_selected_targets["opt_len"].max()}')
        print(f'Avg orig len: {df_selected_targets["orig_len"].mean():.1f}')
        
        target_label_min = df_selected_targets['label'].min()
        target_label_max = df_selected_targets['label'].max()
        target_avg_len = df_selected_targets['orig_len'].mean()
        
        # Step 3: Select 36 background posts from full dataset
        # Constraints: popularity similar to targets (fluctuation <= 30%), length near target avg length
        # Background cannot be in target set
        target_item_ids = set(df_selected_targets['item_id'].values)
        
        bg_candidates = []
        for iid, info in merged_dict.items():
            if iid in target_item_ids:
                continue
            cap = info['caption']
            if not cap:
                continue
            label = info['label']
            cap_len = len(cap)
            # Check popularity within 30% of target range
            label_lo = min(target_label_min, label)
            label_hi = max(target_label_max, label)
            fluct = (label_hi - label_lo) / max(label_lo, 1e-9)
            if fluct > 0.30:
                continue
            # Check length near target avg (within 50% for now)
            bg_candidates.append({
                'item_id': iid,
                'label': label,
                'caption_len': cap_len,
                'caption': cap
            })
        
        df_bg = pd.DataFrame(bg_candidates)
        print(f'\nBackground candidates (popularity ok): {len(df_bg)}')
        
        if len(df_bg) >= 36:
            # Select 36 with length closest to target average length
            df_bg['len_diff'] = abs(df_bg['caption_len'] - target_avg_len)
            df_selected_bg = df_bg.nsmallest(36, 'len_diff')
            
            # Final check: all posts popularity fluctuation
            all_labels = list(df_selected_targets['label'].values) + list(df_selected_bg['label'].values)
            all_lo, all_hi = min(all_labels), max(all_labels)
            total_fluct = (all_hi - all_lo) / max(all_lo, 1e-9)
            
            print(f'\n=== Selected 36 background posts ===')
            print(f'Label range: {df_selected_bg["label"].min():.4f} ~ {df_selected_bg["label"].max():.4f}')
            print(f'Caption len range: {df_selected_bg["caption_len"].min()} ~ {df_selected_bg["caption_len"].max()}')
            print(f'Avg caption len: {df_selected_bg["caption_len"].mean():.1f}')
            
            print(f'\n=== FINAL: All 76 posts ===')
            print(f'Total popularity fluctuation: {total_fluct:.4f} ({total_fluct*100:.1f}%)')
            print(f'Popularity range: {all_lo:.4f} ~ {all_hi:.4f}')
            
            # Save results
            df_selected_targets[['item_id','label','orig_len','opt_len','len_ratio','orig_caption','opt_caption']].to_csv(
                '/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/selected_targets.csv', index=False)
            df_selected_bg[['item_id','label','caption_len','caption']].to_csv(
                '/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/selected_backgrounds.csv', index=False)
            print('\nSaved: selected_targets.csv, selected_backgrounds.csv')
            
            # Copy images to test_images_811
            img_src_dir = '/data/Lushanhr/popularity/data/ICIP/train_imgs'
            img_dst_dir = '/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/test_images_811'
            os.makedirs(img_dst_dir, exist_ok=True)
            
            count = 0
            all_ids = list(df_selected_targets['item_id'].values) + list(df_selected_bg['item_id'].values)
            for iid in all_ids:
                # Try different extensions
                for ext in ['.jpg','.jpeg','.png']:
                    src = os.path.join(img_src_dir, iid + ext)
                    if os.path.exists(src):
                        shutil.copy2(src, img_dst_dir)
                        count += 1
                        break
            print(f'\nCopied {count} images to test_images_811/')
    else:
        print(f'Not enough targets in window: {len(df_in_window)} < 40')
else:
    print('Could not find a suitable window!')
