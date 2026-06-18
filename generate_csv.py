import pickle, json, pandas as pd, numpy as np, os, shutil, re, random
from itertools import product

random.seed(42)
np.random.seed(42)

# ========== 1. Load selected data ==========
df_targets = pd.read_csv('/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/selected_targets.csv')
df_bgs = pd.read_csv('/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/selected_backgrounds.csv')

# Only need 16 backgrounds (from 36), pick 16 closest to target avg length
target_avg_len = df_targets[['orig_len','opt_len']].values.flatten().mean()
df_bgs['len_diff_from_avg'] = abs(df_bgs['caption_len'] - target_avg_len)
df_bgs_16 = df_bgs.nsmallest(16, 'len_diff_from_avg').reset_index(drop=True)

print(f'Targets: {len(df_targets)}, Backgrounds (selected 16): {len(df_bgs_16)}')
print(f'Target avg len: {target_avg_len:.1f}')
print(f'BG caption len: {df_bgs_16["caption_len"].min()}~{df_bgs_16["caption_len"].max()} mean={df_bgs_16["caption_len"].mean():.1f}')

# ========== 2. Update test_images_811: only keep needed images ==========
all_ids = list(df_targets['item_id'].values) + list(df_bgs_16['item_id'].values)

img_src_dir = '/data/Lushanhr/popularity/data/ICIP/train_imgs'
img_dst_dir = '/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/test_images_811'
os.makedirs(img_dst_dir, exist_ok=True)
for f in os.listdir(img_dst_dir):
    os.remove(os.path.join(img_dst_dir, f))

count = 0
for iid in all_ids:
    iid = str(iid)
    for ext in ['.jpg','.jpeg','.png']:
        src = os.path.join(img_src_dir, iid + ext)
        if os.path.exists(src):
            shutil.copy2(src, img_dst_dir)
            count += 1
            break
print(f'Copied {count} images to test_images_811/')

# ========== 3. Generate social_feed_test.csv ==========
# 40 targets -> 10 groups of 4
# Each group: 2^4 = 16 combinations
# Total: 10 * 16 = 160 conditions (feeds)
# Each feed: 4 targets + 16 backgrounds = 20 posts

# Split 40 targets into 10 groups of 4
target_groups = []
for i in range(0, 40, 4):
    target_groups.append(df_targets.iloc[i:i+4].reset_index(drop=True))

# Generate usernames/handles
usernames = [
    'TravelVibes', 'CityLens', 'NatureSnap', 'FoodieGram', 'StyleSpot',
    'DailyCapture', 'LightChaser', 'MomentMaker', 'PixelWander', 'VibeCheck',
    'UrbanLens', 'SoulFrame', 'SnapStories', 'WildHeart', 'SunChaser',
    'DreamScroll', 'LifeInFocus', 'ClickCraze', 'GlowPost', 'FrameTales',
    'StoryStream', 'CatchyView', 'PostPerfect', 'VisualVibe', 'InstaMood',
    'ScrollMagic', 'PhotoFlow', 'TrendSnap', 'CandidShot', 'FeedFever',
    'MoodCapture', 'FlashFrame', 'LensLegend', 'SnapSage', 'ViewVortex',
    'PicPulse', 'InstaVista', 'FocusFreak', 'GramGuru', 'ShotSmith',
    'BuzzFrame', 'ChillPost', 'EpicSnap', 'FlowState', 'GazePoint',
    'HighlightHub', 'ImageImpulse', 'JoyScroll', 'KineticCapture', 'LuminaPost',
    'MicroMoment', 'NovaFrame', 'OpticOasis', 'PeakPixel', 'QuickCapture',
    'RapidReel', 'SilkSnap', 'TrendTracker', 'UpliftView', 'VisionVault',
]

handles = [f'@{u.lower()}' for u in usernames]

# Media URL template
media_base = 'https://media.githubusercontent.com/media/Lushanhr/SocialMediaExp/refs/heads/main/test_images_811'

# Pre-generate deterministic metadata per item_id (consistent across conditions)
# This ensures same item_id always gets same likes/reposts/replies/datetime/user_followers
def generate_item_metadata(iid_str):
    rng = random.Random(hash(iid_str))
    return {
        'likes': rng.randint(15, 85),
        'reposts': rng.randint(0, 7),
        'replies': rng.randint(0, 4),
        'datetime': f'{rng.randint(1,28):02d}.{rng.randint(1,12):02d}.26 {rng.randint(0,23):02d}:{rng.choice([0,15,30,45]):02d}',
        'user_followers': rng.randint(100, 15000),
    }

# Build metadata lookup for all target and background items
item_meta = {}
for group_idx, group in enumerate(target_groups):
    for t_idx in range(4):
        iid = str(group.iloc[t_idx]['item_id'])
        item_meta[iid] = generate_item_metadata(iid)
        item_meta[iid]['username'] = usernames[(group_idx * 4 + t_idx) % len(usernames)]
        item_meta[iid]['handle'] = handles[(group_idx * 4 + t_idx) % len(handles)]
        uname = item_meta[iid]['username']
        item_meta[iid]['user_image'] = f'https://ui-avatars.com/api/?name={uname}&size=150&background=random'

for bg_idx in range(16):
    iid = str(df_bgs_16.iloc[bg_idx]['item_id'])
    item_meta[iid] = generate_item_metadata(iid)
    item_meta[iid]['username'] = usernames[(40 + bg_idx) % len(usernames)]
    item_meta[iid]['handle'] = handles[(40 + bg_idx) % len(handles)]
    uname = item_meta[iid]['username']
    item_meta[iid]['user_image'] = f'https://ui-avatars.com/api/?name={uname}&size=150&background=random'

# Pre-generate sequence: 20 positions, targets NOT in position 1
# Randomly place 4 targets among positions 2-20, backgrounds fill the rest
# The sequence must be consistent: same target always at same position across conditions within a group
# But different groups can have different target positions
def generate_group_sequence(group_idx):
    """Generate a fixed sequence for a group. Targets get positions 2-20, never position 1."""
    rng = random.Random(group_idx * 1000 + 42)
    # Available positions for targets: 2..20 (19 slots)
    target_positions = sorted(rng.sample(range(2, 21), 4))
    # Remaining positions for backgrounds
    bg_positions = [p for p in range(1, 21) if p not in target_positions]
    # Map: t_idx -> sequence number
    target_seq = {t_idx: target_positions[t_idx] for t_idx in range(4)}
    # Map: bg_idx -> sequence number
    bg_seq = {bg_idx: bg_positions[bg_idx] for bg_idx in range(len(bg_positions))}
    return target_seq, bg_seq

# Generate rows
rows = []
condition_idx = 0

for group_idx, group in enumerate(target_groups):
    target_seq, bg_seq = generate_group_sequence(group_idx)
    
    # 4 targets in this group: A, B, C, D
    targets = []
    for t_idx in range(4):
        t = group.iloc[t_idx]
        targets.append({
            'item_id': str(t['item_id']),
            'orig_text': t['orig_caption'],
            'opt_text': t['opt_caption'],
        })
    
    # Generate all 16 combinations (0=orig, 1=opt for each of 4 targets)
    for combo in product([0, 1], repeat=4):
        condition_idx += 1
        condition_name = f'combo_{group_idx+1}_{condition_idx:04d}'
        
        # 4 target posts (doc_id 1-4)
        for t_idx in range(4):
            t = targets[t_idx]
            version = combo[t_idx]  # 0=orig, 1=opt
            text = t['opt_text'] if version == 1 else t['orig_text']
            # Clean nan
            text = text.replace(' nan', '').replace('nan', '').strip()
            
            doc_id = t_idx + 1
            iid = t['item_id']
            meta = item_meta[iid]
            
            row = {
                'doc_id': doc_id,
                'datetime': meta['datetime'],
                'text': text,
                'media': f'{media_base}/{iid}.jpg',
                'alt_text': '',
                'likes': meta['likes'],
                'reposts': meta['reposts'],
                'replies': meta['replies'],
                'username': meta['username'],
                'handle': meta['handle'],
                'user_description': 'Sharing moments that matter',
                'user_image': meta['user_image'],
                'user_followers': meta['user_followers'],
                'commented_post': 0,
                'sponsored': 0,
                'target': '',
                'condition': condition_name,
                'sequence': target_seq[t_idx],
            }
            rows.append(row)
        
        # 16 background posts (doc_id 5-20) - same across all conditions
        for bg_idx in range(16):
            bg = df_bgs_16.iloc[bg_idx]
            iid = str(bg['item_id'])
            text = str(bg['caption']).replace(' nan', '').replace('nan', '').strip()
            
            doc_id = bg_idx + 5
            meta = item_meta[iid]
            
            row = {
                'doc_id': doc_id,
                'datetime': meta['datetime'],
                'text': text,
                'media': f'{media_base}/{iid}.jpg',
                'alt_text': '',
                'likes': meta['likes'],
                'reposts': meta['reposts'],
                'replies': meta['replies'],
                'username': meta['username'],
                'handle': meta['handle'],
                'user_description': 'Sharing moments that matter',
                'user_image': meta['user_image'],
                'user_followers': meta['user_followers'],
                'commented_post': 0,
                'sponsored': 0,
                'target': '',
                'condition': condition_name,
                'sequence': bg_seq[bg_idx],
            }
            rows.append(row)

df_out = pd.DataFrame(rows)
print(f'\nGenerated {len(df_out)} rows, {df_out["condition"].nunique()} conditions')
print(f'Rows per condition: {df_out.groupby("condition").size().unique()}')

# Verify consistency: same item_id should have same likes/username across all conditions
print('\n=== Consistency Check ===')
for col in ['likes', 'reposts', 'replies', 'datetime', 'username', 'handle', 'user_image', 'user_followers']:
    consistent = True
    for group_idx, group in enumerate(target_groups):
        for t_idx in range(4):
            iid = str(group.iloc[t_idx]['item_id'])
            doc_id = t_idx + 1
            sub = df_out[(df_out['doc_id'] == doc_id) & (df_out['condition'].str.startswith(f'combo_{group_idx+1}_'))]
            vals = sub[col].unique()
            if len(vals) > 1:
                print(f'  INCONSISTENT: item {iid}, col {col}, values: {vals}')
                consistent = False
    if consistent:
        print(f'  {col}: consistent across conditions ✓')

# Verify sequence: targets never at position 1
print('\n=== Sequence Check ===')
for group_idx in range(10):
    sub = df_out[df_out['condition'].str.startswith(f'combo_{group_idx+1}_')]
    first_cond = sub['condition'].iloc[0]
    first_feed = sub[sub['condition'] == first_cond].sort_values('sequence')
    target_seqs = first_feed[first_feed['doc_id'] <= 4]['sequence'].values
    bg_seqs = first_feed[first_feed['doc_id'] > 4]['sequence'].values
    print(f'  Group {group_idx+1}: target positions={sorted(target_seqs)}, bg positions={sorted(bg_seqs)}')
    if 1 in target_seqs:
        print(f'    WARNING: target at position 1!')

# Save
out_path = '/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/social_feed_test.csv'
df_out.to_csv(out_path, sep=';', index=False)
print(f'\nSaved to {out_path}')

# Verify
print(f'\nTotal rows: {len(df_out)}')
print(f'Total conditions: {df_out["condition"].nunique()}')
print(f'doc_id range: {df_out["doc_id"].min()} ~ {df_out["doc_id"].max()}')

# Save updated backgrounds
df_bgs_16.to_csv('/data/Lushanhr/popularity/CopyGRPO/SocialMediaExp/selected_backgrounds.csv', index=False)
