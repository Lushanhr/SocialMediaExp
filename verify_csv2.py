import pandas as pd
df = pd.read_csv('social_feed_test.csv', sep=';')

# Check all 16 combos for group 1
print('Group 1 - all 16 combos, target texts (doc_id 1-4):')
g1 = df[df['condition'].str.startswith('combo_1_')]
for cond in sorted(g1['condition'].unique()):
    sub = g1[g1['condition']==cond]
    targets = sub[sub['doc_id']<=4].sort_values('doc_id')
    versions = []
    for _, r in targets.iterrows():
        # Check if it's orig or opt by comparing length
        versions.append(f'd{r["doc_id"]}:{r["text"][:30]}...')
    print(f'  {cond}: {" | ".join(versions)}')
