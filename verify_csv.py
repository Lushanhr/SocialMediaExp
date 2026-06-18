import pandas as pd
df = pd.read_csv('social_feed_test.csv', sep=';')
print(f'Total rows: {len(df)}')
print(f'Total conditions: {df["condition"].nunique()}')
print(f'Rows per condition: {df.groupby("condition").size().unique()}')
print(f'doc_id range: {df["doc_id"].min()} ~ {df["doc_id"].max()}')

df['group'] = df['condition'].apply(lambda x: x.split('_')[1])
for g in sorted(df['group'].unique()):
    sub = df[df['group']==g]
    print(f'  Group {g}: {sub["condition"].nunique()} conditions')

# Spot check: group 1, doc_id=1 text across first 4 combos
print('\nSpot check - Group 1, doc_id=1 across combos:')
g1 = df[(df['group']=='1') & (df['doc_id']==1)]
for _, row in g1.head(4).iterrows():
    print(f'  {row["condition"]}: {str(row["text"])[:60]}...')

# Check backgrounds are same across all conditions
print('\nBackground check - doc_id=5 text in first 3 conditions:')
for cond in sorted(df['condition'].unique())[:3]:
    txt = df[(df['condition']==cond) & (df['doc_id']==5)]['text'].values[0]
    print(f'  {cond}: {str(txt)[:60]}...')
