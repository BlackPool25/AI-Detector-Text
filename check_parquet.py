import pandas as pd

df = pd.read_parquet('data/raw/human_text_1m_mixed.parquet')

print('Source distribution:')
print(df['source'].value_counts())

print('\nSample texts from each source:')
for src in df['source'].unique()[:3]:
    print(f'\n--- {src.upper()} ---')
    print(df[df['source']==src]['text'].iloc[0][:200])
