import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
mskf = MultilabelStratifiedKFold(n_splits=5, random_state=12)

train_df2 = pd.read_csv('data/train.csv')
# train_df2 = train_df2.sample(frac=1).reset_index(drop=True)
train_df2 = train_df2.drop(['grapheme'], axis=1)
train_df2['id'] = train_df2['image_id'].apply(lambda x: int(x.split('_')[1]))
X, y = train_df2[['id', 'grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values[:,0], train_df2.values[:,1:]

train_df2['fold'] = -1
for fld, (_, test_idx) in enumerate(mskf.split(X, y)):
    train_df2.iloc[test_idx, -1] = fld

train_df2.to_csv('train_with_folds.csv')
