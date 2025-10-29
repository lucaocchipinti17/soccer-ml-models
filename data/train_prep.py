import pandas as pd

def prepare_data(df, split=0.8):
    features = [
        'home_adv',
        'rest_days_h', 'rest_days_a',
        'xg_for_roll5_h', 'xg_against_roll5_h', 'xg_diff_roll5_h',
        'xg_for_roll5_a', 'xg_against_roll5_a', 'xg_diff_roll5_a',
        'xg_diff_trend5_h', 'xg_diff_trend5_a'
    ]

    target = 'goal_diff'

    leak_cols = ['match_id','home_id','away_id','home','away',
             'gf_h','ga_h','gf_a','ga_a','xg_h','xg_a']  # drop xg_h/xg_a (post-match)
    keep = [c for c in df.columns if c not in leak_cols]

    df = df[keep]

    # for now, drop NaN
    df = df.dropna(subset=features + [target]).reset_index(drop=True)
    split_index = int(len(df)*split)

    train_df = df.iloc[:split_index].reset_index(drop=True)
    valid_df  = df.iloc[split_index:].reset_index(drop=True)

    mu = train_df[features].mean()
    sd = train_df[features].std(ddof=0).clip(lower=1e-6)  # guard against zero-variance

    X_train = ((train_df[features] - mu) / sd).astype('float32').to_numpy()
    y_train = train_df[target].astype('float32').to_numpy()

    X_valid = ((valid_df[features] - mu) / sd).astype('float32').to_numpy()
    y_valid = valid_df[target].astype('float32').to_numpy()

    print("Train size:", len(train_df), " | Test size:", len(valid_df))
    print("X_train shape:", X_train.shape, " | y_train shape:", y_train.shape)
    print("X_test shape:", X_valid.shape, " | y_test shape:", y_valid.shape)

    return X_train, y_train, X_valid, y_valid
