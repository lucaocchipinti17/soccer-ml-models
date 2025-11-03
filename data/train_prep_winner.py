import pandas as pd
import numpy as np

def prepare_data(df, split=0.8):
    features = [
        'home_adv',
        'rest_days_h', 'rest_days_a',
        'xg_for_roll5_h', 'xg_against_roll5_h', 'xg_diff_roll5_h',
        'xg_for_roll5_a', 'xg_against_roll5_a', 'xg_diff_roll5_a',
        'xg_diff_trend5_h', 'xg_diff_trend5_a'
    ]

    # Use goal_diff to derive winner labels: home win, draw, away win
    # y_onehot order: [home_win, draw, away_win]
    leak_cols = ['match_id','home_id','away_id','home','away',
                 'gf_h','ga_h','gf_a','ga_a','xg_h','xg_a']
    keep = [c for c in df.columns if c not in leak_cols]

    df = df[keep]

    # Drop rows missing any required feature or goal_diff (to derive label)
    df = df.dropna(subset=features + ['goal_diff']).reset_index(drop=True)

    # Chronological split (no shuffle)
    split_index = int(len(df) * split)
    train_df = df.iloc[:split_index].reset_index(drop=True)
    valid_df = df.iloc[split_index:].reset_index(drop=True)

    # Standardize by train stats only
    mu = train_df[features].mean()
    sd = train_df[features].std(ddof=0).clip(lower=1e-6)

    X_train = ((train_df[features] - mu) / sd).astype('float32').to_numpy()
    X_valid = ((valid_df[features] - mu) / sd).astype('float32').to_numpy()

    # Build one-hot labels
    def to_onehot(goal_diff_series: pd.Series) -> np.ndarray:
        gd = goal_diff_series.to_numpy()
        home_win = (gd > 0).astype(np.float32)
        draw = (gd == 0).astype(np.float32)
        away_win = (gd < 0).astype(np.float32)
        return np.stack([home_win, draw, away_win], axis=1)

    y_train = to_onehot(train_df['goal_diff']).astype('float32')
    y_valid = to_onehot(valid_df['goal_diff']).astype('float32')

    print("Train size:", len(train_df), " | Test size:", len(valid_df))
    print("X_train shape:", X_train.shape, " | y_train shape:", y_train.shape)
    print("X_test shape:", X_valid.shape, " | y_test shape:", y_valid.shape)

    return X_train, y_train, X_valid, y_valid


