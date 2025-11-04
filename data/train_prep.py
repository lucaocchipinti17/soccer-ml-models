import pandas as pd

def prepare_data(df, split=0.8):
    # Base rolling features
    base_features = [
        'home_adv',
        'rest_days_h', 'rest_days_a',
        'xg_for_roll5_h', 'xg_against_roll5_h', 'xg_diff_roll5_h',
        'xg_for_roll5_a', 'xg_against_roll5_a', 'xg_diff_roll5_a',
        'xg_diff_trend5_h', 'xg_diff_trend5_a'
    ]
    
    # Contextual metrics from get_aggregate_context_data (excluding 'team' and 'season' identifiers)
    contextual_metrics = [
        'shots_per_game', 'goals_per_game', 'xG_per_game', 'goals_per_xG', 'xG_per_shot',
        'conversion_rate', 'xG_diff_per_game', 'goal_diff_per_game',
        'xG_share_open_play', 'xG_share_set_pieces', 'penalty_goal_share', 
        'set_piece_xG_allowed', 'open_play_xG_diff_per_game',
        'main_form_share', 'num_formations_used', 'xG_per_90_main_formation', 
        'xGA_per_90_main_formation',
        'time_leading_share', 'time_trailing_share', 'xG_for_per_90_when_trailing',
        'xG_against_per_90_when_leading', 'goal_diff_per_90_when_even',
        'early_goals_share', 'late_goals_share', 'xG_diff_first_half', 'xG_diff_second_half',
        'percent_shots_inside_box', 'percent_xG_inside_box', 'xG_allowed_six_yard_ratio',
        'xG_diff_six_yard',
        'fast_attack_xG_share', 'slow_attack_xG_share', 'fast_attack_efficiency',
        'xG_allowed_from_fast_attacks_per_90',
        'percent_blocked_shots', 'percent_shots_on_target', 'goal_conversion_rate',
        'xG_saved_share',
        'shots_against_per_game', 'xG_against_per_game', 'goals_against_per_game',
        'goals_allowed_per_xGA', 'xGA_per_shot',
        'attacking_index', 'defensive_index', 'momentum_profile'
    ]
    
    # Build feature list with home_ and away_ prefixes for contextual metrics
    contextual_features = []
    for metric in contextual_metrics:
        contextual_features.append(f'home_{metric}')
        contextual_features.append(f'away_{metric}')
    
    features = base_features + contextual_features

    target = 'goal_diff'

    leak_cols = ['match_id','home_id','away_id','home','away',
             'gf_h','ga_h','gf_a','ga_a','xg_h','xg_a']  # drop xg_h/xg_a (post-match)
    keep = [c for c in df.columns if c not in leak_cols]

    df = df[keep]

    # Fill NaN values in contextual features with 0.0 (they're safe defaults from clean_team_data)
    for feat in contextual_features:
        if feat in df.columns:
            df[feat] = df[feat].fillna(0.0)
    
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
