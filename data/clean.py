import pandas as pd

W = 5

def add_rolling_features(df):
    # Ensure time order per team
    df = df.sort_values('datetime')

    # Days rest
    df['prev_date'] = df['datetime'].shift(1)
    df['rest_days'] = (df['datetime'] - df['prev_date']).dt.days.astype('float32')

    # Shift raw stats so current match doesn't leak into its own features
    for col in ['gf','ga','xg_for','xg_against']:
        df[f'{col}_pre'] = df[col].shift(1)

    # Rolling means (last W matches, pre-match only)
    roll = df[[f'{c}_pre' for c in ['gf','ga','xg_for','xg_against']]].rolling(W, min_periods=1).mean()
    roll.columns = [c.replace('_pre','') + f'_roll{W}' for c in roll.columns]
    df = pd.concat([df, roll], axis=1)

    # Rolling differentials
    df[f'xg_diff_roll{W}'] = df[f'xg_for_roll{W}'] - df[f'xg_against_roll{W}']
    df[f'gd_roll{W}']      = df[f'gf_roll{W}'] - df[f'ga_roll{W}']

    # Simple trend: rolling mean of xg_diff’s first difference (very light)
    df['xg_diff_pre'] = (df['xg_for'] - df['xg_against']).shift(1)
    df[f'xg_diff_trend{W}'] = df['xg_diff_pre'].diff().rolling(W, min_periods=1).mean()

    # Drop helper columns we don’t want to keep
    drop_cols = ['prev_date'] + [f'{c}_pre' for c in ['gf','ga','xg_for','xg_against']] + ['xg_diff_pre']
    df = df.drop(columns=drop_cols)

    return df

def normalize(data):
    matches = pd.json_normalize(data)
    keep = {
        'id': 'match_id',
        'datetime': 'datetime',
        'isResult': 'is_result',
        'h.id': 'home_id',
        'h.title': 'home',
        'a.id': 'away_id',
        'a.title': 'away',
        'goals.h': 'goals_h',
        'goals.a': 'goals_a',
        'xG.h': 'xg_h',
        'xG.a': 'xg_a',
    }
    m = matches[list(keep.keys())].rename(columns=keep)

    # Types
    m['match_id'] = m['match_id'].astype(str)
    m['home_id']  = m['home_id'].astype(str)
    m['away_id']  = m['away_id'].astype(str)
    m['datetime'] = pd.to_datetime(m['datetime'], errors='coerce')

    for c in ['goals_h','goals_a','xg_h','xg_a']:
        m[c] = pd.to_numeric(m[c], errors='coerce')

    # Mark fixtures vs finished matches
    finished = m[m['is_result'] == True].copy()
    fixtures = m[m['is_result'] == False].copy()


    # shift data to reflect home team and away team statistics, 2 rows per match, for each team
    home_side = finished[['match_id','datetime','home_id','away_id','goals_h','goals_a','xg_h','xg_a']].copy()
    home_side = home_side.rename(columns={
        'home_id':'team_id', 'away_id':'opponent_id',
        'goals_h':'gf', 'goals_a':'ga',
        'xg_h':'xg_for', 'xg_a':'xg_against'
    })
    home_side['is_home'] = 1

    away_side = finished[['match_id','datetime','home_id','away_id','goals_h','goals_a','xg_h','xg_a']].copy()
    away_side = away_side.rename(columns={
        'away_id':'team_id', 'home_id':'opponent_id',
        'goals_a':'gf', 'goals_h':'ga',
        'xg_a':'xg_for', 'xg_h':'xg_against'
    })
    away_side['is_home'] = 0


    team_match = pd.concat([home_side, away_side], ignore_index=True)
    team_match = team_match.sort_values(['team_id','datetime']).reset_index(drop=True)
    team_match = (
        team_match
        .groupby('team_id', group_keys=False)
        .apply(add_rolling_features)
        .reset_index(drop=True)
    )

    team_match['rest_days'] = team_match['rest_days'].fillna(team_match['rest_days'].median())

    home_view = team_match[team_match['is_home']==1].copy()
    away_view = team_match[team_match['is_home']==0].copy()

    # Columns to carry over as features
    base_feat_cols = [
        'rest_days',
        f'xg_for_roll{W}', f'xg_against_roll{W}', f'xg_diff_roll{W}',
        f'gf_roll{W}', f'ga_roll{W}', f'gd_roll{W}',
        f'xg_diff_trend{W}',
    ]
    home_feat = home_view[['match_id','datetime','team_id','opponent_id','gf','ga'] + base_feat_cols].copy()
    away_feat = away_view[['match_id','datetime','team_id','opponent_id','gf','ga'] + base_feat_cols].copy()

    home_feat = home_feat.add_suffix('_h'); home_feat = home_feat.rename(columns={'match_id_h':'match_id','datetime_h':'datetime'})
    away_feat = away_feat.add_suffix('_a'); away_feat = away_feat.rename(columns={'match_id_a':'match_id','datetime_a':'datetime'})

    train_df = (
        home_feat
        .merge(away_feat, on=['match_id','datetime'], how='inner')
        .merge(
            finished[['match_id','goals_h','goals_a','xg_h','xg_a','home_id','away_id','home','away']],
            on='match_id', how='left'
        )
    )

    # Final target and simple extras
    train_df['home_adv']  = 1  # explicit flag (helps simple models)
    train_df['goal_diff'] = train_df['goals_h'] - train_df['goals_a']  # <-- TARGET

    feature_cols = (
        ['home_adv'] +
        [c for c in train_df.columns if c.endswith(('_h','_a')) and any(x in c for x in ['xg_','gd_','rest_days','gf_','ga_'])]
    )

    for c in feature_cols:
        if train_df[c].isna().any():
            train_df[c] = train_df[c].fillna(train_df[c].mean())

    # Keep only columns we need for modeling & IDs for reference
    keep_cols = [
        'match_id','datetime','home_id','away_id','home','away','goal_diff','xg_h','xg_a'
    ] + feature_cols

    train_df = train_df[keep_cols].sort_values('datetime').reset_index(drop=True)
    print(len(train_df), "matches after normalization")
    return train_df