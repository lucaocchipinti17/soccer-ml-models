import json
from understatapi import UnderstatClient

understat = UnderstatClient()
team_context_data = understat.team(team="Manchester United").get_context_data(season="2019")
num_games = len(understat.team(team="Manchester United").get_match_data(season="2019"))


def _get_empty_metrics_dict(team, season):
    """Return a dictionary with all NaN values when data is unavailable"""
    nan = float('nan')
    return {
        # Core efficiency metrics
        'shots_per_game': nan,
        'goals_per_game': nan,
        'xG_per_game': nan,
        'goals_per_xG': nan,
        'xG_per_shot': nan,
        'conversion_rate': nan,
        'xG_diff_per_game': nan,
        'goal_diff_per_game': nan,
        
        # Situational metrics
        'xG_share_open_play': nan,
        'xG_share_set_pieces': nan,
        'penalty_goal_share': nan,
        'set_piece_xG_allowed': nan,
        'open_play_xG_diff_per_game': nan,
        
        # Formation metrics
        'main_form_share': nan,
        'num_formations_used': nan,
        'xG_per_90_main_formation': nan,
        'xGA_per_90_main_formation': nan,
        
        # Game state metrics
        'time_leading_share': nan,
        'time_trailing_share': nan,
        'xG_for_per_90_when_trailing': nan,
        'xG_against_per_90_when_leading': nan,
        'goal_diff_per_90_when_even': nan,
        
        # Timing metrics
        'early_goals_share': nan,
        'late_goals_share': nan,
        'xG_diff_first_half': nan,
        'xG_diff_second_half': nan,
        
        # Shooting metrics
        'percent_shots_inside_box': nan,
        'percent_xG_inside_box': nan,
        'xG_allowed_six_yard_ratio': nan,
        'xG_diff_six_yard': nan,
        
        # Attack speed metrics
        'fast_attack_xG_share': nan,
        'slow_attack_xG_share': nan,
        'fast_attack_efficiency': nan,
        'xG_allowed_from_fast_attacks_per_90': nan,
        
        # Shot result metrics
        'percent_blocked_shots': nan,
        'percent_shots_on_target': nan,
        'goal_conversion_rate': nan,
        'xG_saved_share': nan,
        
        # Defensive metrics
        'shots_against_per_game': nan,
        'xG_against_per_game': nan,
        'goals_against_per_game': nan,
        'goals_allowed_per_xGA': nan,
        'xGA_per_shot': nan,
        
        # Composite metrics
        'attacking_index': nan,
        'defensive_index': nan,
        'momentum_profile': nan,
        
        # Identifier fields
        'team': team,
        'season': season,
    }


def get_aggregate_context_data(team, season):
    try:
        understat = UnderstatClient()
        data = understat.team(team=team).get_context_data(season=season)
        match_data = understat.team(team=team).get_match_data(season=season)
        if not isinstance(data, dict):
            # Return all NaN values if data is invalid
            return _get_empty_metrics_dict(team, season)
        num_games = len(match_data) if match_data else 0
        if num_games == 0:
            # Return all NaN values if no games found
            return _get_empty_metrics_dict(team, season)
    except Exception:
        # Return all NaN values if API call fails
        return _get_empty_metrics_dict(team, season)

    def safe_get(d, key, default=0.0):
        """Safely get value from dictionary, returning default if key doesn't exist"""
        if not isinstance(d, dict):
            return default
        return d.get(key, default)
    
    def safe_get_nested(d, *keys, default=0.0):
        """Safely get nested value from dictionary"""
        current = d
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current if current is not None else default

    situations = data.get('situation', {})
    formations = data.get('formation', {})
    game_state = data.get('gameState', {})
    timing = data.get('timing', {})
    shot_zone = data.get('shotZone', {})
    attack_speed = data.get('attackSpeed', {})
    result = data.get('result', {})

    def safe_div(n, d):
        return n / d if d else 0.0

    situation_keys = ["OpenPlay", "FromCorner", "DirectFreekick", "SetPiece", "Penalty"]
    set_piece_like = ["FromCorner", "DirectFreekick", "SetPiece"] 

    # totals
    total_shots = sum(safe_get_nested(situations, k, "shots", default=0.0) for k in situation_keys)
    total_goals = sum(safe_get_nested(situations, k, "goals", default=0.0) for k in situation_keys)
    total_xG = sum(safe_get_nested(situations, k, "xG", default=0.0) for k in situation_keys)

    total_shots_against = sum(safe_get_nested(situations, k, "against", "shots", default=0.0) for k in situation_keys)
    total_goals_against = sum(safe_get_nested(situations, k, "against", "goals", default=0.0) for k in situation_keys)
    total_xG_against = sum(safe_get_nested(situations, k, "against", "xG", default=0.0) for k in situation_keys)

    # core efficiency metrics
    shots_per_game = safe_div(total_shots, num_games)
    goals_per_game = safe_div(total_goals, num_games)
    xG_per_game = safe_div(total_xG, num_games)
    goals_per_xG = safe_div(total_goals, total_xG)
    xG_per_shot = safe_div(total_xG, total_shots)
    conversion_rate = safe_div(total_goals, total_shots)
    xG_diff_per_game = safe_div(total_xG - total_xG_against, num_games)
    goal_diff_per_game = safe_div(total_goals - total_goals_against, num_games)


    # situational metrics
    xG_share_open_play = safe_div(safe_get_nested(situations, 'OpenPlay', 'xG', default=0.0), total_xG)
    xG_share_set_pieces = safe_div(sum(safe_get_nested(situations, k, 'xG', default=0.0) for k in set_piece_like), total_xG)
    penalty_goal_share = safe_div(safe_get_nested(situations, 'Penalty', 'goals', default=0.0), total_goals)
    set_piece_xG_allowed = safe_div(sum(safe_get_nested(situations, k, 'against', 'xG', default=0.0) for k in set_piece_like), total_xG_against)
    open_play_xG_diff_per_game = safe_div(
        safe_get_nested(situations, 'OpenPlay', 'xG', default=0.0) - safe_get_nested(situations, 'OpenPlay', 'against', 'xG', default=0.0), 
        num_games
    )

    # formation metrics
    total_time_form = sum(safe_get(v, 'time', default=0.0) for v in formations.values() if isinstance(v, dict))
    if formations and total_time_form > 0:
        main_formation = max(formations.keys(), key=lambda k: safe_get_nested(formations, k, 'time', default=0.0))
        main_form = safe_get(formations, main_formation, default={})
    else:
        main_formation = ""
        main_form = {}
    main_form_share = safe_div(safe_get(main_form, 'time', default=0.0), total_time_form)
    num_formations_used = sum(1 for v in formations.values() if isinstance(v, dict) and safe_get(v, 'time', default=0.0) > 0)
    xG_per_90_main_formation = safe_div(safe_get(main_form, 'xG', default=0.0), safe_div(safe_get(main_form, 'time', default=0.0), 90))
    xGA_per_90_main_formation = safe_div(safe_get_nested(main_form, 'against', 'xG', default=0.0), safe_div(safe_get(main_form, 'time', default=0.0), 90))


    # game state metrics
    time_leading = safe_get_nested(game_state, 'Goal diff +1', 'time', default=0.0) + safe_get_nested(game_state, 'Goal diff > +1', 'time', default=0.0)
    time_trailing = safe_get_nested(game_state, 'Goal diff -1', 'time', default=0.0) + safe_get_nested(game_state, 'Goal diff < -1', 'time', default=0.0)
    time_even = safe_get_nested(game_state, 'Goal diff 0', 'time', default=0.0)
    total_time = time_leading + time_trailing + time_even

    time_leading_share = safe_div(time_leading, total_time)
    time_trailing_share = safe_div(time_trailing, total_time)
    xG_for_per_90_when_trailing = safe_div(
        safe_get_nested(game_state, 'Goal diff -1', 'xG', default=0.0) + safe_get_nested(game_state, 'Goal diff < -1', 'xG', default=0.0),
        (time_trailing / 90) if time_trailing > 0 else 1.0
    )
    xG_against_per_90_when_leading = safe_div(
        safe_get_nested(game_state, 'Goal diff +1', 'against', 'xG', default=0.0) + safe_get_nested(game_state, 'Goal diff > +1', 'against', 'xG', default=0.0),
        (time_leading / 90) if time_leading > 0 else 1.0
    )
    goal_diff_per_90_when_even = safe_div(
        safe_get_nested(game_state, 'Goal diff 0', 'goals', default=0.0) - safe_get_nested(game_state, 'Goal diff 0', 'against', 'goals', default=0.0),
        (time_even / 90) if time_even > 0 else 1.0
    )

    # timing metrics
    goals_early = safe_get_nested(timing, '1-15', 'goals', default=0.0) + safe_get_nested(timing, '16-30', 'goals', default=0.0)
    goals_late = safe_get_nested(timing, '76+', 'goals', default=0.0)
    total_goals_timing = sum(safe_get(v, 'goals', default=0.0) for v in timing.values() if isinstance(v, dict))

    early_goals_share = safe_div(goals_early, total_goals_timing)
    late_goals_share = safe_div(goals_late, total_goals_timing)
    xG_first_half = safe_get_nested(timing, '1-15', 'xG', default=0.0) + safe_get_nested(timing, '16-30', 'xG', default=0.0) + safe_get_nested(timing, '31-45', 'xG', default=0.0)
    xGA_first_half = safe_get_nested(timing, '1-15', 'against', 'xG', default=0.0) + safe_get_nested(timing, '16-30', 'against', 'xG', default=0.0) + safe_get_nested(timing, '31-45', 'against', 'xG', default=0.0)
    xG_second_half = safe_get_nested(timing, '46-60', 'xG', default=0.0) + safe_get_nested(timing, '61-75', 'xG', default=0.0) + safe_get_nested(timing, '76+', 'xG', default=0.0)
    xGA_second_half = safe_get_nested(timing, '46-60', 'against', 'xG', default=0.0) + safe_get_nested(timing, '61-75', 'against', 'xG', default=0.0) + safe_get_nested(timing, '76+', 'against', 'xG', default=0.0)
    xG_diff_first_half = safe_div(xG_first_half - xGA_first_half, num_games)
    xG_diff_second_half = safe_div(xG_second_half - xGA_second_half, num_games)

    # shooting metrics
    shots_inside_box = safe_get_nested(shot_zone, 'shotPenaltyArea', 'shots', default=0.0) + safe_get_nested(shot_zone, 'shotSixYardBox', 'shots', default=0.0)
    percent_shots_inside_box = safe_div(shots_inside_box, total_shots)
    xG_inside_box = safe_get_nested(shot_zone, 'shotPenaltyArea', 'xG', default=0.0) + safe_get_nested(shot_zone, 'shotSixYardBox', 'xG', default=0.0)
    percent_xG_inside_box = safe_div(xG_inside_box, total_xG)
    xG_allowed_six_yard_ratio = safe_div(safe_get_nested(shot_zone, 'shotSixYardBox', 'against', 'xG', default=0.0), total_xG_against)
    xG_diff_six_yard = safe_div(
        safe_get_nested(shot_zone, 'shotSixYardBox', 'xG', default=0.0) - safe_get_nested(shot_zone, 'shotSixYardBox', 'against', 'xG', default=0.0), 
        num_games
    )

    # attack speed metrics
    fast = safe_get(attack_speed, 'Fast', default={})
    slow = safe_get(attack_speed, 'Slow', default={})
    fast_attack_xG_share = safe_div(safe_get(fast, 'xG', default=0.0), total_xG)
    slow_attack_xG_share = safe_div(safe_get(slow, 'xG', default=0.0), total_xG)
    fast_attack_efficiency = safe_div(safe_get(fast, 'goals', default=0.0), safe_get(fast, 'xG', default=0.0))
    xG_allowed_from_fast_attacks_per_90 = safe_div(safe_get_nested(fast, 'against', 'xG', default=0.0), num_games)

    # shot result metrics
    blocked = safe_get(result, 'BlockedShot', default={})
    saved = safe_get(result, 'SavedShot', default={})
    goals = safe_get(result, 'Goal', default={})
    total_result_shots = sum(safe_get(v, 'shots', default=0.0) for v in result.values() if isinstance(v, dict))
    percent_blocked_shots = safe_div(safe_get(blocked, 'shots', default=0.0), total_result_shots)
    percent_shots_on_target = safe_div(safe_get(saved, 'shots', default=0.0) + safe_get(goals, 'shots', default=0.0), total_result_shots)
    goal_conversion_rate = safe_div(safe_get(goals, 'shots', default=0.0), total_result_shots)
    xG_saved_share = safe_div(safe_get(saved, 'xG', default=0.0), total_xG)

    # defensive metrics
    shots_against_per_game = safe_div(total_shots_against, num_games)
    xG_against_per_game = safe_div(total_xG_against, num_games)
    goals_against_per_game = safe_div(total_goals_against, num_games)
    goals_allowed_per_xGA = safe_div(total_goals_against, total_xG_against)
    xGA_per_shot = safe_div(total_xG_against, total_shots_against)

    # composite metrics
    attacking_index = (0.4 * xG_per_shot) + (0.3 * goals_per_xG) + (0.3 * fast_attack_xG_share)
    defensive_index = safe_div(1.0, (0.5 * xG_against_per_game + 0.3 * set_piece_xG_allowed + 0.2 * xG_allowed_six_yard_ratio))
    momentum_profile = xG_diff_second_half - xG_diff_first_half

    # return all metrics as a dictionary
    return {
        # Core efficiency metrics
        'shots_per_game': shots_per_game,
        'goals_per_game': goals_per_game,
        'xG_per_game': xG_per_game,
        'goals_per_xG': goals_per_xG,
        'xG_per_shot': xG_per_shot,
        'conversion_rate': conversion_rate,
        'xG_diff_per_game': xG_diff_per_game,
        'goal_diff_per_game': goal_diff_per_game,
        
        # Situational metrics
        'xG_share_open_play': xG_share_open_play,
        'xG_share_set_pieces': xG_share_set_pieces,
        'penalty_goal_share': penalty_goal_share,
        'set_piece_xG_allowed': set_piece_xG_allowed,
        'open_play_xG_diff_per_game': open_play_xG_diff_per_game,
        
        # Formation metrics
        'main_form_share': main_form_share,
        'num_formations_used': num_formations_used,
        'xG_per_90_main_formation': xG_per_90_main_formation,
        'xGA_per_90_main_formation': xGA_per_90_main_formation,
        
        # Game state metrics
        'time_leading_share': time_leading_share,
        'time_trailing_share': time_trailing_share,
        'xG_for_per_90_when_trailing': xG_for_per_90_when_trailing,
        'xG_against_per_90_when_leading': xG_against_per_90_when_leading,
        'goal_diff_per_90_when_even': goal_diff_per_90_when_even,
        
        # Timing metrics
        'early_goals_share': early_goals_share,
        'late_goals_share': late_goals_share,
        'xG_diff_first_half': xG_diff_first_half,
        'xG_diff_second_half': xG_diff_second_half,
        
        # Shooting metrics
        'percent_shots_inside_box': percent_shots_inside_box,
        'percent_xG_inside_box': percent_xG_inside_box,
        'xG_allowed_six_yard_ratio': xG_allowed_six_yard_ratio,
        'xG_diff_six_yard': xG_diff_six_yard,
        
        # Attack speed metrics
        'fast_attack_xG_share': fast_attack_xG_share,
        'slow_attack_xG_share': slow_attack_xG_share,
        'fast_attack_efficiency': fast_attack_efficiency,
        'xG_allowed_from_fast_attacks_per_90': xG_allowed_from_fast_attacks_per_90,
        
        # Shot result metrics
        'percent_blocked_shots': percent_blocked_shots,
        'percent_shots_on_target': percent_shots_on_target,
        'goal_conversion_rate': goal_conversion_rate,
        'xG_saved_share': xG_saved_share,
        
        # Defensive metrics
        'shots_against_per_game': shots_against_per_game,
        'xG_against_per_game': xG_against_per_game,
        'goals_against_per_game': goals_against_per_game,
        'goals_allowed_per_xGA': goals_allowed_per_xGA,
        'xGA_per_shot': xGA_per_shot,
        
        # Composite metrics
        'attacking_index': attacking_index,
        'defensive_index': defensive_index,
        'momentum_profile': momentum_profile,
        
        # Identifier fields
        'team': team,
        'season': season,
    }