import json
from understatapi import UnderstatClient

understat = UnderstatClient()
team_context_data = understat.team(team="Manchester United").get_context_data(season="2019")
num_games = len(understat.team(team="Manchester United").get_match_data(season="2019"))

with open('data/context_data.json', 'r') as f:
    data = json.load(f)


def get_aggregate_context_data(team, season):

    situations = data['situation']
    formations = data['formation']
    game_state = data['gameState']
    timing = data['timing']
    shot_zone = data['shotZone']
    attack_speed = data['attackSpeed']
    result = data['result']


    def safe_div(n, d):
        return n / d if d else 0.0

    situation_keys = ["OpenPlay", "FromCorner", "DirectFreekick", "SetPiece", "Penalty"]
    set_piece_like = ["FromCorner", "DirectFreekick", "SetPiece"] 


    # totals
    total_shots = sum(situations[k]["shots"] for k in situation_keys)
    total_goals = sum(situations[k]["goals"] for k in situation_keys)
    total_xG = sum(situations[k]["xG"] for k in situation_keys)

    total_shots_against = sum(situations[k]["against"]["shots"] for k in situation_keys)
    total_goals_against = sum(situations[k]["against"]["goals"] for k in situation_keys)
    total_xG_against = sum(situations[k]["against"]["xG"]    for k in situation_keys)

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
    xG_share_open_play = safe_div(situations['OpenPlay']['xG'], total_xG)
    xG_share_set_pieces = safe_div(sum(situations[k]['xG'] for k in set_piece_like), total_xG)
    penalty_goal_share = safe_div(situations['Penalty']['goals'], total_goals)
    set_piece_xG_allowed = safe_div(sum(situations[k]['against']['xG'] for k in set_piece_like), total_xG_against)
    open_play_xG_diff_per_game = safe_div(
        situations['OpenPlay']['xG'] - situations['OpenPlay']['against']['xG'], num_games
    )

    # -formation metrics
    total_time_form = sum(v['time'] for v in formations.values())
    main_formation = max(formations.keys(), key=lambda k: formations[k]['time'])
    main_form = formations[main_formation]
    main_form_share = safe_div(main_form['time'], total_time_form)
    num_formations_used = sum(1 for v in formations.values() if v['time'] > 0)
    xG_per_90_main_formation = safe_div(main_form['xG'], main_form['time'] / 90)
    xGA_per_90_main_formation = safe_div(main_form['against']['xG'], main_form['time'] / 90)


    # game state metrics
    time_leading = game_state['Goal diff +1']['time'] + game_state['Goal diff > +1']['time']
    time_trailing = game_state['Goal diff -1']['time'] + game_state['Goal diff < -1']['time']
    time_even = game_state['Goal diff 0']['time']
    total_time = time_leading + time_trailing + time_even

    time_leading_share = safe_div(time_leading, total_time)
    time_trailing_share = safe_div(time_trailing, total_time)
    xG_for_per_90_when_trailing = safe_div(
        game_state['Goal diff -1']['xG'] + game_state['Goal diff < -1']['xG'],
        (time_trailing / 90)
    )
    xG_against_per_90_when_leading = safe_div(
        game_state['Goal diff +1']['against']['xG'] + game_state['Goal diff > +1']['against']['xG'],
        (time_leading / 90)
    )
    goal_diff_per_90_when_even = safe_div(
        game_state['Goal diff 0']['goals'] - game_state['Goal diff 0']['against']['goals'],
        (time_even / 90)
    )

    # timing metrics
    goals_early = timing['1-15']['goals'] + timing['16-30']['goals']
    goals_late = timing['76+']['goals']
    total_goals_timing = sum(block['goals'] for block in timing.values())

    early_goals_share = safe_div(goals_early, total_goals_timing)
    late_goals_share = safe_div(goals_late, total_goals_timing)
    xG_first_half = timing['1-15']['xG'] + timing['16-30']['xG'] + timing['31-45']['xG']
    xGA_first_half = timing['1-15']['against']['xG'] + timing['16-30']['against']['xG'] + timing['31-45']['against']['xG']
    xG_second_half = timing['46-60']['xG'] + timing['61-75']['xG'] + timing['76+']['xG']
    xGA_second_half = timing['46-60']['against']['xG'] + timing['61-75']['against']['xG'] + timing['76+']['against']['xG']
    xG_diff_first_half = safe_div(xG_first_half - xGA_first_half, num_games)
    xG_diff_second_half = safe_div(xG_second_half - xGA_second_half, num_games)

    # shooting metrics
    shots_inside_box = shot_zone['shotPenaltyArea']['shots'] + shot_zone['shotSixYardBox']['shots']
    percent_shots_inside_box = safe_div(shots_inside_box, total_shots)
    xG_inside_box = shot_zone['shotPenaltyArea']['xG'] + shot_zone['shotSixYardBox']['xG']
    percent_xG_inside_box = safe_div(xG_inside_box, total_xG)
    xG_allowed_six_yard_ratio = safe_div(shot_zone['shotSixYardBox']['against']['xG'], total_xG_against)
    xG_diff_six_yard = safe_div(
        shot_zone['shotSixYardBox']['xG'] - shot_zone['shotSixYardBox']['against']['xG'], num_games
    )

    # attack speed metrics
    fast = attack_speed['Fast']
    slow = attack_speed['Slow']
    fast_attack_xG_share = safe_div(fast['xG'], total_xG)
    slow_attack_xG_share = safe_div(slow['xG'], total_xG)
    fast_attack_efficiency = safe_div(fast['goals'], fast['xG'])
    xG_allowed_from_fast_attacks_per_90 = safe_div(fast['against']['xG'], num_games)

    # shot result metrics
    blocked = result['BlockedShot']
    saved = result['SavedShot']
    goals = result['Goal']
    total_result_shots = sum(v['shots'] for v in result.values())
    percent_blocked_shots = safe_div(blocked['shots'], total_result_shots)
    percent_shots_on_target = safe_div(saved['shots'] + goals['shots'], total_result_shots)
    goal_conversion_rate = safe_div(goals['shots'], total_result_shots)
    xG_saved_share = safe_div(saved['xG'], total_xG)

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
        'main_formation': main_formation,
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