# Soccer ML Models

End-to-end experiments and utilities for **soccer match outcome/score prediction**.

---

## What’s Inside

- Web scraping utilities to collect, enhance, and clean historical match data from Top 5 Leagues, seasons 2014-2025
- Training scripts for goal differential and match outcome models.

### Algorithms / Models
- Custom multi-layer neural network implemented in NumPy to predict goal differential (home goals - away goals) and home outcome (Win/Draw/Lose)
- NN complete with two layers of customizable size, Adam optimizer, L2 weight decay, and Huber Loss.
- XGBoost model imported from XGBoost library to benchmark neural networks against.

### Match Data 

Match Data for each game from Top 5 European leagues from 2024-2025 stored in normalized_data.csv, contains rolling averages of key stats for each team over last 5 games.
normalized_data_with_context contains ~80 columns of additional statistics for each team throughout that season, including:
1. Core Efficiency Metrics (Shots per game, Goals per game..)
2. Situational Metrics (% of xG scored in open play, xG from penalties...)
3. Formation Metrics (% of time spent in main formation, xG per 90 in main formation...)
4. Game State Metrics (% of time leading in game, % of time trailing...)
5. Timing Metrics (% of goals scored in first half, xG differential in second half per game...)
6. Shooting Metrics (% of shots inside the box, % of shots outside the box...)
7. Attack Speed Metrics (% of xG from fast attacks, xG allowed from fast attacks per 90...)
8. Shot Result Metrics (% of blocked shots, goal conversation rate...)
9. Defensive Metrics (shots against per game, goals allowed per xGa....)
10. Composite Scores (Attacking Index, Defensive Index, Composite Index)

## Benchmarks: Handmade Neural Network vs. XGBoost

### Overall Performance

| Metric | Neural Network | XGBoost | 
|---|---:|---:|
|Accuracy|0.5358|0.5441|           
|Cross-Entropy Loss|0.9680|0.9658|              


### Per-Class Performance

NEURAL NETWORK - Per-class Performance:
| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Home Win | 0.5986    | 0.7150 | 0.6517   | 1723    |
| Draw     | 0.3362    | 0.1584 | 0.2153   | 1004    |
| Away Win | 0.5115    | 0.5923 | 0.5489   | 1241    |
| Accuracy     | —         | —      | 0.5358   | 3968    |
| Macro Avg    | 0.4821    | 0.4886 | 0.4720   | 3968    |
| Weighted Avg | 0.5050    | 0.5358 | 0.5091   | 3968    |


XGBOOST - Per-class Performance:
| Class    | Precision | Recall | F1-Score | Support |
| -------- | --------- | ------ | -------- | ------- |
| Home Win | 0.5851    | 0.7545 | 0.6591   | 1723    |
| Draw     | 0.3403    | 0.1295 | 0.1876   | 1004    |
| Away Win | 0.5345    | 0.5874 | 0.5597   | 1241    |
| Accuracy     | —         | —      | 0.5441   | 3968    |
| Macro Avg    | 0.4866    | 0.4905 | 0.4688   | 3968    |
| Weighted Avg | 0.5073    | 0.5441 | 0.5087   | 3968    |



   
### Confusion Matrices

**Neural Network**
| Actual \ Predicted | Home Win | Draw | Away Win |
| ------------------ | -------- | ---- | -------- |
| **Home Win**       | 1232     | 157  | 334      |
| **Draw**           | 477      | 159  | 368      |
| **Away Win**       | 349      | 157  | 735      |


**XGBoost**
| Actual \ Predicted | Home Win | Draw | Away Win |
| ------------------ | -------- | ---- | -------- |
| **Home Win**       | 1300     | 125  | 298      |
| **Draw**           | 537      | 130  | 337      |
| **Away Win**       | 385      | 127  | 729      |

Luca Occhipinti
