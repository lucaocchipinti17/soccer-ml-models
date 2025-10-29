from typing import List, Dict, Any, Iterable
import pandas as pd
from understatapi import UnderstatClient
from data.clean import normalize
from pathlib import Path

def _f(x):
    try:
        return None if x is None else float(x)
    except:
        return None

class DataScraper:
    def __init__(self, leagues: Iterable[str], seasons: Iterable[str], output: str = 'normalized_data.csv'):
        self.leagues = list(leagues)
        self.seasons = list(seasons)
        self.us = UnderstatClient()
        self.df = pd.DataFrame()
        self.output = output

    def _fetch_raw_matches(self, league: str, season: str) -> List[Dict[str, Any]]:
        return self.us.league(league=league).get_match_data(season=season)
    
    def fetch_all(self):
        for league in self.leagues:
            for season in self.seasons:
                raw = self._fetch_raw_matches(league, season)
                norm = normalize(raw)
                self.df = pd.concat([self.df, norm], ignore_index=True)
        self.df.sort_values('datetime', inplace=True)
        self.df.to_csv(f'data/{self.output}', index=False)
        return self.df
    
    def get_data(self) -> pd.DataFrame:
        path = Path(f'data/{self.output}')
        if path.exists():
            self.df = pd.read_csv(path)
        else:
            self.fetch_all()
        return self.df.copy()
    
    def prepare_for_training(self, split: float = 0.8):
        from data.train_prep import prepare_data
        return prepare_data(self.df, split=split)
    
    
    