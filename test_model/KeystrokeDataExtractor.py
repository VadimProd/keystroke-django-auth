import pandas as pd
from collections import defaultdict

class KeystrokeDataExtractor:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.period_columns = [col for col in self.df.columns if col not in ['subject', 'sessionIndex', 'rep']]
        self.subject_data = defaultdict(dict)
        self._extract_data()

    def _extract_data(self):
        for subject in self.df['subject'].unique():
            subject_df = self.df[self.df['subject'] == subject][self.period_columns]
            self.subject_data[subject]['first_15'] = subject_df.head(15).reset_index(drop=True)
            self.subject_data[subject]['remaining'] = subject_df.iloc[15:].reset_index(drop=True)

    def get_subject_data(self, subject):
        return self.subject_data.get(subject, {'first_15': pd.DataFrame(), 'remaining': pd.DataFrame()})