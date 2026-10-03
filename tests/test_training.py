import unittest
import pandas as pd
from train import validate_data

class DataTests(unittest.TestCase):
    def setUp(self): self.df=pd.read_csv('data/smoke.csv',keep_default_na=False)
    def test_valid(self): validate_data(self.df)
    def test_event_leakage(self):
        self.df.loc[6,'event_id']=self.df.loc[0,'event_id']
        with self.assertRaises(ValueError): validate_data(self.df)
    def test_text_leakage(self):
        self.df.loc[6,'text']=self.df.loc[0,'text'].upper()
        with self.assertRaises(ValueError): validate_data(self.df)
    def test_missing_class(self):
        with self.assertRaises(ValueError): validate_data(self.df[self.df.label!='other'])

if __name__ == '__main__': unittest.main()
