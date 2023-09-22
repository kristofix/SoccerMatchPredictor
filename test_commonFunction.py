import unittest
import pandas as pd
from commonFunction import removeDotFromColumnNames, dropMinutes, sortByDate, dropNotDraw, oddsFilter, dropInsufficient, dif_threshold, dropUnnecessary

class TestPandasFunctions(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'nameWith.Dot' : [1,2,3,4],
            'framestime': [10, 20, 30, 40],
            'frameshomescore': [1, 1, 2, 2],
            'framesawayscore': [1, 2, 2, 3],
            'datetimestamp': [1631809781000, 1631809782000, 1631809783000, 1631809784000],
            'frameshomeodd': [1.1, 1.6, 1.7, 2.0],
            'framesawayodd': [1.1, 1.6, 1.7, 2.0],
            'frameshomeshotsOnTarget': [1, 1, 2, 2],
            'frameshomeshotsOffTarget': [1, 1, 2, 2],
            'frameshomeattacks': [1, 1, 2, 2],
            'frameshomedangerousAttacks': [1, 1, 2, 2],
            'framesawayshotsOnTarget': [1, 1, 2, 2],
            'framesawayshotsOffTarget': [1, 1, 2, 2],
            'framesawayattacks': [1, 1, 2, 2],
            'framesawaydangerousAttacks': [1, 1, 2, 2]
        })

    def test_removeDotFromColumnNames(self):
        result = removeDotFromColumnNames(self.df)
        self.assertIn("nameWithDot", result.columns)

    def test_dropMinutes(self):
        result = dropMinutes(self.df)
        self.assertEqual(len(result), 1)

    def test_sortByDate(self):
        sorted_df = sortByDate(self.df)
        self.assertTrue((sorted_df['datetimestamp'].is_monotonic_increasing))

    def test_oddsFilter(self):
        result = oddsFilter(self.df)
        self.assertNotIn(1.1, result['frameshomeodd'].values)
        self.assertIn(2.0, result['frameshomeodd'].values)

    def test_dropUnnecessary(self):
        result = dropUnnecessary(self.df)
        for col in ['framestime', 'frameshomescore', 'framesawayscore', 'datetimestamp']:
            self.assertNotIn(col, result.columns)

if __name__ == '__main__':
    unittest.main()
