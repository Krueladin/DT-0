import unittest
import main

class DataSetFixture(object):
    def __init__(self):
        self.emptyset = []
        
        self.heterotargetset = [0, 1, 0, 1, 0, 1]
        self.heterocolset = ['y', 'y', 'y', 'n', 'n', 'n']
        
        self.homotargetset = [1, 1, 1, 1, 1, 1]
        self.homocolset = ['y', 'y', 'y', 'y', 'y', 'y']

class TestCalcEntropy(unittest.TestCase):
    def test_homo_data(self):
        """Calculate Entropy of a homogenous data set."""
        ds = DataSetFixture()
        self.assertEqual(main.calc_entropy(ds.homotargetset), 0.0)

    def test_hetero_data(self):
        """Entropy of a heterogenous data set.
        
        Dataset is evenly divided between two classes.
        """
        ds = DataSetFixture()
        self.assertEqual(main.calc_entropy(ds.heterotargetset), 1.0)
    
    def test_no_data(self):
        """Entropy of an empty data set is 0.0."""
        ds = DataSetFixture()
        self.assertEqual(main.calc_entropy(ds.emptyset), 0.0)

class TestCalcInfoGain(unittest.TestCase):
    
    def test_homo_data_1_class_target(self):
        """Information Gain of a target and dataset column that are both homogenous"""
        ds = DataSetFixture()
        self.assertEqual(main.calc_info_gain(ds.homocolset, ds.homotargetset), 0.0)
    
    def test_hetero_data_1_class_target(self):
        """Information Gain of a homogenous target class column and a 
        dataset column that has two evenly divided, unique labels.
        """
        ds = DataSetFixture()
        self.assertEqual(main.calc_info_gain(ds.heterocolset, ds.homotargetset), 0.0)

    def test_homo_data_2_class_target(self):
        """Information Gain of a two class target and a homogenous data column."""
        ds = DataSetFixture()
        self.assertEqual(main.calc_info_gain(ds.homocolset, ds.heterotargetset), 0.0)
    
    def test_hetero_data_2_class_target(self):
        """Information Gain of a two class target and a two class data column.
        
        Verified by hand:
        IG = 1 - ( (3/6)*(-(2/3)log2(2/3) - (1/3)log2(1/3)) 
                    - (3/6)*((2/3)log2(2/3) - (1/3)log2(1/3)) )
           = ~0.082
        """
        ds = DataSetFixture()
        self.assertAlmostEqual(main.calc_info_gain(ds.heterocolset, ds.heterotargetset), 0.082, 3)

    def test_no_data(self):
        """Information Gain of empty data sets."""
        ds = DataSetFixture()
        self.assertEqual(main.calc_info_gain(ds.emptyset, ds.emptyset), 0.0)

    def test_uneven_sets(self):
        """Information gain of length matching sets throws exception."""
        ds = DataSetFixture()
        self.assertRaises(Exception, main.calc_info_gain, ds.emptyset, ds.homotargetset)

if __name__ == "__main__":
    unittest.main()
