from one_hot_encoder import fit_transform
import unittest


class TestFitTransform(unittest.TestCase):
    def test_moscow(self):
        actual = fit_transform(['Moscow'])
        expected = [('Moscow', [1])]
        self.assertEqual(actual, expected)

    def test_moscow_new_york(self):
        actual = fit_transform(['Moscow', 'New York'])
        expected = [
            ('Moscow', [0, 1]),
            ('New York', [1, 0])
        ]
        self.assertEqual(actual, expected)

    def test_space(self):
        actual = fit_transform([''])
        expected = [('', [0])]
        self.assertNotEqual(actual, expected)

    def test_moscow_new_york_moscow(self):
        actual = fit_transform(['Moscow', 'New York', 'Moscow'])
        expected = [
            ('Moscow', [0, 0, 1]),
            ('New York', [0, 1, 0]),
            ('Moscow', [1, 0, 0])
        ]
        self.assertNotEqual(actual, expected)

    def test_moscow_interception_error(self):
        with self.assertRaises(TypeError):
            fit_transform()
