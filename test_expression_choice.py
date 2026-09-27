"""Expression tie-breaking must surface mild happy/sad instead of neutral."""

import unittest

from server import choose_expression, smoothed_expression, _record_expression


class ChooseExpressionTests(unittest.TestCase):
    def test_clear_expressions_keep_their_labels(self):
        happy = choose_expression('happy', {'happy': 78, 'neutral': 12, 'sad': 4})
        self.assertEqual(happy['label'], 'Happy')
        self.assertFalse(happy['tentative'])

    def test_expressive_lead_over_neutral_wins(self):
        result = choose_expression('neutral', {'neutral': 30, 'happy': 44, 'sad': 8})
        self.assertEqual(result['label'], 'Happy')
        self.assertFalse(result['tentative'])

    def test_small_expressive_lead_is_tentative_not_neutral(self):
        result = choose_expression('neutral', {'neutral': 42, 'happy': 36, 'sad': 8})
        self.assertEqual(result['label'], 'Happy')
        self.assertTrue(result['tentative'])

    def test_dominant_neutral_with_no_competing_expression_stays_neutral(self):
        result = choose_expression('neutral', {'neutral': 80, 'happy': 7, 'sad': 8})
        self.assertEqual(result['label'], 'Neutral')

    def test_sad_is_promoted_too(self):
        result = choose_expression('neutral', {'neutral': 35, 'happy': 9, 'sad': 38})
        self.assertEqual(result['label'], 'Sad')


class SmoothedExpressionTests(unittest.TestCase):
    def tearDown(self):
        import server
        server._expression_history.clear()

    def test_recent_tentative_readings_agree_into_a_confident_label(self):
        _record_expression({'label': 'Happy', 'score': 36.0, 'tentative': True})
        _record_expression({'label': 'Happy', 'score': 34.0, 'tentative': True})
        _record_expression({'label': 'Happy', 'score': 38.0, 'tentative': True})
        result = smoothed_expression()
        self.assertEqual(result['label'], 'Happy')
        self.assertFalse(result['tentative'])

    def test_single_recent_reading_is_used_directly(self):
        _record_expression({'label': 'Sad', 'score': 41.0, 'tentative': False})
        result = smoothed_expression()
        self.assertEqual(result['label'], 'Sad')
        self.assertFalse(result['tentative'])


if __name__ == '__main__':
    unittest.main()
