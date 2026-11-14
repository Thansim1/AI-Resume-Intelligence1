import unittest

from model import (
    compute_match_score, cosine_similarity, preprocess,
    score_label, tf, build_vocabulary, idf, tfidf_vector,
)
from utils import extract_skills_flat, skill_match_rate


class ModelTests(unittest.TestCase):
    def test_identical_text_scores_100(self):
        text = "Python data analysis SQL"
        self.assertEqual(compute_match_score(text, text), 100.0)

    def test_empty_text_scores_zero(self):
        self.assertEqual(compute_match_score("", "Python"), 0.0)

    def test_unrelated_text_has_low_similarity(self):
        self.assertLess(compute_match_score("apple banana fruit", "cloud server network"), 1.0)

    def test_cosine_zero_vector(self):
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 2.0]), 0.0)

    def test_cosine_rejects_different_dimensions(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_technical_tokens(self):
        tokens = preprocess("C C++ C# R Python")
        self.assertIn("c++", tokens)
        self.assertIn("c#", tokens)
        self.assertIn("c", tokens)
        self.assertIn("r", tokens)

    def test_skill_extraction(self):
        skills = extract_skills_flat("Python, C++, machine learning, Power BI and SQL")
        self.assertTrue({"python", "c++", "machine learning", "power bi", "sql"}.issubset(skills))

    def test_skill_match_rate(self):
        self.assertEqual(skill_match_rate({"python", "sql"}, {"python", "sql", "aws"}), 66.7)

    def test_score_labels(self):
        self.assertEqual(score_label(80)[0], "High textual match")
        self.assertEqual(score_label(20)[0], "Low textual match")


if __name__ == "__main__":
    unittest.main()
