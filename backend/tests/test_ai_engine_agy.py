from __future__ import annotations

import unittest

from backend.services import ai_engine


class AgyResponseFilterTests(unittest.TestCase):
    def test_filters_prompt_file_meta_response(self):
        text = "I will read the prompt file to understand the task."
        self.assertTrue(ai_engine._is_agy_internal_text(text, "D:/tmp/prompt.txt"))

    def test_allows_normal_model_response(self):
        text = "[1] This paragraph discloses the claimed feature."
        self.assertFalse(ai_engine._is_agy_internal_text(text, "D:/tmp/prompt.txt"))


if __name__ == "__main__":
    unittest.main()
