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

    def test_filters_internal_git_object_reference(self):
        text = "2(f3266222a3a23365efc5acea40f78dad9536abe7"
        self.assertTrue(ai_engine._is_agy_internal_text(text, "D:/tmp/prompt.txt"))

    def test_internal_only_candidates_do_not_become_response(self):
        candidates = ["2(f3266222a3a23365efc5acea40f78dad9536abe7"]
        self.assertEqual(ai_engine._select_agy_response_candidate(candidates), "")

    def test_recovers_quota_error_from_conversation(self):
        error = "RESOURCE_EXHAUSTED (code 429): Individual quota reached."
        recovered = ai_engine._select_agy_conversation_error([
            "The model API is currently overloaded.",
            error,
            '{"reason":"QUOTA_EXHAUSTED"}',
        ])
        self.assertEqual(recovered, error)


if __name__ == "__main__":
    unittest.main()
