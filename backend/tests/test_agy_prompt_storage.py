from pathlib import Path
import tempfile
import unittest

from backend.services import ai_engine


class AgyPromptStorageTests(unittest.TestCase):
    def test_prompt_directory_is_outside_project_uploads(self):
        expected = Path(tempfile.gettempdir()).resolve()
        actual = ai_engine._AGY_PROMPT_DIR.resolve()

        self.assertTrue(actual.is_relative_to(expected))
        self.assertNotIn("uploads", {part.lower() for part in actual.parts})

    def test_long_prompt_file_is_created_and_cleaned_in_temp_directory(self):
        prompt_arg, prompt_file = ai_engine._agy_prompt_arg(
            "가" * (ai_engine._AGY_ARG_PROMPT_LIMIT + 1)
        )
        self.addCleanup(ai_engine._cleanup_prompt_file, prompt_file)

        self.assertIsNotNone(prompt_file)
        self.assertEqual(prompt_file.parent, ai_engine._AGY_PROMPT_DIR.resolve())
        self.assertIn(str(prompt_file), prompt_arg)
        self.assertEqual(
            prompt_file.read_text(encoding="utf-8"),
            "가" * (ai_engine._AGY_ARG_PROMPT_LIMIT + 1),
        )

        ai_engine._cleanup_prompt_file(prompt_file)
        self.assertFalse(prompt_file.exists())


if __name__ == "__main__":
    unittest.main()
