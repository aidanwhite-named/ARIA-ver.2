from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.routers import analyze as analyze_router


class PrepareUploadOrderTests(unittest.TestCase):
    def test_ordered_pdf_paths_prefers_upload_manifest_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            job_dir = Path(temp_dir)
            pdf_dir = job_dir / "pdfs"
            pdf_dir.mkdir()

            for filename in ("US11350048.pdf", "US20170134704A1.pdf", "WO2022058301A1.pdf"):
                (pdf_dir / filename).write_bytes(b"%PDF-1.4")

            (job_dir / "upload_manifest.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {"filename": "WO2022058301A1.pdf"},
                            {"filename": "US20170134704A1.pdf"},
                            {"filename": "US11350048.pdf"},
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            ordered = analyze_router._ordered_pdf_paths(job_dir)

            self.assertEqual(
                [path.name for path in ordered],
                ["WO2022058301A1.pdf", "US20170134704A1.pdf", "US11350048.pdf"],
            )


if __name__ == "__main__":
    unittest.main()
