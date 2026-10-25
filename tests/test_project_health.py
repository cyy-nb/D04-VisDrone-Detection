"""Repository-level health checks for syntax and basic CLI availability."""

import ast
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ProjectHealthTests(unittest.TestCase):
    def test_all_python_sources_parse(self):
        failures = []
        for path in ROOT.rglob("*.py"):
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                failures.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
        self.assertEqual(failures, [])

    def test_hardcoded_project_root_is_limited_to_default_config(self):
        offenders = []
        for path in ROOT.rglob("*.py"):
            if any(part in {".git", "__pycache__"} for part in path.parts):
                continue
            if path.name == "d04_common.py":
                continue
            hardcoded_root = "F:" + chr(92) + "D04"
            if hardcoded_root in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(ROOT)))
        self.assertEqual(offenders, [])

    def test_core_cli_help_does_not_require_datasets(self):
        for script in (
            "script/convert_visdrone.py",
            "script/yolo2coco.py",
            "script/D04_filter_roi_small32.py",
        ):
            with self.subTest(script=script):
                result = subprocess.run(
                    [sys.executable, str(ROOT / script), "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())


if __name__ == "__main__":
    unittest.main()
