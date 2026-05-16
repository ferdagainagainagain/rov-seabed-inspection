from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_keyframes_vlm.py"
SPEC = importlib.util.spec_from_file_location("analyze_keyframes_vlm", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["analyze_keyframes_vlm"] = module
SPEC.loader.exec_module(module)


class AnalyzeKeyframesVlmTests(unittest.TestCase):
    def test_extract_json_object_from_wrapped_text(self) -> None:
        text = 'Here is JSON:\n{"substrate":"rocks","rocks_present":true}\nDone.'

        parsed = module.extract_json_object(text)

        self.assertEqual(parsed["substrate"], "rocks")
        self.assertTrue(parsed["rocks_present"])
        self.assertFalse(parsed["waste_present"])
        self.assertEqual(parsed["waste_status"], "none")
        self.assertEqual(parsed["rov_equipment_status"], "none")
        self.assertEqual(parsed["rov_equipment_type"], "none")
        self.assertFalse(parsed["rov_equipment_present"])
        self.assertEqual(parsed["water_visibility"], "unclear")

    def test_invalid_json_uses_safe_defaults(self) -> None:
        parsed = module.extract_json_object("not json")

        self.assertEqual(parsed["substrate"], "unclear")
        self.assertFalse(parsed["algae_present"])
        self.assertEqual(parsed["algae_status"], "none")
        self.assertEqual(parsed["waste_status"], "none")
        self.assertEqual(parsed["fauna_status"], "none")
        self.assertEqual(parsed["structure_status"], "none")
        self.assertEqual(parsed["rov_equipment_status"], "none")
        self.assertEqual(parsed["rov_equipment_type"], "none")
        self.assertEqual(parsed["inspection_importance"], "medium")
        self.assertEqual(parsed["uncertainty"], "high")

    def test_parse_timestamp_from_filename(self) -> None:
        self.assertEqual(module.parse_timestamp_sec("frame_0001_t00012.0.jpg"), 12.0)
        self.assertEqual(module.parse_timestamp_sec("frame_0002_t12.5.png"), 12.5)
        self.assertIsNone(module.parse_timestamp_sec("frame_0003.jpg"))

    def test_model_name_argument_is_accepted(self) -> None:
        argv = [
            "analyze_keyframes_vlm.py",
            "--images-dir",
            "images",
            "--output-dir",
            "reports",
            "--model-name",
            "mlx-community/Qwen3-VL-4B-Instruct-4bit",
        ]

        with patch.object(sys, "argv", argv):
            args = module.parse_args()

        self.assertEqual(args.model_name, "mlx-community/Qwen3-VL-4B-Instruct-4bit")

    def test_default_model_is_qwen3(self) -> None:
        argv = [
            "analyze_keyframes_vlm.py",
            "--images-dir",
            "images",
            "--output-dir",
            "reports",
        ]

        with patch.object(sys, "argv", argv):
            args = module.parse_args()

        self.assertEqual(args.model_name, "mlx-community/Qwen3-VL-4B-Instruct-4bit")

    def test_prompt_version_argument_is_accepted(self) -> None:
        argv = [
            "analyze_keyframes_vlm.py",
            "--images-dir",
            "images",
            "--output-dir",
            "reports",
            "--prompt-version",
            "v3",
        ]

        with patch.object(sys, "argv", argv):
            args = module.parse_args()

        self.assertEqual(args.prompt_version, "v3")

    def test_default_prompt_version_is_v3(self) -> None:
        argv = [
            "analyze_keyframes_vlm.py",
            "--images-dir",
            "images",
            "--output-dir",
            "reports",
        ]

        with patch.object(sys, "argv", argv):
            args = module.parse_args()

        self.assertEqual(args.prompt_version, "v3")

    def test_prompt_version_choices_are_v3_only(self) -> None:
        parser = module.build_parser()
        prompt_action = next(action for action in parser._actions if "--prompt-version" in action.option_strings)

        self.assertEqual(prompt_action.choices, ["v3"])

    def test_v2_prompt_contains_new_instructions(self) -> None:
        prompt = module.PROMPTS["v2"]

        self.assertIn('substrate = "mixed"', prompt)
        self.assertIn("low uncertainty", prompt)
        self.assertIn("possible debris", prompt)
        self.assertIn("Do not guess species", prompt)
        self.assertIn("Do not invent species or objects", prompt)

    def test_v3_prompt_contains_status_schema(self) -> None:
        prompt = module.PROMPTS["v3"]

        self.assertIn('"algae_status": "none | possible | clear"', prompt)
        self.assertIn('"waste_status": "none | possible | clear"', prompt)
        self.assertIn('"fauna_status": "none | possible | clear"', prompt)
        self.assertIn('"structure_status": "none | possible | clear"', prompt)
        self.assertIn('"rov_equipment_status": "none | possible | clear"', prompt)
        self.assertIn('"rov_equipment_type": "none | tether | cable | robot_part | other"', prompt)
        self.assertIn("Do not count ROV equipment as environmental waste", prompt)
        self.assertIn("Do not count ROV equipment as a man-made structure", prompt)
        self.assertIn('substrate = "mixed"', prompt)
        self.assertIn('prefer "possible" instead of "clear"', prompt)

    def test_status_normalization_works(self) -> None:
        parsed = module.normalize_annotation(
            {
                "algae_status": "clear",
                "waste_status": "maybe",
                "fauna_status": "none",
                "structure_status": "false",
                "rov_equipment_status": "clear",
                "rov_equipment_type": "tether",
            }
        )

        self.assertEqual(parsed["algae_status"], "clear")
        self.assertEqual(parsed["waste_status"], "possible")
        self.assertEqual(parsed["fauna_status"], "none")
        self.assertEqual(parsed["structure_status"], "none")
        self.assertEqual(parsed["rov_equipment_status"], "clear")
        self.assertEqual(parsed["rov_equipment_type"], "tether")
        self.assertTrue(parsed["algae_present"])
        self.assertTrue(parsed["waste_present"])
        self.assertFalse(parsed["fauna_present"])
        self.assertFalse(parsed["structure_present"])
        self.assertTrue(parsed["rov_equipment_present"])

    def test_old_boolean_outputs_are_still_parsed(self) -> None:
        parsed = module.extract_json_object(
            """
            {
              "waste_present": true,
              "fauna_present": false,
              "structure_present": true,
              "algae_present": true
            }
            """
        )

        self.assertEqual(parsed["waste_status"], "possible")
        self.assertEqual(parsed["fauna_status"], "none")
        self.assertEqual(parsed["structure_status"], "possible")
        self.assertEqual(parsed["algae_status"], "possible")
        self.assertTrue(parsed["waste_present"])
        self.assertFalse(parsed["fauna_present"])
        self.assertTrue(parsed["structure_present"])

    def test_default_record_includes_model_name(self) -> None:
        record = module.default_report_record(
            Path("frame_0001_t00012.0.jpg"),
            raw_model_output="{}",
            model_name="mlx-community/Qwen3-VL-4B-Instruct-4bit",
        )

        self.assertEqual(record["model_name"], "mlx-community/Qwen3-VL-4B-Instruct-4bit")
        self.assertEqual(record["algae_status"], "none")
        self.assertEqual(record["waste_status"], "none")
        self.assertEqual(record["fauna_status"], "none")
        self.assertEqual(record["structure_status"], "none")
        self.assertEqual(record["rov_equipment_status"], "none")
        self.assertEqual(record["rov_equipment_type"], "none")
        self.assertFalse(record["rov_equipment_present"])


if __name__ == "__main__":
    unittest.main()
