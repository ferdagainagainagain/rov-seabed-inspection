from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "synthesize_report.py"
SPEC = importlib.util.spec_from_file_location("synthesize_report", SCRIPT_PATH)
assert SPEC is not None
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["synthesize_report"] = module
SPEC.loader.exec_module(module)


class SynthesizeReportTests(unittest.TestCase):
    def test_load_frame_reports_normalizes_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame_reports.json"
            path.write_text(json.dumps([_report("a.jpg", 0.0, waste_present=True)]))

            reports = module.load_frame_reports(path)

        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]["waste_status"], "possible")
        self.assertTrue(reports[0]["waste_present"])
        self.assertEqual(reports[0]["rov_equipment_status"], "none")
        self.assertEqual(reports[0]["rov_equipment_type"], "none")

    def test_semantic_signature_uses_expected_fields(self) -> None:
        report = module.normalize_report(_report("a.jpg", 0.0, algae_status="clear"))

        signature = module.semantic_signature(report)

        self.assertIn("clear", signature)
        self.assertIn("sand", signature)

    def test_duplicate_group_keeps_one_representative(self) -> None:
        reports = [
            module.normalize_report(_report("a.jpg", 0.0, uncertainty="high")),
            module.normalize_report(_report("b.jpg", 10.0, uncertainty="low")),
        ]

        selected = module.select_final_frames(reports, merge_window_sec=20.0, max_gap_sec=90.0)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["image_name"], "b.jpg")

    def test_representative_choice_prioritizes_importance_and_uncertainty(self) -> None:
        group = [
            module.normalize_report(_report("low.jpg", 0.0, inspection_importance="low", uncertainty="low")),
            module.normalize_report(_report("high.jpg", 1.0, inspection_importance="high", uncertainty="high")),
        ]

        representative = module.choose_representative(group)

        self.assertEqual(representative["image_name"], "high.jpg")

    def test_possible_status_is_preserved_conservatively(self) -> None:
        report = module.normalize_report(_report("waste.jpg", 0.0, waste_status="possible"))

        self.assertTrue(module.is_important(report))
        self.assertEqual(report["waste_status"], "possible")
        self.assertTrue(report["waste_present"])

    def test_max_gap_keeps_long_uniform_sections(self) -> None:
        reports = [
            module.normalize_report(_report("a.jpg", 0.0)),
            module.normalize_report(_report("b.jpg", 100.0)),
            module.normalize_report(_report("c.jpg", 200.0)),
        ]

        selected = module.select_final_frames(reports, merge_window_sec=200.0, max_gap_sec=90.0)

        self.assertEqual(len(selected), 3)
        self.assertEqual(selected[1]["keep_reason"], "max_gap_representative")

    def test_markdown_generation_includes_sources_and_representatives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            frame_reports = tmp_path / "frame_reports.json"
            frame_reports.write_text("[]")
            reports = [module.normalize_report(_report(str(tmp_path / "a.jpg"), 0.0))]
            reports[0]["final_id"] = 1
            reports[0]["keep_reason"] = "first_frame"
            module.write_final_report(
                frame_reports_path=frame_reports,
                output_dir=tmp_path,
                title="Title",
                all_reports=reports,
                final_reports=reports,
                synthesis="Synthesis text.",
                copied_frames=False,
            )

            text = (tmp_path / "final_report.md").read_text()

        self.assertIn("Frame reports JSON", text)
        self.assertIn("Representative Keyframes", text)
        self.assertIn("a.jpg", text)

    def test_final_keyframes_csv_fields_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final_keyframes.csv"
            reports = [module.normalize_report(_report("a.jpg", 0.0))]
            reports[0]["final_id"] = 1
            reports[0]["keep_reason"] = "first_frame"
            reports[0]["original_image_path"] = reports[0]["image_path"]
            reports[0]["final_image_path"] = ""
            module.write_final_csv(reports, path)

            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["final_id"], "1")
        self.assertIn("waste_status", rows[0])
        self.assertIn("rov_equipment_status", rows[0])
        self.assertIn("rov_equipment_type", rows[0])
        self.assertIn("rov_equipment_present", rows[0])
        self.assertIn("keep_reason", rows[0])

    def test_rov_equipment_does_not_count_as_waste_or_structure(self) -> None:
        report = module.normalize_report(
            _report(
                "tether.jpg",
                0.0,
                rov_equipment_status="clear",
                rov_equipment_type="tether",
            )
        )

        self.assertTrue(report["rov_equipment_present"])
        self.assertFalse(report["waste_present"])
        self.assertFalse(report["structure_present"])
        self.assertFalse(module.is_important(report))

    def test_final_synthesis_mentions_rov_equipment_separately(self) -> None:
        reports = [
            module.normalize_report(
                _report(
                    "tether.jpg",
                    0.0,
                    rov_equipment_status="clear",
                    rov_equipment_type="tether",
                )
            )
        ]

        synthesis = module.summarize_reports(reports, reports)

        self.assertIn("No clear environmental debris was detected.", synthesis)
        self.assertIn("No fixed man-made structures were detected.", synthesis)
        self.assertIn("ROV equipment/tether was clearly visible in 1 frame", synthesis)

    def test_clear_rov_equipment_can_be_kept_with_specific_reason(self) -> None:
        reports = [
            module.normalize_report(_report("a.jpg", 0.0)),
            module.normalize_report(
                _report(
                    "b.jpg",
                    10.0,
                    rov_equipment_status="clear",
                    rov_equipment_type="cable",
                )
            ),
        ]

        selected = module.select_final_frames(reports, merge_window_sec=20.0, max_gap_sec=90.0)

        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[1]["keep_reason"], "rov_equipment_visible")


def _report(
    image_name: str,
    timestamp_sec: float,
    substrate: str = "sand",
    rocks_present: bool = False,
    cobbles_present: bool = False,
    algae_status: str = "none",
    waste_status: str = "none",
    fauna_status: str = "none",
    structure_status: str = "none",
    rov_equipment_status: str = "none",
    rov_equipment_type: str = "none",
    waste_present: bool | None = None,
    inspection_importance: str = "medium",
    uncertainty: str = "medium",
) -> dict:
    report = {
        "image_path": image_name,
        "image_name": Path(image_name).name,
        "timestamp_sec": timestamp_sec,
        "substrate": substrate,
        "rocks_present": rocks_present,
        "cobbles_present": cobbles_present,
        "algae_status": algae_status,
        "waste_status": waste_status,
        "fauna_status": fauna_status,
        "structure_status": structure_status,
        "rov_equipment_status": rov_equipment_status,
        "rov_equipment_type": rov_equipment_type,
        "water_visibility": "medium",
        "inspection_importance": inspection_importance,
        "uncertainty": uncertainty,
        "short_description": "A short description.",
    }
    if waste_present is not None:
        report.pop("waste_status")
        report["waste_present"] = waste_present
    return report


if __name__ == "__main__":
    unittest.main()
