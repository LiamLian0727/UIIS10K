"""Checkpoint transformations required by the teacher/student release flow."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools import convert_teacher_checkpoint, export_checkpoint


class CheckpointToolTests(unittest.TestCase):
    def test_teacher_conversion_moves_backbone_and_keeps_shared_weights(self):
        backbone = torch.tensor([1.0])
        head = torch.tensor([2.0])
        converted = convert_teacher_checkpoint.convert_state_dict(
            {
                "backbone.vision_encoder.layer": backbone,
                "roi_head.mask_head.weight": head,
            }
        )

        self.assertEqual(
            set(converted),
            {"teacher.vision_encoder.layer", "roi_head.mask_head.weight"},
        )
        self.assertIs(converted["teacher.vision_encoder.layer"], backbone)
        self.assertIs(converted["roi_head.mask_head.weight"], head)

    def test_teacher_conversion_rejects_non_teacher_or_colliding_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "no backbone"):
            convert_teacher_checkpoint.convert_state_dict(
                {"roi_head.weight": torch.tensor(1)}
            )
        with self.assertRaisesRegex(ValueError, "Duplicate key"):
            convert_teacher_checkpoint.convert_state_dict(
                {
                    "backbone.layer": torch.tensor(1),
                    "teacher.layer": torch.tensor(2),
                }
            )

    def test_student_export_removes_distillation_modules(self):
        backbone = torch.tensor([1.0])
        exported = export_checkpoint.student_state_dict(
            {
                "backbone.layer": backbone,
                "roi_head.weight": torch.tensor([2.0]),
                "teacher.layer": torch.tensor([3.0]),
                "distiller.connector.weight": torch.tensor([4.0]),
            }
        )

        self.assertEqual(set(exported), {"backbone.layer", "roi_head.weight"})
        self.assertIs(exported["backbone.layer"], backbone)

    def test_cli_outputs_only_model_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trained_teacher = root / "trained_teacher.pth"
            prepared_teacher = root / "prepared_teacher.pth"
            trained_student = root / "trained_student.pth"
            exported_student = root / "exported_student.pth"

            torch.save(
                {
                    "state_dict": {
                        "backbone.layer": torch.tensor([1.0]),
                        "roi_head.weight": torch.tensor([2.0]),
                    },
                    "optimizer": {"unneeded": True},
                    "meta": {"private_path": "/some/local/path"},
                },
                trained_teacher,
            )
            with patch.object(
                sys,
                "argv",
                [
                    "convert_teacher_checkpoint.py",
                    str(trained_teacher),
                    str(prepared_teacher),
                ],
            ):
                convert_teacher_checkpoint.main()
            converted = torch.load(
                prepared_teacher, map_location="cpu", weights_only=True
            )
            self.assertEqual(set(converted), {"state_dict"})
            self.assertEqual(
                set(converted["state_dict"]), {"teacher.layer", "roi_head.weight"}
            )

            torch.save(
                {
                    "state_dict": {
                        "backbone.layer": torch.tensor([3.0]),
                        "teacher.layer": torch.tensor([1.0]),
                        "distiller.layer": torch.tensor([4.0]),
                        "roi_head.weight": torch.tensor([2.0]),
                    },
                    "optimizer": {"unneeded": True},
                },
                trained_student,
            )
            with patch.object(
                sys,
                "argv",
                ["export_checkpoint.py", str(trained_student), str(exported_student)],
            ):
                export_checkpoint.main()
            exported = torch.load(
                exported_student, map_location="cpu", weights_only=True
            )
            self.assertEqual(set(exported), {"state_dict"})
            self.assertEqual(
                set(exported["state_dict"]), {"backbone.layer", "roi_head.weight"}
            )


if __name__ == "__main__":
    unittest.main()
