"""Scheduler-free tests for the complete submission DAG and artifact collector."""
import argparse
import copy
import subprocess
import tarfile
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import campaign


class CampaignTests(unittest.TestCase):
    def test_full_dag_resources_and_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = campaign.make_plan(tmp, "/fake/data")
            self.assertEqual(len(plan["tasks"]), 25)
            jobs = {}
            for i, task in enumerate(plan["tasks"], 100):
                cmd = campaign.batch_command(Path(tmp) / "manifest.json", task, jobs)
                self.assertNotIn("--mem", cmd)
                self.assertEqual(cmd[cmd.index("--qos") + 1], "normal")
                if task["deps"]:
                    kind = "afterok" if task["gpu"] else "afterany"
                    expected = kind + ":" + ":".join(jobs[d]["job_id"] for d in task["deps"])
                    self.assertEqual(cmd[cmd.index("--dependency") + 1], expected)
                self.assertEqual(cmd[cmd.index("--gres") + 1], "gpu:1" if task["gpu"] else "none")
                jobs[task["name"]] = {"job_id": str(i)}
            bad = copy.deepcopy(plan["tasks"])
            bad[0]["deps"] = ["collect"]
            with self.assertRaises(ValueError):
                campaign.validate_plan(bad)
            bad = copy.deepcopy(plan["tasks"])
            bad[1]["flags"] += ["--limit", "8"]
            with self.assertRaises(ValueError):
                campaign.validate_plan(bad)

    def test_submission_persists_ids_and_refuses_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = campaign.make_plan(tmp, "/fake/data")
            manifest = Path(tmp) / "manifest.json"
            campaign.save_json(plan, manifest)
            args = argparse.Namespace(manifest=str(manifest), test_only=False, continue_submit=False)
            replies = [subprocess.CompletedProcess([], 0, f"{100+i}\n", "") for i in range(25)]
            with patch("campaign.command", side_effect=replies) as mocked:
                campaign.submit(args)
            self.assertEqual(mocked.call_count, 25)
            jobs = campaign.read_json(Path(tmp) / "jobs.json")
            self.assertEqual(len(jobs), 25)
            self.assertEqual(jobs["collect"]["job_id"], "124")
            with self.assertRaises(FileExistsError):
                campaign.submit(args)
            args.continue_submit = True
            with patch("campaign.command") as mocked:
                campaign.submit(args)
                mocked.assert_not_called()

    def test_missing_results_are_not_claimed_complete_and_weights_not_archived(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = campaign.make_plan(tmp, "/fake/data")
            root = Path(tmp)
            campaign.save_json({t["name"]: {"job_id": str(100+i)} for i, t in enumerate(plan["tasks"])},
                               root / "jobs.json")
            (root / "private.pt").write_bytes(b"not a real checkpoint")
            (root / "patient.nii.gz").write_bytes(b"not a real scan")
            snapshots = [{"task": t["name"], "slurm_state": "CANCELLED",
                          "task_state": "not-started"} for t in plan["tasks"]]
            with patch("campaign.snapshot", return_value=snapshots):
                campaign.collect(plan)
            report = campaign.read_json(root / "campaign_results.json")
            self.assertFalse(report["complete"])
            self.assertEqual(len(report["missing_evaluations"]), 7)
            self.assertTrue(report["failed_or_unverified_tasks"])
            with tarfile.open(root / "report_artifacts.tar.gz") as tar:
                names = tar.getnames()
                self.assertIn("RESULTS.md", names)
                self.assertNotIn("private.pt", names)
                self.assertNotIn("patient.nii.gz", names)


if __name__ == "__main__":
    unittest.main()