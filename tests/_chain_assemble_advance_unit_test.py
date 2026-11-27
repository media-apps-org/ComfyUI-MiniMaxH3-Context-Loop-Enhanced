#!/usr/bin/env python3
"""Standalone envelope tests for Advanced H3 multitrack assembly."""

import importlib.util
import math
import pathlib
import sys
import types
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = "h3_chain_assemble_advance_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package

core = types.ModuleType(PACKAGE + ".chain_nodes")
core.MiniMaxH3ChainAssemble = type("MiniMaxH3ChainAssemble", (), {})
core.np = np
core.math = math
core._resolved_scene_audio_policy = lambda plan, shot: {
    "source_audio_target": shot.get(
        "source_audio_target",
        plan.get("compatibility", {}).get("source_audio_target", "off"))
}
sys.modules[core.__name__] = core

spec = importlib.util.spec_from_file_location(
    PACKAGE + ".chain_assemble_advance", ROOT / "chain_assemble_advance.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class AdvancedAudioEnvelopeTests(unittest.TestCase):
    def test_no_overlay_preserves_source_and_silences_generated(self):
        source, generated = module._build_scene_envelopes(
            100, 100, [], 1.0, 0.8, 0.25, 0.1, 0.1)
        np.testing.assert_allclose(source, 1.0)
        np.testing.assert_allclose(generated, 0.0)

    def test_scene_ducks_source_and_fades_generated(self):
        source, generated = module._build_scene_envelopes(
            100, 100, [(20, 80)], 1.0, 0.75, 0.2, 0.1, 0.1)
        self.assertAlmostEqual(float(source[20]), 1.0)
        self.assertAlmostEqual(float(source[30]), 0.2, places=5)
        self.assertAlmostEqual(float(source[50]), 0.2, places=5)
        self.assertAlmostEqual(float(source[70]), 0.2, places=5)
        self.assertAlmostEqual(float(source[79]), 1.0, places=5)
        self.assertAlmostEqual(float(generated[20]), 0.0)
        self.assertAlmostEqual(float(generated[30]), 0.75, places=5)
        self.assertAlmostEqual(float(generated[50]), 0.75, places=5)
        self.assertAlmostEqual(float(generated[79]), 0.0, places=5)

    def test_overlapping_ranges_keep_duck_and_overlay(self):
        source, generated = module._build_scene_envelopes(
            100, 100, [(10, 50), (40, 80)], 1.0, 1.0, 0.3, 0.1, 0.1)
        self.assertTrue(np.all(source[20:70] <= 0.3))
        self.assertTrue(np.all(generated[40:50] > 0.0))

    def test_auto_selection_respects_scene_lock_and_local_lipsync(self):
        manifest = {
            "compatibility": {"source_audio_target": "locked"},
            "segments": [
                {"index": 1},
                {"index": 2, "source_audio_target": "off"},
                {"index": 3, "source_audio_target": "locked"},
                {"index": 4, "source_audio_target": "off",
                 "lip_sync_source_asset": {"asset_id": "song-vocal"}},
            ],
        }
        records = [{"kind": "scene", "scene": index}
                   for index in range(1, 5)]
        selected = module.MiniMaxH3ChainAssembleAdvance._scene_selection(
            "auto", manifest, records)
        self.assertEqual(selected, {2})

    def test_explicit_scene_selection_is_validated(self):
        manifest = {"segments": [{"index": 1}, {"index": 2}]}
        records = [{"kind": "scene", "scene": 1},
                   {"kind": "scene", "scene": 2}]
        self.assertEqual(
            module.MiniMaxH3ChainAssembleAdvance._scene_selection(
                "2,1", manifest, records), {1, 2})
        with self.assertRaises(ValueError):
            module.MiniMaxH3ChainAssembleAdvance._scene_selection(
                "3", manifest, records)

    def test_invalid_gain_is_rejected(self):
        with self.assertRaises(ValueError):
            module._build_scene_envelopes(
                10, 10, [], -0.1, 1.0, 0.2, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
