import os
import contextlib
import io
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))

from music21 import meter, note, stream
import torch

from core.abc_to_midi import parse_abc, to_midi
import benchmark_ood
from benchmark_ood import first_tune, measure_validation, run_cell, validate_tune


VALID_ABC = """X:1
T:Validation test
M:4/4
L:1/4
K:C
CDEF|CDEF|CDEF|CDEF|C|
"""


class AbcValidationTests(unittest.TestCase):
    def test_valid_abc_parses_renders_and_has_no_internal_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = os.path.join(tmp, "valid.mid")
            result = validate_tune(VALID_ABC, midi_path)

            self.assertTrue(result["raw_abc_parse_ok"])
            self.assertTrue(result["sanitized_midi_ok"])
            self.assertEqual(result["measure_errors"], 0)
            self.assertEqual(result["measure_checked"], 3)
            self.assertTrue(os.path.exists(midi_path))
        self.assertFalse(os.path.exists(midi_path))

    def test_sanitization_is_distinct_from_strict_parsing(self):
        malformed = VALID_ABC.replace("CDEF|CDEF|", "^=CDEF|CDEF|")
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = os.path.join(tmp, "repaired.mid")
            result = validate_tune(malformed, midi_path)

            self.assertFalse(result["raw_abc_parse_ok"])
            self.assertTrue(result["sanitized_midi_ok"])
            self.assertTrue(os.path.exists(midi_path))
            self.assertTrue(to_midi(malformed, os.path.join(tmp, "compat.mid")))

    def test_validation_warnings_can_be_suppressed(self):
        malformed = VALID_ABC.replace("CDEF|CDEF|", "X1CDEF|")
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = validate_tune(malformed, os.path.join(tmp, "quiet.mid"),
                                       quiet_warnings=True)

        self.assertTrue(result["raw_abc_parse_ok"])
        self.assertEqual(stderr.getvalue(), "")

    def test_underfull_internal_measure(self):
        abc = VALID_ABC.replace("CDEF|CDEF|", "CDE|CDEF|")
        score = parse_abc(abc)
        result = measure_validation(score)

        self.assertEqual(result["measure_underfull"], 1)
        self.assertEqual(result["measure_overfull"], 0)
        self.assertEqual(result["measure_errors"], 1)

    def test_overfull_internal_measure(self):
        score = stream.Score()
        part = stream.Part()
        score.insert(0, part)
        for number, length in enumerate((4, 5, 4, 1)):
            measure = stream.Measure(number=number)
            if number == 0:
                measure.insert(0, meter.TimeSignature("4/4"))
            measure.append(note.Note("C", quarterLength=length))
            part.append(measure)

        result = measure_validation(score)
        self.assertEqual(result["measure_overfull"], 1)
        self.assertEqual(result["measure_underfull"], 0)
        self.assertEqual(result["measure_errors"], 1)

    def test_pickup_and_incomplete_final_measure_are_excluded(self):
        abc = VALID_ABC.replace("CDEF|CDEF|CDEF|CDEF|C|", "C|CDEF|CDEF|CDEF|C|")
        result = measure_validation(parse_abc(abc))

        self.assertEqual(result["measure_errors"], 0)
        self.assertEqual(result["measure_checked"], 3)

    def test_first_generated_tune_excludes_accidental_second_tune(self):
        second = VALID_ABC.replace("X:1", "X:2").replace("Validation test", "Second")
        generated = VALID_ABC + "\n" + second
        first = first_tune(generated)

        self.assertIn("T:Validation test", first)
        self.assertNotIn("X:2", first)
        self.assertEqual(first.count("X:"), 1)

    def test_benchmark_keeps_all_sample_score_and_adds_conditional_score(self):
        valid_body = "CDEF|" * 55
        repaired_body = "^=CDEF|" * 45
        rows = [{"meter": "4/4", "mode": "C", "type": "reel", "body": "C" * 160},
                {"meter": "4/4", "mode": "C", "type": "reel", "body": "C" * 160}]
        outputs = iter((valid_body, repaired_body))

        class FakeModel:
            def generate(self, idx, new, temperature, top_k):
                body = next(outputs)
                return torch.tensor([[stoi[ch] for ch in prompt_chars + body]],
                                    dtype=torch.long)

        prompt_chars = "X:1\nM:4/4\nK:C\n"
        chars = sorted(set(prompt_chars + valid_body + repaired_body))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = chars
        classes = {"meter": ["4/4"], "mode": ["C"], "type": ["reel"]}

        def fake_mean_probs(_judge, _jcfg, _jstoi, _device, body):
            value = 0.8 if "^=" not in body else 0.2
            return {head: torch.tensor([value]) for head in ("meter", "mode", "type")}

        args = SimpleNamespace(new=420, temp=0.85, topk=18,
                               quiet_validation_warnings=False)
        with mock.patch.object(benchmark_ood, "mean_probs", fake_mean_probs):
            result = run_cell(FakeModel(), stoi, itos, None, None, None, classes,
                              "cpu", rows, [0, 1], {"meter": "4/4", "type": "reel"},
                              "scratch", args)

        self.assertAlmostEqual(result["score"], 0.5)
        self.assertAlmostEqual(result["score_valid_abc"], 0.8)
        self.assertEqual(result["valid_abc_samples"], 1)
        self.assertEqual(result["validation"]["raw_abc_valid"], 1)
        self.assertEqual(result["validation"]["sanitized_midi_ok"], 2)
        self.assertEqual(result["invalid"], {"vocab": 0, "short_body": 0})
        self.assertEqual(result["coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
