# Autor: Adam Skrodzki
"""OOD transfer matrix: ekspert × docelowa domena. Jak wypada model POZA swoją domeną?

Dla każdej pary (model, cel): pula = val melodie DOMENY CELU (type+meter jak prepare_data),
prompt z nagłówkami celu (scratch) lub z ćwiartką ciała melodii celu (continuation).
PRAWDA = to, o co prosimy (etykiety z melodii celu), nie domena modelu — model jest proszony,
więc nie ma wymówki. Porażka w OOD (śmieci, znaki spoza słownika) = 0, jak w benchmarku
domenowym; pokrycie raportowane osobno.

Metryki w komórce macierzy: score/ref — score = średnie P sędziego przy prawdziwych klasach,
ref = to samo dla PRAWDZIWYCH melodii celu (sufit). hb = home-bias: P sędziego przy KLASIE
DOMOWEJ modelu (meter dla scratch; średnia meter+type dla continuation) — rozróżnia sztywnego
modelu (wysoki hb, ignoruje prompt) od papki (niskie wszystko). * = komórka domenowa (diagonala).

Użycie:
  python src/tools/benchmark_ood.py                       # wszystkie ckpt z domenami
  python src/tools/benchmark_ood.py --models data/models/jig_ckpt.pt --samples 50
  python src/tools/benchmark_ood.py --json > data/benchmarks/ood.json
"""
import argparse, collections, contextlib, glob, io, json, re, sys, os, random, tempfile
TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(TOOL_DIR)
sys.path.insert(0, TOOL_DIR)
sys.path.insert(0, SRC_DIR)
import torch
from core.gpt import GPT
from core.judge import JudgeGPT, HEADS
from core.abc_to_midi import parse_abc, render_score
from train_judge import load_rows, group_split, MIN_BODY
from benchmark_judge import mean_probs


HEADER_PREFIXES = ("X:", "T:", "C:", "M:", "K:", "N:", "%")


def first_tune(text):
    """Return the first ABC tune, stopping before a subsequent ``X:`` line.

    The prompt starts the first tune, but a model can accidentally emit another
    tune header.  Keeping the first tune bounded makes validation and judging
    refer to the same generated object.
    """
    headers = list(re.finditer(r"(?m)^X:", text))
    if len(headers) > 1:
        text = text[:headers[1].start()]
    return text


def tune_body(tune):
    """Remove ABC headers for the judge, preserving the old judge input rules."""
    return "\n".join(line for line in tune.split("\n")
                     if not line.startswith(HEADER_PREFIXES)).strip()


def generate_tune(model, stoi, itos, device, prompt, a):
    """Generate ``(first_tune, body, reason)``.

    ``first_tune`` is retained even when the body is too short so structural
    validation still covers every generation that made it past vocabulary
    checking.  ``body`` and ``reason`` retain the benchmark's old eligibility
    behavior.
    """
    if any(c not in stoi for c in prompt):
        return None, None, "vocab"
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    with torch.no_grad():
        gen = model.generate(idx, a.new, temperature=a.temp, top_k=a.topk)[0].tolist()
    # GPT.generate() returns the original prompt followed by the new tokens.
    # Decode that complete sequence once; prepending prompt here would create a
    # fake second tune header and make first_tune() discard the generated body.
    full = "".join(itos[t] for t in gen)
    tune = first_tune(full)
    body = tune_body(tune)
    if len(body) < MIN_BODY:
        return tune, None, "short_body"
    return tune, body, None

def gen_body(model, stoi, itos, device, prompt, a):
    """Ciało wygenerowanej melodii (bez linii nagłówkowych) albo None, gdy prompt
    niekodowalny / ciało zbyt krótkie (porażka OOD = 0 punktu)."""
    _, body, why = generate_tune(model, stoi, itos, device, prompt, a)
    return body, why


def measure_validation(score, tolerance=1e-6):
    """Count internal measures whose duration differs from their bar duration."""
    result = {"measure_checked": 0, "measure_errors": 0,
              "measure_underfull": 0, "measure_overfull": 0}
    parts = list(getattr(score, "parts", []) or [])
    if not parts:
        parts = [score]
    for part in parts:
        measures = list(part.getElementsByClass("Measure"))
        # Pickups and incomplete endings are legitimate; only internal bars are
        # structural candidates for this check.
        for measure in measures[1:-1]:
            bar_duration = getattr(measure, "barDuration", None)
            if bar_duration is None:
                continue
            bar_length = getattr(bar_duration, "quarterLength", None)
            measure_length = getattr(measure.duration, "quarterLength", None)
            if bar_length is None or measure_length is None:
                continue
            result["measure_checked"] += 1
            difference = float(measure_length) - float(bar_length)
            if difference < -tolerance:
                result["measure_underfull"] += 1
            elif difference > tolerance:
                result["measure_overfull"] += 1
    result["measure_errors"] = (result["measure_underfull"] +
                                 result["measure_overfull"])
    return result


def _failure_reason(exc):
    """Use only a compact, stable reason in benchmark output."""
    return type(exc).__name__


def validate_tune(tune, midi_path, quiet_warnings=False):
    """Strictly parse and independently render one generated ABC tune.

    The sanitized path is deliberately run even when strict parsing fails: it
    measures renderer repairability, not raw ABC validity.
    """
    result = {
        "raw_abc_parse_ok": False,
        "sanitized_midi_ok": False,
        "measure_checked": 0,
        "measure_errors": 0,
        "measure_underfull": 0,
        "measure_overfull": 0,
        "failure_reasons": {"raw_abc_parse": {}, "sanitized_midi": {}},
    }
    warning_output = contextlib.nullcontext()
    if quiet_warnings:
        # music21 emits ABC parser diagnostics directly to stderr rather than
        # through Python's warnings module.  Keep the validation result and
        # compact failure counters, but do not print one line per generated tune.
        warning_output = contextlib.redirect_stderr(io.StringIO())
    with warning_output:
        try:
            score = parse_abc(tune, sanitize_input=False)
            result["raw_abc_parse_ok"] = True
            result.update(measure_validation(score))
        except Exception as exc:
            result["failure_reasons"]["raw_abc_parse"][_failure_reason(exc)] = 1

        try:
            sanitized_score = parse_abc(tune, sanitize_input=True)
            render_score(sanitized_score, midi_path)
            result["sanitized_midi_ok"] = True
        except Exception as exc:
            result["failure_reasons"]["sanitized_midi"][_failure_reason(exc)] = 1
    return result


def _merge_failure_counts(total, current):
    for stage, reasons in current.items():
        for reason, count in reasons.items():
            total[stage][reason] += count


def _validation_summary(records, n):
    """Aggregate per-sample validation records using all requested samples."""
    summary = {
        "raw_abc_valid": sum(r["raw_abc_parse_ok"] for r in records),
        "sanitized_midi_ok": sum(r["sanitized_midi_ok"] for r in records),
        "measure_checked": sum(r["measure_checked"] for r in records),
        "measure_errors": sum(r["measure_errors"] for r in records),
        "measure_underfull": sum(r["measure_underfull"] for r in records),
        "measure_overfull": sum(r["measure_overfull"] for r in records),
    }
    summary["raw_abc_valid_rate"] = summary["raw_abc_valid"] / n
    summary["sanitized_midi_rate"] = summary["sanitized_midi_ok"] / n
    return summary

def run_cell(model, stoi, itos, judge, jcfg, jstoi, classes, device, rows,
             sample_ids, home, task, a):
    """Jedna komórka macierzy (model × cel × zadanie). Zwraca statystyki."""
    truth = {h: {c: i for i, c in enumerate(classes[h])} for h in HEADS}
    heads = HEADS if task == "continuation" else [h for h in HEADS if h != "type"]
    home_heads = [h for h in ("meter", "type") if h in heads and home.get(h) in truth[h]]
    p_sum = {h: 0.0 for h in heads}; hb_sum = {h: 0.0 for h in home_heads}
    score, score_valid, raw_valid = 0.0, 0.0, 0
    invalid = {"vocab": 0, "short_body": 0}
    validation_records = []
    failure_reasons = {
        "raw_abc_parse": collections.Counter(),
        "sanitized_midi": collections.Counter(),
    }
    n = max(len(sample_ids), 1)
    with tempfile.TemporaryDirectory(prefix="benchmark_ood_") as midi_dir:
        for sample_number, i in enumerate(sample_ids):
            row = rows[i]
            prompt = f"X:1\nM:{row['meter']}\nK:{row['mode']}\n"
            if task == "continuation":
                prompt += row["body"][:max(40, len(row["body"]) // 4)]
            tune, body, why = generate_tune(model, stoi, itos, device, prompt, a)
            if tune is None:
                # No model output exists when the prompt itself is outside the
                # vocabulary.  It is still an invalid sample for both rates.
                missing_reason = why or "not_generated"
                validation = {
                    "raw_abc_parse_ok": False,
                    "sanitized_midi_ok": False,
                    "measure_checked": 0,
                    "measure_errors": 0,
                    "measure_underfull": 0,
                    "measure_overfull": 0,
                    "failure_reasons": {
                        "raw_abc_parse": {missing_reason: 1},
                        "sanitized_midi": {missing_reason: 1},
                    },
                }
            else:
                validation = validate_tune(
                    tune, os.path.join(midi_dir, f"sample_{sample_number}.mid"),
                    quiet_warnings=a.quiet_validation_warnings)
            validation_records.append(validation)
            _merge_failure_counts(failure_reasons, validation["failure_reasons"])
            raw_valid += int(validation["raw_abc_parse_ok"])

            if body is None:
                invalid[why] += 1
                continue   # = 0 punktu
            probs = mean_probs(judge, jcfg, jstoi, device, body)
            sample_score = 0.0
            for h in heads:
                t = truth[h][row[h]]
                p_true = float(probs[h][t])
                sample_score += p_true
                p_sum[h] += p_true
                if h in home_heads:
                    hb_sum[h] += float(probs[h][truth[h][home[h]]])
            score += sample_score
            if validation["raw_abc_parse_ok"]:
                score_valid += sample_score
    n = max(len(sample_ids), 1)
    nh = len(heads)
    validation = _validation_summary(validation_records, n)
    return {"score": score / (n * nh),
            "score_valid_abc": (score_valid / (raw_valid * nh)
                                 if raw_valid else None),
            "valid_abc_samples": raw_valid,
            "p": {h: p_sum[h] / n for h in heads},
            "home_bias": {h: hb_sum[h] / n for h in home_heads},
            "invalid": invalid, "coverage": 1 - sum(invalid.values()) / n,
            "validation": validation,
            "validation_failure_reasons": {
                stage: dict(reasons) for stage, reasons in failure_reasons.items()
            }}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="ckpty do macierzy (domyślnie wszystkie *_ckpt.pt z domeną)")
    ap.add_argument("--judge", default="data/models/judge_v2.pt")
    ap.add_argument("--csv", default="data/tunes.csv")
    ap.add_argument("--domains", default="data/models/domains.json")
    ap.add_argument("--samples", type=int, default=100, help="melodii na komórkę")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--new", type=int, default=420)
    ap.add_argument("--temp", type=float, default=0.85)
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--quiet-validation-warnings", "--quiet-validation", "--quiet",
                    dest="quiet_validation_warnings", action="store_true",
                    help="suppress per-tune music21 warnings; keep aggregate summaries")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    print(f"urządzenie: {device} | seed: {a.seed} | temp {a.temp} topk {a.topk} new {a.new}")

    with open(a.domains, encoding="utf-8") as f:
        domains = json.load(f)

    ck = torch.load(a.judge, map_location="cpu", weights_only=False)
    jcfg, chars, classes = ck["config"], ck["chars"], ck["classes"]
    jstoi = {c: j + 1 for j, c in enumerate(chars)}
    judge = JudgeGPT(jcfg, {h: len(classes[h]) for h in HEADS}, pool=ck.get("pool", "mean"))
    judge.load_state_dict(ck["model"]); judge.eval().to(device)

    # cele = wszystkie różne domeny (type, meter); diag = domena modelu
    # (_comment w domains.json to string — bierzemy tylko wpisy słownikowe)
    targets = sorted({(d["type"], d["meter"]) for n, d in domains.items()
                      if isinstance(d, dict) and d.get("type") and not n.startswith("judge")})
    print(f"cele: " + ", ".join(f"{t}:{m}" for t, m in targets))

    print(f"czytam {a.csv} ...")
    rows = load_rows(a.csv)
    _, va = group_split(rows, a.split_seed)

    # pula i próbki per CEL: niezależne od modeli (ten sam seed -> te same melodie
    # w kolumnie dla każdego modelu) + referencja = prawdziwe melodie celu
    pools, refs = {}, {}
    for ttype, tmeter in targets:
        key = f"{ttype}:{tmeter}"
        rng = random.Random(f"{a.seed}|{key}")   # deterministyczny per cel
        pool = [i for i in va
                if ttype in rows[i]["type"].lower()
                and rows[i]["meter"].strip() == tmeter
                and len(rows[i]["body"]) >= 4 * 40]
        ids = rng.sample(pool, min(a.samples, len(pool)))
        pools[key] = ids
        # referencja: sędzia na prawdziwych ciałach celu (nie zależy od modelu)
        s = {h: 0.0 for h in HEADS}
        for i in ids:
            probs = mean_probs(judge, jcfg, jstoi, device, rows[i]["body"])
            for h in HEADS:
                s[h] += float(probs[h][classes[h].index(rows[i][h])])
        refs[key] = sum(s.values()) / (len(ids) * len(HEADS))
        print(f"  cel {key}: pula {len(pool)}, referencja {refs[key]:.3f}")

    models = a.models or sorted(glob.glob("data/models/*_ckpt.pt"))
    cells = {}
    for mp in models:
        name = os.path.basename(mp)
        dom = domains.get(name)
        if dom is None or dom.get("type") is None:
            print(f"pomijam {name}: brak domeny w {a.domains}")
            continue
        home = {"type": dom["type"], "meter": dom["meter"]}
        mck = torch.load(mp, map_location=device, weights_only=False)
        stoi, itos, mcfg = mck["stoi"], mck["itos"], mck["config"]
        model = GPT(mcfg); model.load_state_dict(mck["model"]); model.eval().to(device)
        for ttype, tmeter in targets:
            key = f"{ttype}:{tmeter}"
            diag = (ttype == dom["type"] and tmeter == dom["meter"])
            for task in ("scratch", "continuation"):
                st = run_cell(model, stoi, itos, judge, jcfg, jstoi, classes, device,
                              rows, pools[key], home, task, a)
                cells[(name, key, task)] = st
                if not a.json:
                    print(f"  {name} x {key} [{task}]: score {st['score']:.3f} "
                          f"(hb {st['home_bias']}, pokrycie {st['coverage']:.0%})")
                    validation = st["validation"]
                    total = len(pools[key])
                    conditional = (f"{st['score_valid_abc']:.3f}"
                                   if st["score_valid_abc"] is not None else "—")
                    print(f"    ABC raw: {validation['raw_abc_valid']}/{total} valid")
                    print(f"    MIDI sanitized: {validation['sanitized_midi_ok']}/{total} rendered")
                    print(f"    measures: {validation['measure_errors']} errors / "
                          f"{validation['measure_checked']} checked")
                    print(f"    judge score on valid ABC: {conditional}")
        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    if a.json:
        print(json.dumps({"seed": a.seed, "targets": [f"{t}:{m}" for t, m in targets],
                          "refs": refs, "cells": {f"{n}|{k}|{t}": v
                                                  for (n, k, t), v in cells.items()}},
                         ensure_ascii=False, indent=2))
        return

    # macierze: wiersz = model, kolumna = cel; komórka score/ref hb.. c..% (* = diag)
    for task in ("scratch", "continuation"):
        w = 24
        print(f"\n=== {task} | komórka: score/ref h=home-bias c=pokrycie ===")
        print(f"{'model':22s}" + "".join(f"{k:>{w}}" for k in refs))
        for name in sorted({n for n, _, _ in cells}):
            row = f"{name:22s}"
            for key in refs:
                st = cells.get((name, key, task))
                if st is None:
                    row += "—".rjust(w)
                    continue
                diag = "*" if domains.get(name, {}).get("type") == key.split(":")[0] else ""
                hb = "/".join(f"{v:.2f}" for v in st["home_bias"].values()) or "—"
                cell = f"{st['score']:.2f}/{refs[key]:.2f}h{hb}c{st['coverage']:.0%}{diag}"
                row += cell[:w].rjust(w)
            print(row)
    print("\nref = sędzia na prawdziwych melodiach celu (sufit); hb wysokie = model ignoruje "
          "prompt i wraca do domeny; niskie score i niskie hb = papka. * = komórka domenowa.")

if __name__ == "__main__":
    main()
