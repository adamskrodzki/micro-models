"""Przygotowanie korpusu ABC z thesession.org tunes.csv.
Filtr: typ + metrum z argumentów (domyślnie jigi 6/8). Buduje grające bloki ABC + normalizuje tonację.
Użycie: python src/data/prepare_data.py [typ] [metrum] [wyjście]
  np. python src/data/prepare_data.py waltz 3/4 data/corpus/waltz.abc
"""
import csv, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.abc_corpus import ALLOWED, clean_abc, norm_key

csv.field_size_limit(10**7)

TYPE_KW = sys.argv[1] if len(sys.argv) > 1 else "jig"
METER   = sys.argv[2] if len(sys.argv) > 2 else "6/8"
OUT     = sys.argv[3] if len(sys.argv) > 3 else "data/jigs.abc"

def main():
    rows_out, n_total, n_kept = [], 0, 0
    with open("data/tunes.csv", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            n_total += 1
            if TYPE_KW not in row["type"].lower():
                continue
            if row["meter"].strip() != METER:
                continue
            body = clean_abc(row["abc"])
            if not (40 <= len(body) <= 700):
                continue
            if any(ch not in ALLOWED for ch in body.replace("\n", "")):
                continue
            key = norm_key(row["mode"])
            block = f"X:1\nM:{METER}\nK:{key}\n{body}\n"
            rows_out.append(block)
            n_kept += 1

    text = "\n".join(rows_out)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(text)

    vocab = sorted(set(text))
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"melodii w pliku        : {n_total}")
    print(f"{TYPE_KW} ({METER}) zachowane: {n_kept}")
    print(f"znaki łącznie          : {len(text):,}")
    print(f"słownik ({len(vocab)})        : {''.join(vocab)!r}")
    print("\n--- pierwszy blok ---")
    print(rows_out[0] if rows_out else "BRAK")

if __name__ == "__main__":
    main()
