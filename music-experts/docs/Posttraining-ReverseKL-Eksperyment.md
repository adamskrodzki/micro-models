# Posttraining: reverse KL z rotującym nauczycielem (eksperyment E-RKL)

Status: ZREALIZOWANY (iteracja 1: continuation+scratch, α=0.1, reverse). Wyniki niżej.

## Motywacja

Macierz transferu OOD (`benchmark_ood.py`, log `ood_data_20260904_180711`) pokazuje:

- **scratch**: eksperci są domeny-zablokowani — poza domeną score 0.07–0.15 przy suficie
  0.82/0.50, home-bias 0.89–0.95. Model ignoruje nagłówki `M:`/`K:` i pisze w swoim metrum.
- **continuation**: zakotwiczenie promptem realnym ciałem melodii podwaja/trójkuje score
  poza domeną (reel 0.14→0.34, waltz 0.07→0.24), ale hb spada tylko do ~0.4–0.55 — styl
  domowy dalej przecieka.
- **Skala nie pomaga**: większe/dłużej trenowane ckpty (e128, e192) mają najlepszy wynik
  w domenie i najwyższą sztywność (hb ~0.9–0.97 wszędzie).

Wniosek: eksperci uczą się stylu domowego, ale nie uczą się *słuchać promptu*. Model
zachowuje się źle we własnym rozkładzie generacji — a dokładnie tam działa trening
on-policy.

## Hipoteza

On-policy reverse KL (student‖nauczyciel) na zadaniu continuation, z nauczycielem
kompetentnym w domenie promptu, nauczy jeden checkpoint przełączać styl zależnie od
nagłówków — tj. obniży home-bias poza domeną bez utraty jakości w domenie — tam, gdzie
off-policy SFT na pełnym korpusie tego nie zrobi (eksposure bias: SFT nigdy nie widzi
własnych rolloutów studenta).

## Metoda

- **Start**: jeden wybrany ekspert checkpoint (kandydaci: `jig_e128_s3` — najlepszy
  w domenie i najbardziej sztywny; `jig_e32_s3` — najbardziej elastyczny i najsłabszy;
  decyzja przed startem, wynik może się mocno różnić).
- **Zadanie**: continuation — prompt = nagłówki celu + ćwiartka ciała melodii celu
  (identyczna konstrukcja jak w benchmarku OOD, filtr długości ciała ≥ 4×40 znaków).
- **Pętla treningowa**:
  1. wybierz domenę-nauczyciela (rotacja per batch, równa reprezentacja jig / reel /
     waltz — nie proporcjonalna do wielkości puli; pula waltz = 348, będzie resampling),
  2. wylosuj prompty z puli walidacyjnej tej domeny,
  3. wygeneruj kontynuacje ze studenta (temp 0.85, top-k 18 — jak w benchmarku),
  4. policz per-token reverse KL(student‖nauczyciel) na wygenerowanej sekwencji
     (term nauczyciela stały względem parametrów studenta — brak REINFORCE),
  5. krok optymalizacji.
- **Wygładzenie nauczyciela**: ε-smoothing lub obcięcie top-k rozkładu nauczyciela
  (reverse KL karze masy studenta tam, gdzie nauczyciel ~0 — a nauczyciel to mały,
  szumny model).

## Ramy porównawcze (ten sam checkpoint, ten sam budżet tokenowy)

| Arm | Trening | Rola |
|---|---|---|
| A | mixed SFT na pełnym korpusie, continuation, off-policy CE | kontrola — może zabić pomysł |
| B | on-policy reverse KL, rotacja 3-drożna | pomysł główny |
| C | on-policy forward KL (teacher‖student), ta sama rotacja | izoluje kierunek KL |

Opcjonalny kontekst (nie kontrola): generalista trenowany od zera na całym zbiorze.

## Metryki

- **Macierz OOD** (`benchmark_ood.py`, ta sama konfiguracja co log bazowy): score, ref,
  hb, pokrycie — scratch i continuation.
- **Benchmark domenowy** własnego checkpointu: zapominanie domeny domowej.
- **Trajektoria hb** w trakcie treningu — bezpośredni odczyt uczenia się słuchania promptu.
- **Entropia generacji per domena** — wykrywanie kolapsu dywersyjności wewnątrz domeny
  (rotacja chroni dywersyjność między domenami, ale nie wewnątrz).
- **Sanity-check na start**: średni log-prob nauczyciela na rolloutach studenta dla każdej
  pary (student × nauczyciel). Jeśli nauczyciel waltz daje katastrofalnie niski log-prob
  na śmieciach jig-studenta, gradienty będą zdominowane przez pozycje „unikaj niskiego
  prawdopodobieństwa" zamiast „bądź jak waltz" — przerwać i przemyśleć (np. maskowanie
  pozycji, mocniejsze wygładzenie).

## Kryteria sukcesu / dyskryminatory

Mixed SFT (arm A) najpewniej naprawi łatwą część: hb spadnie, score OOD wzrośnie
(mapowanie nagłówek→domena trywialne przy teacher forcing). On-policy KL musi wygrać na:

1. **Jakość długich generacji** przy temp ~1.0, gdy student jest sam (SFT nie widzi
   własnych rolloutów → degradacja własnej dystrybucji),
2. **Kalibracja score sędziego** na próbkach studenta vs nauczyciela,
3. **Zapominanie** przy równym budżecie.

Jeżeli A dorównuje B we wszystkich trzech — mechanizm on-policy nieuzasadniony.

## Znane ryzyka

- **Mode-seeking reverse KL** → ostrość studenta na argmax nauczyciela; monitorować
  entropię per domena.
- **Nauczyciel nieinformatywny na rolloutach studenta** poza domeną — sanity-check wyżej.
- **Mała pula waltz** (348) → resampling, efektywna repetycja; ewentualna augmentacja.
- **Scratch może się nie poprawić** — trening tylko na continuation; hb w scratch może
  zostać wysokie (brak zakotwiczenia). Ewaluować oba zadania, nie zakładać transferu.
- **Kolaps dywersyjności wewnątrz domeny** mimo rotacji.

---

## Realizacja i odchylenia od planu

Implementacja: `src/train/train_rkl.py`. Kluczowe zmiany względem planu, wynikłe z
analizy błędów i wyników pośrednich:

1. **Wspólny słownik zamiast konwencji na brakujące znaki** — nauczyciele przetrenowani
   z `VOCAB_FROM=universalist_ckpt.pt` (`*_sh_ckpt.pt`; proces: `prepare_data.py` →
   `train_gpt.py` z VOCAB_FROM). Ryzyko zauważone przy okazji: znaki spoza korpusu
   nauczyciela mają losowe logity (weight tying) → stąd α-mixing zamiast ε-floor.
2. **Kontekstowa ocena okien** — pierwsza wersja ciąła sekwencję na okna 128 bez
   zachodzenia; nauczyciel oceniał środki melodii bez prawdziwego poprzedzenia i jego
   rozkłady były rozmyte → student spłaszczał się w stronę uniform (entropia 2.2→3.0
   nats/znak). Poprawka: okna ze stride 96, pierwsze 32 znaki okna to tylko kontekst —
   entropia wróciła do ~1.0 i przestała rosnąć.
3. **Zadanie scratch dodane w iteracji 2** — trening tylko na continuation obniżył
   jakość generacji scratch (reel scratch 0.48→0.26); rotacja zadań (6 kombinacji
   domena×zadanie w cyklu batchy) naprawiła regresję i podniosła wszystkie komórki.
4. **Best-by-KL słabo skorelowany z jakością benchmarkową** — `last` był lepszy na
   komórkach istotnych (scratch) mimo wyższego val KL; oba zapisywane.

## Wyniki (judge_v2, seed 42; ref: jig 0.799 / reel 0.766 / waltz 0.544)

Macierz OOD, komórka = `scratch` / `continuation`:

| model | jig:6/8 | reel:4/4 | waltz:3/4 |
|---|---|---|---|
| nauczyciel jig_sh (diag) | 0.64 / 0.72 | 0.14 / 0.31 | 0.16 / 0.23 |
| nauczyciel reel_sh (diag) | 0.25 / 0.43 | 0.57 / 0.59 | 0.16 / 0.21 |
| nauczyciel waltz_sh (diag) | 0.14 / 0.38 | 0.12 / 0.33 | 0.53 / 0.45 |
| universalist (baseline) | 0.52 / 0.62 | 0.48 / 0.52 | 0.20 / 0.29 |
| RKL continuation-only | 0.50 / 0.63 | 0.26 / 0.56 | 0.37 / 0.43 |
| **RKL cont+scratch (last)** | **0.62 / 0.66** | **0.53 / 0.57** | **0.54 / 0.43** |

(off-diagonal nauczycieli z `ood_teachers_*.log`; checkpoint continuation-only usunięty
przy sprzątaniu — liczby z logu `ood_student_rkl_*.log`)

**Wnioski:**

1. **Second-hand działa**: student po RKL zbliża się do każdego specjalisty na jego
   własnej przekątnej (0.02–0.04), trenowany w fazie RKL tylko na prawdopodobieństwach
   nauczycieli — bez surowych danych ich domen.
2. **Waltz: 0.20→0.54 scratch** — sufit sędziega (0.544), poziom własnego nauczyciela
   (0.53). Najmniejsza domena (pula 348) domknięta.
3. **Pareto-dominacja nad universalistą** w każdej komórce — brak kosztu zapominania;
   off-diagonal hb spadło do 0.06–0.15 (student słucha nagłówków we wszystkich kierunkach).
4. **Scratch musi być w treningu**, żeby nie zregresować (wniosek 3 z realizacji).
5. **Rigidity ekspertów nie przeniosła się na studenta**: nauczyciele poza domeną
   generują w swoim metrum (hb 0.9+), rotacja trzech nauczycieli nauczyła studenta
   przełączania zależnie od nagłówków zamiast jednego sztywnego stylu.
6. Sanitarny wskaźnik w trakcie treningu: tlp (log-prob nauczyciela na rolloutach
   studenta) rosnący + stabilna entropia = zdrowy przebieg; rosnąca entropia przy
   spadającym KL = spłaszczanie (sygnał do przerwania / zmiany α).

**Otwarte:** forward KL (`--direction forward`) — izolacja kierunku KL; off-policy
distillation (nauczyciel na prawdziwych sekwencjach) — izolacja on-policy; sensitivity
α i top-k; odtworzenie checkpointu continuation-only dla pełnej tabeli abacyjnej.
