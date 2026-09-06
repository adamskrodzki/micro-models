---
type: nauka-log
title: "Posttraining: reverse KL z rotującym nauczycielem "
status: ZREALIZOWANY
data: 2026-09-05
created_at: 2026-09-05
author: Adam Skrodzki
tags: [nauka, llm, muzyka, capstone, gpt, ngram, realizacja]
repo_github: "https://github.com/adamskrodzki/micro-models"
---

# Posttraining: reverse KL z rotującym nauczycielem (eksperyment E-RKL)

Status: ZREALIZOWANY (iteracja 2: continuation+scratch, α=0.1, reverse) oraz
ablacja kierunku forward KL. Wyniki główne i walidacja niżej.

## Motywacja

Macierz transferu OOD (`benchmark_ood.py`, log `ood_data_20260904_180711`) pokazuje:

- **scratch**: eksperci są trenowani wyłącznie na jednym stylu, poza domeną wynik 0.07–0.15 
  przy suficie 0.82/0.50, home-bias 0.89–0.95. Model ignoruje nagłówki 
  `M:`/`K:` i pisze w swoim metrum.
- **continuation**: zakotwiczenie promptem realnym ciałem melodii podwaja/ potraja wynik
  poza domeną (reel 0.14→0.34, waltz 0.07→0.24), ale hb (home bias) spada tylko do ~0.4–0.55 — styl
  domowy dalej przecieka.
- **Skala nie pomaga**: większe/dłużej trenowane ckpty (e128, e192) mają najlepszy wynik
  w domenie i najwyższą sztywność (hb ~0.9–0.97 wszędzie).

Wniosek: eksperci uczą się stylu domowego, ale nie uczą się *słuchać promptu*. Model
zachowuje się źle we własnym rozkładzie generacji — a dokładnie tam działa trening
on-policy.

## Hipoteza

On-policy reverse KL (student‖nauczyciel) z nauczycielem kompetentnym w domenie promptu
nauczy jeden checkpoint przełączać styl zależnie od nagłówków — obniży home-bias poza
domeną bez utraty jakości w domenie. W trakcie pracy pytanie zostało zaostrzone do:
**ile da się nauczyć „z drugiej ręki" — przez same prawdopodobieństwa nauczycieli, bez
widzenia surowych danych w fazie posttrainingu?** Metryka:
luka między studentem a universalistą (który te dane widział) w każdej komórce macierzy.

## Metoda (stan po realizacji)

- **Student**: `universalist_ckpt.pt` — GPT trenowany od zera na całym korpusie mieszanym
  (`prepare_data.py --mixed`: 47k melodii, 10.5M znaków, 54-znakowy słownik, block 128,
  8000 iteracji ≈ 3 epoki). Początkowo tylko baseline „pierwszej ręki"; w toku pracy
  stał się naturalnym punktem startowym posttrainingu (zastąpił pomysł startu z
  eksperta — patrz Przebieg, krok 0).
- **Nauczyciele**: trzej specjaliści przetrenowani na WSPÓLNYM słowniku
  (`VOCAB_FROM=universalist_ckpt.pt`): `jig_sh` (jig 6/8), `reel_sh` (reel 4/4),
  `waltz_sh` (waltz 3/4), każdy 2000 iteracji na własnej domenie.
- **Zadania**: continuation (nagłówki + prefiks ciała o STAŁEJ długości 96 znaków — równe
  długości promptów umożliwiają batchową generację) i scratch (same nagłówki). 
  Rotacja per batch: domena (jig→reel→waltz) × zadanie — 6 kombinacji w cyklu.
- **Pule promptów**: TRENINGOWE (split treningowy; val zostaje czysta dla ewaluacji),
  filtr jak w benchmarku OOD: type substring + meter równość + ciało ≥ 160 znaków.
- **Rollout i loss**: student generuje 420 znaków (temp 0.85, top-k 18 — jak
  w benchmarku); per-token KL liczony TYLKO na wygenerowanych pozycjach, w oknach 128
  ze stride 96, gdzie pierwsze 32 znaki okna to tylko KONTEKST (nauczyciel i student
  przewidują z prawdziwego prefixu, nie z amputowanego środka melodii).
- **α-mixing**: `p_mix = (1−α)·p_teacher + α/|V|`, gdzie |V| = rozmiar wspólnego
  słownika (54 znaki), α = 0.1. Czyli: rozkład nauczyciela zmiksowany w 10% z rozkładem
  JEDNOSTAJNYM (α/|V| = 0.1/54 ≈ 0.0019 masy na każdy znak). Dwa efekty:
  (a) **podłoga prawdopodobieństwa** — żaden znak nie ma w p_mix mniej niż α/|V|, więc
  log p_mix jest ograniczony od dołu (≈ −6.3 nats); gradient KL nigdy nie eksploduje
  na znakach, których nauczyciel nie zna (losowe logity, patrz Przebieg krok 1) ani tam,
  gdzie jest pewny, a student generuje nie-nauczycielskie treści;
  (b) **lekkie ściągnięcie studenta ku uniform** — kosztem: zbyt duże α spłaszcza
  studenta (dlatego entropia jest metryką kontrolną, a sygnałem alarmowym jej wzrost
  przy spadającym KL). Zastępuje ε-smoothing z pierwotnego planu — przy wspólnym
  słowniku to to samo działanie zapisane inaczej.
- **Optymalizacja**: AdamW lr 1e-4 (fine-tune), warmup 100 + cosine, clip 1.0, bf16;
  4000 iteracji, batch 16. Ewaluacja co 200 iteracji: KL / tlp / entropia per
  domena×zadanie na utwalonej puli val; zapisywane checkpointy best (min val KL) i last.

## Ramy porównawcze

| Arm | Trening | Status |
|---|---|---|
| baseline | universalista: SFT od zera na całym korpusie | zrobiony — punkt odniesienia „pierwszej ręki" i punkt startowy studenta |
| B | on-policy reverse KL, rotacja domena×zadanie | ZREALIZOWANY — wyniki niżej |
| A | mixed SFT fine-tune, off-policy CE | ODRZUCONY jako kontrola: widzi surowe dane domen, więc nie odpowiada na pytanie second-hand; jego rolę pełni baseline universalisty |
| A′ | off-policy distillation: KL α-miksowanego nauczyciela na PRAWDZIWYCH sekwencjach (teacher forcing) | zaproponowany — izoluje on-policy przy identycznym zestawie informacji |
| C | on-policy forward KL (`--direction forward`), ta sama rotacja | ZREALIZOWANY — ablacja niżej |

## Metryki

- **Macierz OOD** (`benchmark_ood.py`, ta sama konfiguracja co log bazowy): score, ref,
  hb, pokrycie — scratch i continuation.
- **Benchmark domenowy** własnego checkpointu: zapominanie domeny domowej.
- **Trajektoria hb** w trakcie treningu — bezpośredni odczyt uczenia się słuchania promptu.
- **Entropia generacji per domena** — wykrywanie kolapsu różnorodności wewnątrz domeny
  (rotacja chroni różnorodności między domenami, ale nie wewnątrz).
- **Sanity-check na start**: średni log-prob nauczyciela na rolloutach studenta dla każdej
  pary (student × nauczyciel). Jeśli nauczyciel waltz daje katastrofalnie niski log-prob
  na śmieciach jig-studenta, gradienty będą zdominowane przez pozycje „unikaj niskiego
  prawdopodobieństwa" zamiast „bądź jak waltz" — przerwać i przemyśleć (np. maskowanie
  pozycji, mocniejsze wygładzenie).

## Kryteria sukcesu / dyskryminatory

Po zmianie punktu startowego na universalistę kryteria przeformułowane — porównanie
z ekspertem-startem straciło sens, liczy się luka do universalisty (first-hand) i do
nauczycieli (specialist ceiling). RKL musiało:

1. **Podnieść komórki poza domeną** — zwłaszcza waltz (najsłabsza: 0.20/0.29),
2. **Nie pogorszyć jig** (zapominanie domeny największej jakościowo),
3. **Obniżyć home bias poza domeną** (słuchanie nagłówków).

Wszystkie trzy były spełnione w pierwotnej ocenie iteracji 2 — patrz Wyniki i Wnioski.
Późniejszy rerun z walidacją ABC/MIDI pokazuje dodatkowo koszt w syntaktycznej
poprawności ABC; interpretacja tego trade-offu znajduje się w sekcji rebenchmarku.

## Znane ryzyka

- **Mode-seeking reverse KL** → wrażliwość studenta na argmax nauczyciela; monitorować
  entropię per domena.
- **Nauczyciel nieinformatywny na rolloutach studenta** poza domeną — sanity-check wyżej.
- **Mała pula waltz** (348) → resampling, efektywna repetycja; ewentualna augmentacja.
- **Scratch może się nie poprawić** — trening tylko na continuation; hb w scratch może
  zostać wysokie (brak zakotwiczenia). Ewaluować oba zadania, nie zakładać transferu.
- **Kolaps dywersyjności wewnątrz domeny** mimo rotacji.

---

## Przebieg (chronologicznie)

Implementacja: `src/train/train_rkl.py`. Kroki i decyzje w kolejności, w jakiej
zapadały:

0. **Baseline: universalista.** Trening mixed SFT na pełnym korpusie (10.5M znaków,
   8000 iteracji) i benchmark OOD: przełącza style (reel scratch 0.48 vs 0.07–0.15
   ekspertów), ale w każdej domenie płytko (jig 0.52/0.62, waltz 0.20/0.29). Ustalony
   jako punkt odniesienia first-hand — i, w trakcie dyskusji nad punktem startowym
   posttrainingu (kandydaci pierwotni: `jig_e128_s3` vs `jig_e32_s3`), zarzucony
   na rzecz universalisty jako studenta: start z eksperta dawałby rigidity do zdjęcia,
   a start z universalisty testuje czysty przyrost „na drugą rękę".
1. **Problem słownika (decyzja architektoniczna).** Modele są char-level: słownik =
   tokenizer = `sorted(set(korpusu))`, każdy checkpoint ma własny. Dwa poziomy problemu:
   (a) ten sam znak ma inne ID w różnych ckpt — trywialne, remap po znaku; (b) znak
   nieobecny w korpusie nauczyciela NIE ma wytrenowanego embeddingu ani logitu
   (weight tying: nieużywane wiersze nie dostają gradientu) — nauczyciel nie potrafi
   go reprezentować, a jego logit to szum inicjalizacji. Rozważane konwencje:
   renormalizacja do supportu nauczyciela / ε-floor / pomijanie pozycji. Decyzja: rozwiązać
   u źródła — JEDEN wspólny tokenizer z pełnego zbioru (54 znaki), przetrenowanie
   nauczycieli z `VOCAB_FROM=universalist_ckpt.pt` (`*_sh_ckpt.pt`). To usuwa problem
   reprezentowalności, ale nie kalibracji: znak wspólnego słownika niewystępujący
   w korpusie nauczyciela nadal ma losowy logit. Stąd α-mixing zamiast ε-floor —
   przy wspólnym słowniku to to samo działanie zapisane inaczej (definicja i mechanika
   w Metodzie), a dodatkowo ogranicza gradienty tam, gdzie nauczyciel jest pewny,
   a student (jeszcze) generuje nie-nauczycielskie treści.
2. **Nauczyciele `_sh`: sanity benchmark.** Diagonale nie gorsze niż u starych ekspertów
   (jig 0.64, reel 0.57, waltz 0.53 — reel i waltz lepiej), pokrycie 100% (szumne logity
   niewytrenowanych wierszy nie psują generacji), rigidity zachowana (hb 0.9+ poza
   domeną). Zweryfikowani jako supervisorzy.
3. **Iteracja 1: RKL continuation-only.** Pierwsza wersja ciąła sekwencje na okna 128
   bez zachodzenia — nauczyciel oceniał środki melodii bez prawdziwego poprzedzenia,
   jego rozkłady były rozmyte i student SPŁASZCZAŁ SIĘ do uniform (entropia 2.2→3.0
   nats/znak przy spadającym KL: mechanizm „tania zgodność z rozmytym celem"). Poprawka:
   okna ze stride 96, pierwsze 32 znaki okna to tylko kontekst — entropia wróciła do
   ~1.0, tlp (log-prob nauczyciela na rolloutach studenta) zaczęło rosnąć. Zbiegł do
   val KL 0.155. Benchmark: waltz 0.20→0.37/0.43, reel cont 0.52→0.56, jig bez zmian —
   ALE reel scratch 0.48→0.26: tryb nigdy nieobecny w treningu zregresował.
4. **Iteracja 2: dodanie zadania scratch.** Rotacja domena×zadanie (6 kombinacji
   w cyklu batchy). Val KL 0.117, entropia stabilna ~1.3–1.4, tlp najlepsze z przebiegu.
   Benchmark: reel scratch naprawiony (0.26→0.53), wszystkie komórki ≥ universalisty —
   Pareto-dominacja (pełna tabela niżej).
5. **Uwaga metodologiczna: best-by-KL słabo skorelowany z jakością benchmarkową** —
   `last` był lepszy na komórkach istotnych (scratch) mimo wyższego val KL; oba
   checkpointy zapisywane od tej pory.
6. **Ablacja: start z eksperta zamiast z universalisty.** Pytanie: ile wyniku bierze się
   z faktu, że universalista widział wszystkie domeny PRZED RKL? Start: `jig_sh` (specjalista,
   nigdy nie widział reel/waltz na żadnym etapie) + ci sami trzej nauczyciele, ten sam
   budżet (`student_jigstart_*`). Przebieg zdrowy (KL 0.146, entropia stabilna); reel
   domykał się najwolniej (KL 0.32 vs 0.065 waltz). Benchmark patrz tabela ablacji
   w Wynikach.

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

### Ablacja: universalist-start vs specialist-start

Ten sam trening (B), inny punkt startowy — student nigdy nie widzący reel/waltz
na żadnym etapie (`student_jigstart_last`, log `ood_jigstart_*.log`):

| komórka (scr / cont) | universalist | RKL uni-start | **RKL jig-start** | jig_sh (init/nauczyciel) |
|---|---|---|---|---|
| jig | 0.52 / 0.62 | 0.62 / 0.66 | 0.60 / 0.63 | 0.62 / 0.70 |
| reel | 0.48 / 0.52 | 0.53 / 0.57 | 0.39 / 0.53 | 0.15 / 0.31 |
| waltz | 0.20 / 0.29 | 0.54 / 0.43 | 0.55 / 0.44 | 0.14 / 0.22 |

Wniosek z ablacji: **pre-RKL ekspozycja na dane wszystkich domen wnosi niemal nic** —
waltz i reel-continuation identyczne między startami; jedyna różnica to reel scratch
(0.39 vs 0.53), czyli tryb, w którym model musi wygenerować cały styl z samego priora,
bez melodicznego zakotwiczenia — dokładnie tam pierwszy kontakt z prawdziwymi danymi
pomaga, a nadzór prawdopodobieństwami najmniej. Dodatkowo: rigidity specjalisty jest
w pełni zdejmowalna (hb 0.94 → 0.04–0.25), a cena za domyk w dwóch obcych domenach to
lekki spadek w domenie własnej (jig 0.63 vs 0.70 nauczyciela w continuation).

**Wnioski:**

1. **Second-hand działa — i to prawie w całości**: student po RKL zbliża się do każdego
   specjalisty na jego własnej przekątnej (0.02–0.04), trenowany w fazie RKL tylko na
   prawdopodobieństwach nauczycieli; ablacja (start z eksperta bez kontaktu z danymi
   reel/waltz w ogóle) pokazuje, że pre-RKL ekspozycja universalisty wnosi wyłącznie
   ~0.14 na komórce reel scratch.
2. **Waltz: 0.20→0.54 scratch** — sufit sędziego (0.544), poziom własnego nauczyciela
   (0.53). Najmniejsza domena (pula 348) domknięta.
3. **Pareto-dominacja nad universalistą** w każdej komórce — brak kosztu zapominania;
   off-diagonal hb spadło do 0.06–0.15 (student słucha nagłówków we wszystkich kierunkach).
4. **Scratch musi być w treningu**, żeby nie zregresować (Przebieg, krok 4).
5. **Rigidity ekspertów nie przeniosła się na studenta**: nauczyciele poza domeną
   generują w swoim metrum (hb 0.9+), rotacja trzech nauczycieli nauczyła studenta
   przełączania zależnie od nagłówków zamiast jednego sztywnego stylu.
6. Sanitarny wskaźnik w trakcie treningu: tlp (log-prob nauczyciela na rolloutach
   studenta) rosnący + stabilna entropia = zdrowy przebieg; rosnąca entropia przy
   spadającym KL = spłaszczanie (sygnał do przerwania / zmiany α).

**Otwarte:** off-policy distillation (nauczyciel na prawdziwych sekwencjach) — izolacja
on-policy; sensitivity α i top-k; odtworzenie checkpointu continuation-only dla pełnej
tabeli abacyjnej. Forward KL został uruchomiony jako kontrola kierunku i jest opisany
poniżej.

**Zastrzeżenie (leak generatory↔val sędziego):** sędzia nie wycieka do treningu w żadnej
formie (w `train_rkl.py` z `train_judge` importowane są tylko narzędzia danych; sygnał
treningowy to wyłącznie rozkłady nauczycieli; pule promptów RKL z splitu TRENINGOWEGO).
Ale generatory (universalista, nauczyciele `_sh`) trenowały na pełnym `tunes.csv`,
w tym na melodiach z walidacyjnego splitu sędziego — benchmark może więc mierzyć częściowo
zapamiętywanie (najbardziej continuation: prompt = ćwiartka ciała melodii walidacyjnej;
najmniej scratch). Wyciek wspólny dla wszystkich porównywanych modeli, więc wnioski
relatywne (RKL vs universalista, uni-start vs jig-start) pozostają w mocy; wartości
absolutne traktować jako optymistyczne. Clean-room wariant i szczegóły: [[Judge-Sedzia-Generacji]],
sekcja „Zastrzeżenie: split chroni sędziego, ale nie generatory".

### Konsekwencje: co zostaje, a co nie

Wyciek działa jak stała dopłatka do wyników każdego generatora, więc:

- **Porównania (różnice) zostają**: RKL vs universalista, uni-start vs jig-start,
  nauczyciel vs student — wszystkie modele dziedziczą ten sam nadmiar informacji,
  więc różnice między nimi mierzą realny efekt interwencji, nie wyciek.
- **Ratia score/ref zostają**: sufit (`ref`) liczony na prawdziwych melodiach jest
  niewrażliwy na zapamiętywanie przez generatory; normalizacja sufitem częściowo
  usuwa wspólną dopłatkę.
- **Wartości absolutne nie zostają**: score 0.54 na waltz to górne oszacowanie —
  część tej liczby to odtworzenie zapamiętanych ciał, nie opanowanie stylu.
- **Granica zaufania**: dopłatka nie musi być identyczna między modelami (pojemność
  i liczba epok na korpusie z wyciekiem różnią się między checkpointami) — wnioski
  oparte na dużych lukach (≥0.1, np. waltz 0.20→0.54) są bezpieczne; porównania na
  granicy szumu (±0.03–0.05) przy wycieku tracą jeszcze trochę wiarygodności.

---

## Plan stopniowego uwiarygodnienia eksperymentu

Plan ma najpierw doprecyzować i ustabilizować obecny wynik, a dopiero później rozszerzać
zakres tezy. Nie zmieniamy od razu eksperymentu w większy program badawczy.

### Zakres tezy po doprecyzowaniu

Najbezpieczniejsze twierdzenie brzmi:

> On-policy multi-teacher KL post-training uczy wspólnego studenta warunkować generację
> na jawnym promptcie metrum/tonacji i poprawia wynik OOD w zadaniu muzycznym.

Trzy wyniki należy rozdzielać:

- **scratch** — najczystszy test transferu z samych prawdopodobieństw nauczyciela oraz
  nagłówków; podczas RKL nie ma prawdziwego prefiksu melodii;
- **continuation** — transfer przy dodatkowym zakotwiczeniu w prawdziwym prefiksie melodii;
  nie jest to transfer „bez danych wejściowych", bo student widzi ten prefiks;
- **obecne domeny** — test sterowania stylem skojarzonym z metrum, nie test rozdzielenia
  stylu od metrum.

`M:` jest tu zamierzonym sygnałem sterującym, a nie błędem pomiaru. Ograniczenie polega
na tym, że w obecnym układzie `M:6/8`, `M:4/4` i `M:3/4` prawie jednoznacznie wskazują
odpowiednio jiga, reela i walca. Wynik pokazuje więc przede wszystkim nauczenie się
warunkowania generacji na tym sygnale. Nie rozstrzyga jeszcze, czy student nauczył się
reprezentacji stylu niezależnej od metrum.

### Etap 1 — małe poprawki w kodzie i pomiarze

1. **Sprawdzić i poprawić wyrównanie okien w `train_rkl.py`.** Przy 32 znakach kontekstu
   pierwszą pozycję po kontekście przewiduje logit na pozycji 31. Obecne `lo=ctx_chars`
   może pomijać jeden token w każdym kolejnym oknie.
2. **Dodać walidację ABC/MIDI do benchmarku.** Obecny `benchmark_ood.py` sprawdza słownik
   i minimalną długość ciała, ale nie parsuje wygenerowanego ABC. Raportować osobno:
   odsetek poprawnie parsowalnych melodii, sukces konwersji do MIDI, błędy długości taktów
   oraz score sędziego warunkowy na poprawnym parsowaniu.
3. **Ustabilizować ewaluację.** Pule promptów są stałe, ale rollouty są losowe. Używać
   kilku stałych seedów generacji i raportować średnią ± odchylenie, zamiast wybierać
   checkpoint wyłącznie na podstawie jednego stochastycznego przebiegu.
4. **Zapisywać metadane eksperymentu:** commit kodu, seed, checkpointy nauczycieli,
   `alpha`, temperaturę, `top-k`, proporcję zadań, długość promptu, budżet iteracji
   oraz hash słownika i danych.

### Etap 2 — tanie ablacje

5. **Uruchomić forward KL.** Kod już obsługuje `--direction forward`; identyczny budżet
   pozwoli sprawdzić, czy efekt zależy konkretnie od reverse KL.
6. **Uruchomić RKL tylko dla `scratch`.** Kod obsługuje `--tasks scratch`. To będzie
   główny wariant dla twierdzenia o transferze „z drugiej ręki".
7. **Porównać trzy warianty:** scratch-only, continuation-only oraz cont+scratch.
   Dzięki temu będzie wiadomo, czy poprawa wynika z nauki obsługi nagłówków, z realnego
   prefiksu, czy z ich połączenia.
8. **Dodać kontrolę dodatkowego post-trainingu bez nauczyciela**, przy tym samym studencie,
   promptach, liczbie iteracji i budżecie. Pozwoli to odróżnić efekt RKL od samego dalszego
   treningu.
9. **Powtórzyć główne warianty na co najmniej trzech seedach.** Wystarczy początkowo:
   universalista, reverse-KL cont+scratch, forward-KL cont+scratch oraz reverse-KL
   scratch-only.

### Etap 3 — kontrola wycieku danych

10. **Zbudować clean-room split generatorów.** Dodać do `prepare_data.py` możliwość
    wykluczenia listy `tune_id` użytych w walidacji sędziego, a następnie przetrenować
    universalistę i nauczycieli bez tych melodii.
11. **Rozdzielić nazewnictwo wyników:** continuation opisywać jako „distillation z
    prawdziwym prefiksem", a scratch jako „distillation z nagłówków i prawdopodobieństw
    nauczyciela". To usuwa niejednoznaczność bez zmiany metody.

### Etap 4 — dopiero później: test ponad routingiem po metrum

12. **Użyć domen o tym samym metrum**, np. reel kontra hornpipe (4/4) albo waltz kontra
    mazur (3/4), oraz dodać jawny identyfikator stylu, np. `D:reel`. Wtedy `M:` nie może
    samo wybrać nauczyciela i będzie można sprawdzić, czy student przełącza styl, a nie
    tylko kojarzy różne metra z różnymi ekspertami.

Ten etap nie jest konieczny do obrony węższej tezy o prompt-conditioned generation.
Jest konieczny dopiero dla mocniejszego twierdzenia o reprezentacji stylu niezależnej
od metrum.

### Priorytet

Wyrównanie okien, walidacja ABC/MIDI oraz ablacja forward KL są wykonane. Następne
priorytety to: stałe powtórzenia na kilku seedach → scratch-only i pozostałe ablacje →
clean-room bez melodii walidacyjnych → domeny o wspólnym metrum. VQ/SAE, meta-atencja
i pełny router nie są teraz potrzebne do uwiarygodnienia tego wyniku.

### Interpretacja walidacji ABC/MIDI

Benchmark raportuje teraz trzy rozdzielne własności wygenerowanego wyniku:

- `score` pozostaje dotychczasowym wynikiem sędziego liczonym dla wszystkich żądanych
  próbek; błędy słownika i zbyt krótkie ciała nadal obniżają go do zera i pozostają w
  `invalid` oraz `coverage`;
- `score_valid_abc` to ten sam wynik sędziego, ale warunkowo na próbkach, które dały się
  ściśle sparsować jako ABC (`valid_abc_samples`); jeśli żadna taka próbka nie istnieje,
  wynik ma wartość `null`;
- `validation.sanitized_midi_ok` oznacza, że istniejąca ścieżka sanitizacji/naprawy ABC
  zdołała wyrenderować MIDI. Nie oznacza to, że surowe ABC było poprawne — dlatego
  `raw_abc_valid` i `sanitized_midi_ok` są raportowane osobno.

Surowe ABC jest parsowane dokładnie w postaci pierwszej wygenerowanej melodii, bez
sanitizacji. Błędy długości są liczone z tego właśnie, ściśle sparsowanego score'a:
porównywana jest długość taktu z `barDuration`, z tolerancją `1e-6`. Pierwszy i ostatni
takt każdej partii są wyłączone, ponieważ pickup oraz niepełne zakończenie są legalnymi
wyjątkami; sprawdzane są wyłącznie wewnętrzne takty z dostępnym `barDuration`.

W ten sposób ścisły błąd parsowania i sukces renderera naprawiającego zapis pozostają
odrębnymi informacjami o generacji, a nie dwoma nazwami tej samej metryki.
Pliki MIDI są tworzone w katalogu tymczasowym i usuwane po ocenie; benchmark nie zostawia
artefaktów w repozytorium.

### Rebenchmark: clean-data baseline, reverse KL i forward KL

Po naprawieniu dekodowania pełnej sekwencji zwracanej przez `GPT.generate()` wykonano
porównanie na tych samych ustawieniach benchmarku (`seed=42`, temperatura `0.85`,
`top-k=18`, 420 nowych znaków, 100 próbek na komórkę):

```text
benchmark_ood.py --models clean_data_ckpt.pt student_clean_all_ckpt.pt \
  student_clean_all_forward_ckpt.pt --quiet-validation-warnings
```

`clean_data_ckpt.pt` jest bazowym modelem bez RL, `student_clean_all_ckpt.pt` to
student po reverse KL, a `student_clean_all_forward_ckpt.pt` to student po forward KL.
W tym rerunie referencje sędziego wynosiły: jig `0.782`, reel `0.757`, waltz `0.584`.

Średnie poniżej są prostą średnią z sześciu komórek (trzy metra × scratch/continuation),
a nie średnią ważoną liczbą taktów:

| model | score (wszystkie próbki) | score_valid_abc | raw ABC valid | MIDI sanitized | błędy taktów |
|---|---:|---:|---:|---:|---:|
| clean-data, bez RL | 0.533 | 0.561 | 92.3% | 92.5% | 3295 / 28316 = 11.6% |
| reverse KL | 0.568 | 0.592 | 89.2% | 89.5% | 1353 / 26792 = 5.0% |
| forward KL | 0.568 | 0.595 | 86.8% | 87.2% | 2350 / 26932 = 8.7% |

Najważniejsze obserwacje:

1. **Oba kierunki KL poprawiają wynik sędziego względem bazowego modelu**: średni
   `score` rośnie z `0.533` do `0.568`. Reverse KL jest minimalnie lepszy w
   continuation (`0.585` vs `0.576` dla forward KL), natomiast forward KL jest
   minimalnie lepszy w scratch (`0.560` vs `0.551`). Ogólny wynik obu studentów jest
   praktycznie remisowy.
2. **Reverse KL daje najlepszą regularność rytmiczną.** Liczba błędnych taktów spada
   z `11.6%` do `5.0%`; forward KL osiąga wynik pośredni (`8.7%`). Jest to osobna
   własność od poprawności składni ABC: score może dać się sparsować, ale mieć
   niepełne albo przepełnione takty.
3. **Ścisła poprawność ABC nie poprawiła się.** Bazowy model ma `92.3%` surowo
   parsowalnych melodii, reverse KL `89.2%`, a forward KL `86.8%`. Zatem w obecnym
   ustawieniu posttraining poprawia jakość muzyczną i długość taktów kosztem części
   syntaktycznej poprawności generowanego ABC.
4. **Sanitizacja jest prawie bezczynna jako naprawa.** Różnica między `raw_abc_valid`
   i `sanitized_midi_ok` wynosi tylko 1–2 próbki na 600. Sukces renderera należy więc
   interpretować jako oddzielną informację o tolerancji ścieżki MIDI, a nie jako dowód,
   że surowe ABC było poprawne.
5. **Pokrycie nie jest źródłem różnic.** Dla wszystkich modeli scratch ma 100% pokrycia,
   a continuation odpowiednio 99% dla jig, 93% dla reel i 83% dla waltz. Różnice w
   `score` wynikają więc z generacji/judge'a, a nie z nowej walidacji ABC ani z
   odrzucania próbek z powodu krótkiego ciała.
6. **Największy zysk pozostaje w waltz**, gdzie wynik scratch rośnie z `0.270` do
   `0.494` (reverse) i `0.497` (forward), a continuation z `0.415` do `0.448` i
   `0.466`. Reel pozostaje prawie bez zmiany, a jig jest stabilny, choć continuation
   studentów jest nieco niższe niż bazowe (`0.711`: `0.701` reverse, `0.684` forward).
7. **Home-bias w continuation spada u studentów**, szczególnie po forward KL: dla jig
   `0.87 → 0.82`, a dla waltz `0.18 → 0.14` (bazowy → forward). To jest zgodne z
   lepszym reagowaniem na prompt, ale nie powinno być utożsamiane z poprawnością ABC.
   W scratch home-bias jig pozostaje wysoki (`~0.92`) u obu studentów, co wskazuje na
   utrzymanie silnego priora domenowego w tym kierunku.

Ten rerun kwalifikuje wcześniejsze sformułowanie o pełnej „Pareto-dominacji” nad bazą:
pozostaje ono prawdziwe dla wcześniejszej tabeli benchmarku i jej celu jakościowego,
ale nie dla wszystkich nowych metryk ani dla każdego pojedynczego wyniku. W szczególności
nowa walidacja pokazuje wyraźny trade-off: reverse KL jest najlepszy strukturalnie,
forward KL ma najwyższy warunkowy `score_valid_abc`, lecz bazowy model ma najwyższą
surową parsowalność ABC. Różnice rzędu `0.01–0.03` należy traktować ostrożnie przy
pojedynczym seedzie i 100 próbkach na komórkę; potrzebne są powtórzenia na kilku seedach.
