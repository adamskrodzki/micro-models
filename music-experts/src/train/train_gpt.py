"""Trening GPT od zera na korpusie ABC (L03/L08 w praktyce).
Batche -> strata cross-entropy -> backprop -> AdamW -> val loss -> checkpoint.
"""
import os, re, time, math, sys, argparse
from contextlib import nullcontext
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.gpt import GPT, GPTConfig

ap = argparse.ArgumentParser()
ap.add_argument("data", nargs="?", default="data/jigs.abc")
ap.add_argument("ckpt", nargs="?", default="data/models/jig_ckpt.pt")
ap.add_argument("losslog", nargs="?", default="data/models/jig_loss_log.csv")
ap.add_argument("vocab_from", nargs="?", default=None,
                help="wspólny słownik z innego ckpt (do stitchu)")
ap.add_argument("seed", nargs="?", type=int, default=20260620)   # niezależne modele do E_CKA/E0.5
ap.add_argument("--max-iters", type=int, default=2000,
                help="skalować z rozmiarem korpusu (mixed ~4x jig)")
a = ap.parse_args()

# --- hiperparametry ---
block_size  = 128
batch_size  = 32
n_layer     = int(os.environ.get("N_LAYER", 4))    # ENV: sweep skali (E_CKA)
n_head      = 4
n_embd      = int(os.environ.get("N_EMBD", 128))    # ENV: sweep skali (E_CKA)
dropout     = 0.1
lr          = 3e-4
max_iters   = a.max_iters
eval_interval = 200
eval_iters  = 100
warmup      = 100

SEED = a.seed
torch.manual_seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
sys.stdout.reconfigure(encoding="utf-8")
print(f"urządzenie: {device} | bf16: {use_bf16}")

# --- ścieżki z argumentów ---
DATA, CKPT, LOSSLOG, VOCAB_FROM = a.data, a.ckpt, a.losslog, a.vocab_from
print(f"dane: {DATA} -> checkpoint: {CKPT} | max_iters: {max_iters}")

# --- dane: char-level, okna per melodia (nigdy nie przekraczają granicy X:) ---
text = open(DATA, encoding="utf-8").read()
if VOCAB_FROM:
    vck = torch.load(VOCAB_FROM, map_location="cpu", weights_only=False)
    stoi, itos = vck["stoi"], vck["itos"]
    chars = [itos[i] for i in range(len(itos))]
    print(f"wspólny słownik z {VOCAB_FROM}: {len(chars)} znaków")
else:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
# pad: znak spoza korpusu (brak w ALLOWED); targety na padzie = -100 -> cross_entropy je ignoruje
PAD_ID = stoi.get("\x00", len(itos))
if "\x00" not in stoi:
    stoi["\x00"] = PAD_ID
    itos[PAD_ID] = "\x00"
starts = [m.start() for m in re.finditer(r"(?m)^X:", text)]
tunes = []
for i, s in enumerate(starts):
    e = starts[i + 1] if i + 1 < len(starts) else len(text)
    tunes.append(torch.tensor([stoi[c] for c in text[s:e]], dtype=torch.long))
perm = torch.randperm(len(tunes))                 # tasowanie melodii -> reprezentatywny val
tunes = [tunes[j] for j in perm]
n = int(0.9 * len(tunes))
train_tunes, val_tunes = tunes[:n], tunes[n:]
print(f"słownik: {len(itos)} | melodie: train {len(train_tunes)} / val {len(val_tunes)} | "
      f"znaki: {sum(t.size(0) for t in tunes):,}")

def get_batch(split):
    pool = train_tunes if split == "train" else val_tunes
    xs, ys = [], []
    for _ in range(batch_size):
        t = pool[torch.randint(len(pool), (1,)).item()]
        L = t.size(0)
        if L > block_size:                        # mieści pełne okno block_size+1
            if torch.rand(1).item() < 0.25 or L == block_size + 1:
                off = 0                           # 25% okien od nagłówka (X: na pozycji 0)
            else:
                off = int(torch.randint(1, L - block_size, (1,)).item())
        else:
            off = 0                               # krótka melodia: całość + pad
        w = t[off:off + block_size + 1]           # +1 na przesunięte targety
        if w.size(0) < block_size + 1:
            pad = block_size + 1 - w.size(0)
            w = torch.cat([w, torch.full((pad,), PAD_ID, dtype=torch.long)])
        xs.append(w[:-1])
        ys.append(w[1:])
    x = torch.stack(xs)
    y = torch.stack(ys)
    y[y == PAD_ID] = -100                         # nie uczymy przewidywać padu
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def lr_at(it):  # warmup + cosine decay (L08)
    if it < warmup:
        return lr * it / warmup
    r = (it - warmup) / (max_iters - warmup)
    return lr * 0.1 + 0.5 * lr * 0.9 * (1 + math.cos(math.pi * r))

cfg = GPTConfig(vocab_size=len(itos), block_size=block_size,
                n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout)
model = GPT(cfg).to(device)
print(f"parametry modelu: {model.num_params():,}")
opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=0.1)

best_val = float("inf")
log = [("iter", "train_loss", "val_loss")]
t0 = time.time()
for it in range(max_iters + 1):
    if it % eval_interval == 0 or it == max_iters:
        L = estimate_loss()
        ppl = math.exp(L["val"])
        print(f"iter {it:4d} | train {L['train']:.3f} | val {L['val']:.3f} | ppl {ppl:.2f} | {time.time()-t0:.0f}s")
        log.append((it, round(L["train"], 4), round(L["val"], 4)))
        if L["val"] < best_val:           # zapis najlepszego (early-stop logic)
            best_val = L["val"]
            torch.save({"model": model.state_dict(), "config": cfg,
                        "stoi": stoi, "itos": itos, "val_loss": best_val},
                       CKPT)
    if it == max_iters:
        break
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    x, y = get_batch("train")
    with ctx:
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()

with open(LOSSLOG, "w", encoding="utf-8") as f:
    f.write("\n".join(",".join(map(str, row)) for row in log))
print(f"\ngotowe. best val loss: {best_val:.3f} (ppl {math.exp(best_val):.2f})")
print(f"checkpoint -> {CKPT} | krzywa -> {LOSSLOG}")
