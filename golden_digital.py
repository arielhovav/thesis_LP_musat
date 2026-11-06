"""Hand-built a^n b^n counter in Musat's regime: ternary weights with delta = M = 1,
causal SUM attention (exact integer counts, no 1/t), attention on loop 1 only (no
iterated sums), activations rounded and clipped to a finite alphabet, no LN, NoPE."""
import math, torch, torch.nn.functional as F
from model import LoopedTransformer
from quantization import QuantConfig
from grammar import ANBN, BOS, A, B, END, PAD, collate, make_train_set, log2_prob_of_n
LN2 = math.log(2)

def build(act_bits=4, act_max=8.0, loops=6, quant="ternary", task="anbn"):
    """quant: ternary (M=1), int4 (M=7) or int8 (M=127), all with delta = 1 -- Musat's three integer cases.
    task 'anbncn' builds the two-comparator program (24 weights, all +-1)."""
    if task == "anbncn":
        return build_anbncn(act_bits, act_max, loops, quant)
    m = LoopedTransformer(5, 32, 4, 128, loops, QuantConfig(quant, "fixed", 1.0), pos="nope",
                          norm="none", attn="sum", attn_loops=1, bias="ffh",
                          act_bits=act_bits, act_max=act_max)
    with torch.no_grad():
        for n, w in m.quantizable_named_weights(): w.zero_()
        m.emb.weight[A, 0] = 1;  m.emb.weight[B, 0] = -1      # d0: token sign (a +, b -, bos 0)
        m.emb.weight[B, 2] = 1;  m.emb.weight[BOS, 2] = -1    # d2: phase (own token)
        m.block.attn.wv.weight[0, 0] = 1                       # v0 <- d0
        m.block.attn.wo.weight[1, 0] = 1                       # d1 <- SUM_prefix v0 = 5.66 (n_a - n_b)
        m.block.attn.wo.weight[3, 0] = 1                       # d3 <- same (a second copy = a weight of 2)
        m.block.ff_out.bias[4] = 1                             # d4 <- +1 per loop = 6: a constant
        m.head.weight[END, 1] = -1; m.head.weight[END, 3] = -1 # logit(#) = 5.66 (phase) - 2 * 5.66 (n_a - n_b)
        m.head.weight[END, 2] = 1
        m.head.weight[A, 2] = -1                               # logit(a) = 1 - 5.66 (b-run), 1 + 5.66 (bos)
        m.head.bias[A] = 1                                     # coin: a - b = 1 -> P(a) = 0.73
        m.head.weight[PAD, 4] = -1; m.head.weight[BOS, 4] = -1 # <pad>, <bos> logits = -6
    m.eval(); return m

def build_anbncn(act_bits=4, act_max=8.0, loops=6, quant="ternary"):
    """a^n b^n c^n #: two integer comparators s1 = n_a - n_b (closes the b-run) and
    s2 = n_a - n_c (closes the c-run), two own-token phases (b, c), one coin.
    24 weights, all +-1. Logits (all sums clipped to +-8 -> 0 / 8):
      a  = 1 - [b] - [c]          (13 at bos, 1 in the a-run, -5 after)
      b  = 0                      (reference)
      c  = [b] - 2*s1 + s2        (b-run: -2 while s1 > 0, 14 at s1 = 0; c-run: 8 while s2 > 0, 0 at s2 = 0)
      #  = [c] - s2               (c-run: -2 while s2 > 0, 6 at s2 = 0; elsewhere <= -6)
      pad, bos = -s1              (<= -8 wherever a or b must be predicted)
    where [b], [c] are +6 on the b / c token, -6 on <bos>, 0 otherwise."""
    from grammar import ANBNCN, C
    # attention on TWO loops: the static embeddings are summed twice, so a lead of
    # exactly one (6) already saturates to 8 and every comparator reads 0 / 8.
    m = LoopedTransformer(6, 32, 4, 128, loops, QuantConfig(quant, "fixed", 1.0), pos="nope",
                          norm="none", attn="sum", attn_loops=2, bias="ffh",
                          act_bits=act_bits, act_max=act_max)
    with torch.no_grad():
        for n, w in m.quantizable_named_weights(): w.zero_()
        m.emb.weight[A, 0] = 1;  m.emb.weight[B, 0] = -1       # d0: s1 source (a +, b -)
        m.emb.weight[A, 4] = 1;  m.emb.weight[C, 4] = -1       # d4: s2 source (a +, c -)
        m.emb.weight[B, 2] = 1;  m.emb.weight[BOS, 2] = -1     # d2: phase [b]
        m.emb.weight[C, 6] = 1;  m.emb.weight[BOS, 6] = -1     # d6: phase [c]
        m.block.attn.wv.weight[0, 0] = 1                       # v0 <- d0
        m.block.attn.wv.weight[1, 4] = 1                       # v1 <- d4
        m.block.attn.wo.weight[1, 0] = 1                       # d1 <- SUM v0 = 6 s1
        m.block.attn.wo.weight[3, 0] = -1                      # d3 <- -6 s1
        m.block.attn.wo.weight[5, 1] = 1                       # d5 <- 6 s2
        m.head.weight[A, 2] = -1;   m.head.weight[A, 6] = -1;  m.head.bias[A] = 1
        m.head.weight[C, 2] = 1;    m.head.weight[C, 1] = -1;  m.head.weight[C, 3] = 1; m.head.weight[C, 5] = 1
        m.head.weight[END, 6] = 1;  m.head.weight[END, 5] = -1
        m.head.weight[PAD, 3] = 1;  m.head.weight[BOS, 3] = 1
    m.eval(); return m


@torch.no_grad()
def score(m, max_n=1000, task=None):
    task = task or ANBN
    tr = make_train_set(500, seed=100, p=0.3, task=task)
    inp, tgt = collate(tr.seqs, "cpu")
    lp = F.log_softmax(m(inp).float(), -1); p = lp.exp().clamp_min(1e-10)
    tb = -(p.log()/LN2).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)*(tgt!=PAD)
    out = {"DH": round(float(tb.sum()),1), "DH_opt": round(-sum(log2_prob_of_n(n,0.3) for n in tr.ns),1)}
    fails=None; gfail=None
    for n in list(range(1,41))+list(range(45,max_n+1,5)):
        inp,tgt = collate([task.string_for_n(n)],"cpu"); lp=F.log_softmax(m(inp).float(),-1)
        p=lp.exp().clamp_min(1e-10); tb=-(p.log()/LN2).gather(-1,tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)[0]
        lo,hi = task.counter_span(n); cb=float(tb[lo:hi].sum()); ok=bool((lp.argmax(-1)[0,lo:hi]==tgt[0,lo:hi]).all())
        if n in (25,50,100,300,1000): out[f"c{n}"]=round(cb,3)
        if cb>0.5 and fails is None: fails=n
        if not ok and gfail is None: gfail=n
    out["exact_fail"]=fails; out["greedy_fail"]=gfail
    for k in (1,20,300): inp=torch.tensor([[BOS]+[A]*k]); out[f"pb{k}"]=round(float(F.softmax(m(inp).float(),-1)[0,-1,B]),3)
    inp=torch.tensor([[BOS]]); out["P(a|bos)"]=round(float(F.softmax(m(inp).float(),-1)[0,-1,A]),3)
    L0=sum(int((w!=0).sum()) for _,w in m.quantizable_named_weights()); out["L0"]=L0
    out["L2sq"]=round(float(sum((w**2).sum() for _,w in m.quantizable_named_weights())),1)
    mlv = m.qcfg.n_levels_per_sign or 1                     # 1 ternary, 7 int4, 127 int8
    out["H_natural"]=round(L0*(math.log2(m.param_report()["quantizable_params"])+math.log2(2*mlv)),1)
    return out

if __name__=="__main__":
    import sys, os
    if len(sys.argv) > 1 and sys.argv[1] == "--save":      # --save <quant> <path> [anbn|anbncn]
        q, path = sys.argv[2], sys.argv[3]
        tname = sys.argv[4] if len(sys.argv) > 4 else "anbn"
        from grammar import get_task
        m = build(4, 8.0, quant=q, task=tname)
        cfg = dict(task=tname, d_model=32, n_heads=4, d_ff=128, n_loops=6, quant=q, scale="fixed", fixed_delta=1.0,
                   pos="nope", attn="sum", norm="none", attn_loops=(2 if tname == "anbncn" else 1),
                   bias="ffh", act_bits=4, act_max=8.0,
                   train_size=500, seed=100, p_geom=0.3, weight_decay=0.0, steps=0, batch_size=256, lr=0.0,
                   test_extra=50, polish={"p_end": 0.3, "hbits": "hand-built digital"})
        torch.save({"state_dict": m.state_dict(), "config": cfg, "step": 0}, path)
        print(f"{q} delta=1 {tname}: {score(m, max_n=300, task=get_task(tname))} -> {path}"); sys.exit()
    for ab, am in [(0,8.0),(4,8.0),(3,8.0)]:
        m=build(ab,am); print(f"act_bits={ab} act_max={am}:", score(m))
    m=build(4,8.0)
    inp,_ = collate([ANBN.string_for_n(3)],"cpu")
    with torch.no_grad():
        lg=m(inp)[0]; pr=F.softmax(lg,-1)
    print("aaabbb# logits pad,bos,a,b,#:"); print(lg.numpy().round(2)); print("P(a),P(b),P(#):"); print(pr[:,2:].numpy().round(3))
