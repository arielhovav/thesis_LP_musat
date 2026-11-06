"""Hand-built minimal a^n b^n counter for the no-LN NoPE softmax looped transformer, int8 grid."""
import math, torch, torch.nn.functional as F
from model import LoopedTransformer
from quantization import QuantConfig
from grammar import ANBN, BOS, A, B, END, PAD, collate, make_train_set, log2_prob_of_n
LN2 = math.log(2)
D = 0.05
def build(lv, loops=6):
    m = LoopedTransformer(5, 32, 4, 128, loops, QuantConfig("int8","fixed",D), pos="nope", norm="none", bias="ffh")
    with torch.no_grad():
        for n, w in m.quantizable_named_weights(): w.zero_()
        e, p_b, p_bos, wv, wo, h_cmp, h_ph, h_a, beta_a, beta_b = [k*D for k in lv]
        m.emb.weight[A, 0] = e;  m.emb.weight[B, 0] = -e          # d0: token sign (+a, -b, bos 0)
        m.emb.weight[B, 2] = p_b; m.emb.weight[BOS, 2] = -p_bos   # d2: own-token phase (b +, bos -)
        m.block.attn.wv.weight[0, 0] = wv                          # v0 <- d0
        m.block.attn.wo.weight[1, 0] = wo                          # d1 <- mean_prefix(v0) = G (n_a-n_b)/t
        m.head.weight[END, 1] = -h_cmp                             # logit(#) = -h_cmp*d1 + h_ph*d2
        m.head.weight[END, 2] = h_ph
        m.head.weight[A, 2] = -h_a                                 # logit(a) = beta - h_a*d2
        m.head.bias[A] = beta_a; m.head.bias[B] = beta_b           # coin: beta_a-beta_b = ln(0.7/0.3); both >> 0 so <pad>,<bos> logits (0) get no mass
    m.eval(); return m
@torch.no_grad()
def score(m, max_n=300):
    tr = make_train_set(500, seed=100, p=0.3, task=ANBN)
    inp, tgt = collate(tr.seqs, "cpu")
    lp = F.log_softmax(m(inp).float(), -1); p = lp.exp().clamp_min(1e-10)
    tb = -(p.log()/LN2).gather(-1, tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)*(tgt!=PAD)
    dh = float(tb.sum()); opt = -sum(log2_prob_of_n(n,0.3) for n in tr.ns)
    out = {"DH": dh, "DH_opt": opt}
    fails=None; gfail=None
    for n in list(range(1,41))+list(range(45,max_n+1,5)):
        inp,tgt = collate([ANBN.string_for_n(n)],"cpu"); lp=F.log_softmax(m(inp).float(),-1)
        p=lp.exp().clamp_min(1e-10); tb=-(p.log()/LN2).gather(-1,tgt.clamp_min(0).unsqueeze(-1)).squeeze(-1)[0]
        lo,hi = ANBN.counter_span(n); cb=float(tb[lo:hi].sum()); ok=bool((lp.argmax(-1)[0,lo:hi]==tgt[0,lo:hi]).all())
        if n in (25,50,100,150,200,300): out[f"c{n}"]=round(cb,3)
        if cb>0.5 and fails is None: fails=n
        if not ok and gfail is None: gfail=n
    out["exact_fail"]=fails; out["greedy_fail"]=gfail
    for k in (1,20,100,300):
        inp=torch.tensor([[BOS]+[A]*k]); out[f"pb{k}"]=round(float(F.softmax(m(inp).float(),-1)[0,-1,B]),4)
    inp=torch.tensor([[BOS]]); out["P(a|bos)"]=round(float(F.softmax(m(inp).float(),-1)[0,-1,A]),4)
    L0=sum(int((w!=0).sum()) for _,w in m.quantizable_named_weights()); out["L0"]=L0
    out["H_natural"]=round(L0*(math.log2(m.param_report()["quantizable_params"])+math.log2(254)),1)
    return out
if __name__=="__main__":
    import sys
    # levels: e, p_b, p_bos, wv, wo, h_cmp, h_ph, h_a, beta
    for lv in [(127,20,20,127,127,127,35,127,127,110)]:
        m=build(lv); s=score(m); print(lv); print(s)
