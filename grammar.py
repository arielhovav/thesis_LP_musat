"""
grammar.py
==========
Formal-language data generation and analytical optima, following the
evaluation protocol of Abudy, Well, Chemla, Katzir & Lan (2025),
"A Minimum Description Length Approach to Regularization in Neural Networks".

Tasks (select with --task; default 'anbn' keeps every existing run intact)
--------------------------------------------------------------------------
  anbn    : a^n b^n #      vocab {<pad>,<bos>,a,b,#}      (Abudy et al. Table 1)
  anbncn  : a^n b^n c^n #  vocab {<pad>,<bos>,a,b,#,c}    (Abudy et al. Table 1)
Both draw n ~ Geometric(p=0.3) with support {1, 2, ...}, P(n) = p * (1-p)^(n-1),
so both share the SAME analytical optimum -log2 P(n) per string (the paper's
tables show 2.94 for both languages). The anbn vocabulary is a strict prefix
of the anbncn one ('c' gets the new id 5), so the token ids of <pad>, <bos>,
a, b and '#' are identical across tasks and every anbn checkpoint, config and
analysis script keeps working unchanged. A checkpoint records its task in
config["task"]; configs written before this field existed resolve to 'anbn'.

The model is trained on next-token prediction: given <bos> + prefix,
predict every symbol of the string including the final '#'.

Evaluation (paper Section 2.6):
  * Accuracy is NOT used. We measure |D:H| = cross-entropy in *bits*.
  * The test set is EXHAUSTIVE: all n from 1 to (n_max_train + extra),
    weighted by their true probability P(n) (renormalized over the set).
  * The score is compared to the analytically computed optimum:
        Optimal = sum_n  w_n * (-log2 P(string_n))
    For the geometric grammar this converges to H(Geom(0.3))/1 ~ 2.94 bits
    per string, matching the "(Golden) 2.94" entries in the paper's tables.
  * Delta% = (|D:H| - Optimal) / Optimal * 100.
    Delta% > 0  => the model is worse than the true distribution (underfit /
                   failure to generalize);
    Delta% < 0  => the model beats the true distribution on the finite set,
                   i.e. it has overfit the test set's truncation.

The predictive structure of a^n b^n under the geometric prior
-------------------------------------------------------------
By memorylessness of the geometric distribution, the true next-symbol
distribution is:
  * position 1 (after <bos>):       'a' with prob 1            (n >= 1)
  * inside the a-run, after k a's:  'a' with prob 1-p = 0.7,
                                    'b' with prob   p = 0.3
  * once the first 'b' appears:     fully deterministic
                                    (b's until counts match, then '#').
So a *perfect* network must (i) implement the 0.7/0.3 coin during the a-run,
and (ii) implement an exact counter for the deterministic b-run. Failures on
long strings show up as Delta% > 0 driven by the counting part.

For a^n b^n c^n the coin part is identical (the a-run decides n); once the
first 'b' appears the remainder  b^(n-1) c^n #  is fully deterministic, so the
counter must now be re-used twice (count b's against a's, then c's against
a's -- Abudy et al.: "a^n b^n c^n requires two counters"). In both tasks the
deterministic part is target positions n+1 .. k*n (k = number of letter runs),
which is what Task.counter_span() returns and what every "counter bits" number
in probe_lengths / audit_counters / ga_polish is computed over.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

# -----------------------------------------------------------------------------
# Vocabulary
# -----------------------------------------------------------------------------
PAD, BOS, A, B, END = 0, 1, 2, 3, 4
C = 5                                   # only used by the anbncn task
VOCAB = {"<pad>": PAD, "<bos>": BOS, "a": A, "b": B, "#": END}   # anbn vocab
ID2TOK = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)                 # == 5: the a^n b^n vocabulary size,
                                        # kept for backward compatibility.
                                        # New code should use task.vocab_size.

P_GEOM_DEFAULT = 0.3


@dataclass(frozen=True, eq=False)   # singletons: identity equality/hash
class Task:
    """A counting language  l_1^n l_2^n ... l_k^n #  over the geometric prior.

    letters : token ids of the k runs, in order.
    end     : token id of the end-of-string marker.
    vocab   : name -> id, INCLUDING <pad> and <bos>; len(vocab) is the model's
              embedding/head size, so it must not change for an existing task.
    """
    name: str
    letters: tuple[int, ...]
    end: int
    vocab: dict

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def k(self) -> int:
        """Number of letter runs (2 for a^n b^n, 3 for a^n b^n c^n)."""
        return len(self.letters)

    @property
    def id2tok(self) -> dict:
        return {v: k for k, v in self.vocab.items()}

    def string_for_n(self, n: int) -> list[int]:
        """Token ids of l_1^n ... l_k^n # (without <bos>)."""
        out: list[int] = []
        for tok in self.letters:
            out += [tok] * n
        return out + [self.end]

    def seq_len(self, n: int) -> int:
        return self.k * n + 1

    def counter_span(self, n: int) -> tuple[int, int]:
        """Half-open range of TARGET indices that are deterministic given the
        first 'b' has appeared: n+1 .. k*n (inclusive of the final '#').
        Targets 0..n are the coin (a-run + the first 'b' that resolves it)."""
        return n + 1, self.k * n + 1

    def counter_run_spans(self, n: int) -> dict[str, tuple[int, int]]:
        """counter_span(n) split per letter run, so a failure can be located:
        {'b': (n+1, 2n), 'c': (2n, 3n+1)} for a^n b^n c^n; {'b': (n+1, 2n+1)}
        for a^n b^n. The final '#' is attached to the last run (predicting it
        is that run's last count). The spans partition counter_span(n)."""
        names = [k for k, v in self.vocab.items() if v in self.letters[1:]]
        names = sorted(names, key=lambda k: self.letters.index(self.vocab[k]))
        out = {}
        for j, name in enumerate(names, start=2):        # run index j = 2..k
            lo = (j - 1) * n + (1 if j == 2 else 0)      # first b is the coin
            hi = j * n + (1 if j == self.k else 0)       # last run owns '#'
            out[name] = (lo, hi)
        return out

    def prefix(self, counts) -> list[int]:
        """Token ids of l_1^{c_1} l_2^{c_2} ... for a partial string (no #)."""
        out: list[int] = []
        for tok, c in zip(self.letters, counts):
            out += [tok] * c
        return out


ANBN = Task("anbn", letters=(A, B), end=END, vocab=dict(VOCAB))
ANBNCN = Task("anbncn", letters=(A, B, C), end=END,
              vocab={**VOCAB, "c": C})
TASKS = {t.name: t for t in (ANBN, ANBNCN)}
TASK_NAMES = tuple(TASKS)


def get_task(name: str = "anbn") -> Task:
    if name not in TASKS:
        raise ValueError(f"unknown task {name!r}; choose from {TASK_NAMES}")
    return TASKS[name]


def task_from_config(cfg: dict) -> Task:
    """Task of a saved checkpoint/results config. Configs written before the
    --task flag existed have no 'task' key and were all a^n b^n."""
    return get_task(cfg.get("task", "anbn"))


def string_for_n(n: int, task: Task = ANBN) -> list[int]:
    """Token ids of the task's string for n (without <bos>). Default: a^n b^n #."""
    return task.string_for_n(n)


def log2_prob_of_n(n: int, p: float = P_GEOM_DEFAULT) -> float:
    """log2 P(n) under Geometric(p), support {1,2,...}."""
    return math.log2(p) + (n - 1) * math.log2(1.0 - p)


# -----------------------------------------------------------------------------
# Training data
# -----------------------------------------------------------------------------
@dataclass
class TrainSet:
    ns: list[int]                 # the sampled n for each training string
    seqs: list[list[int]]         # token id sequences (a^n b^n #)
    n_max: int                    # largest n seen in training
    p: float
    task: Task = ANBN

    @property
    def optimal_bits_per_string(self) -> float:
        """Average -log2 P(s) of the empirical training sample = the training
        |D:H| (per string) that a perfect model of the true grammar achieves."""
        return -sum(log2_prob_of_n(n, self.p) for n in self.ns) / len(self.ns)


def make_train_set(num_strings: int, seed: int, p: float = P_GEOM_DEFAULT,
                   task: Task = ANBN) -> TrainSet:
    """The sampled n's depend only on (seed, p), so the anbn and anbncn train
    sets for a given seed are the SAME n's rendered in different languages --
    a matched comparison across tasks."""
    rng = np.random.default_rng(seed)
    ns = rng.geometric(p, size=num_strings).tolist()   # support {1,2,...}
    seqs = [task.string_for_n(n) for n in ns]
    return TrainSet(ns=ns, seqs=seqs, n_max=max(ns), p=p, task=task)


# -----------------------------------------------------------------------------
# Exhaustive test set (paper Section 2.6 / Appendix A)
# -----------------------------------------------------------------------------
@dataclass
class TestSet:
    ns: list[int]
    seqs: list[list[int]]
    weights: np.ndarray           # renormalized P(n) over the set
    optimal_bits: float           # sum_n w_n * (-log2 P(n))
    n_train_max: int              # boundary between in-range and OOD lengths
    p: float
    task: Task = ANBN

    def split_masks(self, near_width: int = 10):
        """Boolean masks over the exhaustive set:
          in_range : n <= n_train_max
          near_ood : n_train_max < n <= n_train_max + near_width
                     (the VALIDATION slice: OK to use for checkpoint/HP choice)
          far_ood  : n > n_train_max + near_width
                     (REPORT-ONLY: never used for any selection decision)
        Splitting OOD this way avoids test-set snooping: the peak-then-decay
        shape of length generalization tempts picking the checkpoint where the
        test metric was best, which would bias every reported number."""
        ns = np.asarray(self.ns)
        in_range = ns <= self.n_train_max
        near = (~in_range) & (ns <= self.n_train_max + near_width)
        far = ns > self.n_train_max + near_width
        return in_range, near, far


def make_exhaustive_test_set(
    n_train_max: int,
    extra: int = 50,
    p: float = P_GEOM_DEFAULT,
    task: Task = ANBN,
) -> TestSet:
    """All n = 1 .. n_train_max + extra, weighted by true probability.

    The paper uses extra = 1000 for RNNs; for a transformer, sequence length is
    k*n+1 tokens (k=2 for a^n b^n, 3 for a^n b^n c^n) and attention is O(L^2),
    so the default here is extra = 50. Increase --test_extra if you want a
    harsher length-generalization probe. The optimum is task-independent: the
    string for n has probability P(n) in both languages.
    """
    n_hi = n_train_max + extra
    ns = list(range(1, n_hi + 1))
    raw = np.array([2.0 ** log2_prob_of_n(n, p) for n in ns], dtype=np.float64)
    weights = raw / raw.sum()
    optimal_bits = float(np.sum(weights * np.array([-log2_prob_of_n(n, p) for n in ns])))
    return TestSet(
        ns=ns,
        seqs=[task.string_for_n(n) for n in ns],
        weights=weights,
        optimal_bits=optimal_bits,
        n_train_max=n_train_max,
        p=p,
        task=task,
    )


# -----------------------------------------------------------------------------
# Batching helpers
# -----------------------------------------------------------------------------
def collate(seqs: list[list[int]], device: torch.device):
    """Pad a list of strings and build (inputs, targets).

    inputs[i]  = <bos> s_1 ... s_{L-1}   (shifted right)
    targets[i] = s_1 ... s_L             (PAD where padded; PAD is ignored in loss)
    """
    L = max(len(s) for s in seqs)
    B = len(seqs)
    inp = torch.full((B, L), PAD, dtype=torch.long)
    tgt = torch.full((B, L), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        t = torch.tensor(s, dtype=torch.long)
        inp[i, 0] = BOS
        inp[i, 1 : len(s)] = t[:-1]
        tgt[i, : len(s)] = t
    return inp.to(device), tgt.to(device)


def entropy_of_geometric(p: float = P_GEOM_DEFAULT) -> float:
    """H(Geom(p)) in bits = (-p log2 p - (1-p) log2 (1-p)) / p  ~ 2.938 for p=0.3.
    This is the asymptotic optimum the paper's tables report as 2.94."""
    return (-p * math.log2(p) - (1 - p) * math.log2(1 - p)) / p


if __name__ == "__main__":
    for task in (ANBN, ANBNCN):
        ts = make_train_set(500, seed=100, task=task)
        te = make_exhaustive_test_set(ts.n_max, extra=50, task=task)
        print(f"[{task.name}] vocab={task.vocab_size} k={task.k}  "
              f"example n=2: {[task.id2tok[t] for t in task.string_for_n(2)]}")
        print(f"  train: {len(ts.seqs)} strings, n_max={ts.n_max}, "
              f"optimal train bits/string = {ts.optimal_bits_per_string:.3f}")
        print(f"  test:  n=1..{ts.n_max + 50}, optimal = {te.optimal_bits:.3f} bits "
              f"(asymptotic H = {entropy_of_geometric():.3f}); "
              f"longest sequence = {task.seq_len(ts.n_max + 50)} tokens")