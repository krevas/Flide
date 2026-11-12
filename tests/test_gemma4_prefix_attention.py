"""Gemma 4 prefix attention (Plan.attention "prefix"): the global layers read each prefix both ways and each branch one
way over its prefix and itself; the sliding layers stay causal inside their window. Checked on the CPU against a
reference mask, for the packed path (Gemma4.attention_meta) and the CUDA-graph staging (GemmaGraphs._stage + _body).

The attention kernel reads, per query tile, the union of the queries' key ranges: whole key blocks inside every range
unmasked, the rest filtered by the metadata (job, branch, position, block number, window). So when, for every query,
"keys in range" equals "keys the metadata shows", the result does not depend on the tiling; both must equal the
reference mask.
"""
from types import SimpleNamespace

import torch

from flide.backbones.gemma4 import Gemma4
from flide.backbones.gemma4.graphs import GemmaGraphs
from flide.formats.base import Read
from flide.runtime.pack import Job, pack

W = 4  # a small window, so the window edge is covered too
# (prefix, branches): the second has one branch (the graph path runs it as one sequence), the third a branch longer
# than the window
TREES = [([1, 2, 3, 4, 5, 6], [[10, 11], [12], [13, 14, 15]]),
         ([7, 8, 9], [[20, 21, 22]]),
         ([30, 31, 32, 33, 34], [[40], [41, 42, 43, 44, 45, 46]])]


def _reference(ident, layer):
    """[T, T] boolean: does token q (row) see token k (column). ident[i] = (tree, branch or -1, index) or None."""
    T = len(ident)
    out = torch.zeros(T, T, dtype=torch.bool)
    for q, a in enumerate(ident):
        if a is None:
            out[q, q] = True   # padding sees itself only
            continue
        for k, b in enumerate(ident):
            if b is None or a[0] != b[0]:
                continue
            p = len(TREES[a[0]][0])
            qp = a[2] if a[1] < 0 else p + a[2]
            kp = b[2] if b[1] < 0 else p + b[2]
            if b[1] >= 0 and b[1] != a[1]:
                continue   # another branch
            if layer == "full_attention":
                out[q, k] = (a[1] < 0 and b[1] < 0) or (a[1] >= 0 and (b[1] < 0 or kp <= qp))
            else:
                out[q, k] = kp <= qp and qp - kp < W
    return out


def _visible(tok, ranges, window):
    """The kernel's two tests per (query, key): inside the query's key ranges, and shown by the metadata."""
    job, br, pos = (x.long() for x in tok[:3])
    blk = tok[3].long() if len(tok) == 4 else None
    T = job.shape[0]
    k = torch.arange(T)[None, :]
    r = ranges.long()
    in_range = ((k >= r[:, 0:1]) & (k < r[:, 1:2])) | ((k >= r[:, 2:3]) & (k < r[:, 3:4]))
    seen = pos[None, :] <= pos[:, None]
    if blk is not None:
        seen = seen | ((blk[None, :] == blk[:, None]) & (blk[:, None] >= 0))
    vis = (job[None, :] == job[:, None]) & ((br[None, :] == -1) | (br[None, :] == br[:, None])) & seen
    if window:
        vis = vis & ((pos[:, None] - pos[None, :]) < window)
    return in_range, vis


def _check(metas, ident):
    for layer, m in metas.items():
        in_range, vis = _visible(m.tok, m.ranges, W if layer == "sliding_attention" else 0)
        assert torch.equal(in_range, vis), f"{layer}: key ranges and metadata disagree"
        assert torch.equal(vis, _reference(ident, layer)), f"{layer}: differs from the reference mask"


def _packed(attention="prefix"):
    jobs = [Job(list(p), [list(s) for s in ss], [(Read(None),)] * len(ss), prefix_two_way=attention == "prefix")
            for p, ss in TREES]
    b = pack(jobs, "cpu")
    ident = []
    for t, (p, ss) in enumerate(TREES):
        ident += [(t, -1, i) for i in range(len(p))]
    for t, (p, ss) in enumerate(TREES):
        for bi, s in enumerate(ss):
            ident += [(t, bi, i) for i in range(len(s))]
    return b, (b.job.int(), b.branch.int(), b.positions.int()), ident


def test_prefix_attention_flags():
    assert Gemma4.prefix_attention and GemmaGraphs.prefix_attention
    assert "gemma4_unified_text" in Gemma4.model_types


def test_packed_prefix_meta_matches_reference():
    b, tok, ident = _packed()
    me = SimpleNamespace(config=SimpleNamespace(sliding_window=W), bidir=False, placeholders=set(),
                         _two_way_meta=lambda *a: Gemma4._two_way_meta(me, *a),
                         _block_meta=lambda *a: Gemma4._block_meta(me, *a))
    _check(Gemma4.attention_meta(me, tok, b.key_ranges, b.input_ids.numpy(), b), ident)


class _Stream:
    def synchronize(self):
        pass

    def record(self):
        pass


def _fake_graphs(n: int, bmax: int = 64):
    g = SimpleNamespace()
    g.bb = SimpleNamespace(bidir=False, placeholders={999}, pad_id=0, config=SimpleNamespace(sliding_window=W),
                           lm=SimpleNamespace(embed_tokens=lambda ids: torch.zeros(ids.shape[0], 1)))
    g.ids, g.raw, g.pos = (torch.zeros(n, dtype=torch.long) for _ in range(3))
    g.meta = torch.zeros(4, n, dtype=torch.int32)
    g.ranges = torch.zeros(n, 4, dtype=torch.int32)
    g.gblk = torch.zeros(n, dtype=torch.int32)
    g.reads = torch.zeros(bmax, dtype=torch.long)
    g.emb = torch.zeros(n, 1)
    g.host = torch.zeros(12, n, dtype=torch.long)
    g.host_reads = torch.zeros(bmax, dtype=torch.long)
    g.staged = _Stream()
    g.bmax = bmax
    g.nreq = GemmaGraphs.nreq
    g.nread = lambda L: GemmaGraphs.nread(g, L)
    return g


def test_graph_stage_prefix_matches_reference():
    """A one-branch request runs its prefix and branch as one sequence; the rest run as prefix + forked branches."""
    L = 64
    g = _fake_graphs(L)
    reqs = [(list(p), [list(s) for s in ss], None, "prefix") for p, ss in TREES]
    GemmaGraphs._stage(g, reqs, L)
    got = {}
    g._decode = lambda L_, full, sliding: got.update(full_attention=full, sliding_attention=sliding)
    GemmaGraphs._body(g, L, True)
    ident = []
    for t, (p, ss) in enumerate(TREES):
        ident += [(t, -1, i) for i in range(len(p))]
        if len(ss) == 1:
            ident += [(t, 0, i) for i in range(len(ss[0]))]
    for t, (p, ss) in enumerate(TREES):
        if len(ss) > 1:
            for bi, s in enumerate(ss):
                ident += [(t, bi, i) for i in range(len(s))]
    ident += [None] * (L - len(ident))
    _check(got, ident)
