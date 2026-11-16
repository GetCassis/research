"""Token alignment of decision points, continuation scoring, unit spans, head patching."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from _wbfixtures import model, sc, tok  # noqa: E402,F401  (fixtures; also sets sys.path for the imports below)
from wb.hf import base_model
from talkdemo import prompt as tdprompt
from wb import heads, steps
from wb.grads import integrated_gradients
from wb.model import Head, score_labels, tiny_model
from wb.tokens import (
    ALTS,
    PrefixError,
    align_decision,
    encode,
    render_prompt,
    special_token_ids,
    token_labels,
    unit_spans,
)


def full_logprobs(model, ids):
    """Reference: one plain forward through the model's own head (soft cap included)."""
    with torch.no_grad():
        return torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0].float(), -1)


def naive(model, al):
    out = {}
    for alt in ALTS:
        full = al.full(alt)
        lp = full_logprobs(model, full[:-1])
        out[alt] = sum(float(lp[p - 1, full[p]]) for p in al.label_positions(alt))
    return out["follows"], out["violates"]


def some_spec(sc):
    return sc.spec(rules="all", examples=sc.examples, broken={"ex_total": ["net"]})


# ---------------------------------------------------------------------------- prompts and spans


def test_prompt_is_the_builders_text_with_thinking_off(sc, tok):
    q = sc.questions[0]
    built = sc.build(q, some_spec(sc))
    p = tdprompt.build_prompt(sc.td, some_spec(sc), q)
    text = render_prompt(tok, built.messages)
    assert p.system in text and p.user in text  # byte-identical to what the builder returned
    assert text.endswith("<|turn>model\n<|channel>thought\n<channel|>")  # empty thought channel = thinking off
    enc = encode(tok, built.messages, q.sql)
    ids = tok.apply_chat_template(built.messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    ids = list(ids["input_ids"]) if not isinstance(ids, list) else ids
    assert enc.ids[: enc.prompt_len] == ids and enc.ids.count(tok.bos_token_id) == 1
    assert "".join(enc.pieces(enc.prompt_len)) == q.sql


def test_unit_spans_cover_their_text_in_order(sc, tok):
    q = sc.questions[0]
    built = sc.build(q, some_spec(sc))
    enc = encode(tok, built.messages, q.sql)
    spans = unit_spans(enc, built)
    assert list(spans) == [u.id for u in built.units]
    assert {"instructions", "schema", "rule.net", "rule.no_test", "rule.euros", "ex1.q", "ex5.sql", "question"} <= set(
        spans
    )
    end = 0
    for u in built.units:
        s, e = spans[u.id]
        assert end <= s < e <= enc.prompt_len  # in order, no overlap, inside the prompt
        covered = enc.text[enc.offsets[s][0] : enc.offsets[e - 1][1]]
        assert u.text in covered or covered in u.text  # a boundary token may belong to the previous unit
        assert len(covered) >= len(u.text) - 4
        end = e
    by_id = {u.id: u.text for u in built.units}
    assert "price_cents - refund_cents" not in by_id["ex2.sql"] and "price_cents - refund_cents" in by_id["ex1.sql"]
    labels = token_labels(enc, spans, special_token_ids(tok))
    assert (
        labels[0] == "special"
        and labels[enc.prompt_len - 1] == "special"
        and set(labels[enc.prompt_len :]) == {"output"}
    )
    assert labels.count("question") == spans["question"][1] - spans["question"][0]


def test_one_line_rule_is_one_unit(sc, tok):
    built = sc.build(sc.questions[0], sc.spec(rules="one_line"))
    assert [u.id for u in built.units if u.kind == "rule"] == ["rule.one_line"]
    unit_spans(encode(tok, built.messages), built)


# ---------------------------------------------------------------------------- decision alignment


def test_decisions_align_on_tokens(sc, tok):
    opts = {}
    for q in sc.questions:
        for spec in (sc.spec(), some_spec(sc)):
            built = sc.build(q, spec)
            ref = encode(tok, built.messages, q.sql)
            for c in sc.applies(q):
                al = steps.aligned(sc, tok, built, q, c, opts)
                d = sc.decision(q, c, opts["_extend"][(q.id, c)])
                assert al.cont["follows"][0] != al.cont["violates"][0]  # the decision token differs
                assert al.common[: al.prompt_len] == ref.ids[: ref.prompt_len]
                for alt, text in (("follows", d.follows), ("violates", d.violates)):
                    assert tok.decode(al.full(alt)[al.prompt_len :]) == d.prefix + text
                # the reference SQL runs through the decision token of `follows`
                n = len(al.common)
                assert ref.ids[: n + 1] == al.full("follows")[: n + 1]
                assert al.decision_pos == n - 1 and al.label_positions("follows")[0] == n


def test_alignment_follows_token_boundaries_not_characters(sc, tok):
    built = sc.build(sc.questions[0], sc.spec())
    d = SimpleNamespace(convention="x", prefix="SELECT COUNT(", follows="DISTINCT shop)", violates="*)")
    al = align_decision(tok, built, d)
    pieces = {alt: al.enc[alt].pieces(len(al.common)) for alt in ALTS}
    assert pieces["follows"][0] == "(" and pieces["violates"][0] == "(*)"  # "(*)" is one Gemma token
    assert al.decision_char - al.nominal_char == -1
    with pytest.raises(PrefixError):
        align_decision(
            tok,
            built,
            SimpleNamespace(convention="x", prefix="SELECT 1 WHERE a", follows=" AND", violates=" AND NOT b"),
        )


# ---------------------------------------------------------------------------- scoring


def test_continuation_scores_match_plain_forwards(sc, tok, model):
    q = sc.questions[0]
    jobs = []
    for spec in (sc.spec(), sc.spec(rules="all"), some_spec(sc)):
        built = sc.build(q, spec)
        jobs += [({"c": c}, steps.aligned(sc, tok, built, q, c, {})) for c in sc.applies(q)]
    rows = steps.score_decisions(model, tok, jobs, opts={"max_batch_tokens": 700}, label="t")  # several batches
    for row, (_, al) in zip(rows, jobs):
        f, v = naive(model, al)
        assert row["lp_follows"] == pytest.approx(f, abs=2e-3) and row["lp_violates"] == pytest.approx(v, abs=2e-3)
        assert row["diff"] == pytest.approx(f - v, abs=4e-3) and 0 < row["p_follow"] < 1
        assert len(row["top"]) == 5


def test_a_prefix_is_read_from_the_longer_sequence(model):
    rng = np.random.default_rng(0)
    a = [2] + rng.integers(5, 5000, 30).tolist()
    b = a + rng.integers(5, 5000, 8).tolist()
    c = a[:10] + [7, 8, 9]
    items = [(a, [20, 29, 30]), (b, [31, 38]), (c, [12]), (a[:12], [11])]
    got = score_labels(model, items, max_batch_tokens=64)
    for (full, pos), res in zip(items, got):
        lp = full_logprobs(model, full[:-1])
        np.testing.assert_allclose(res["logp"], [float(lp[p - 1, full[p]]) for p in pos], atol=2e-3)


def test_surprisal_tokens_are_the_reference_sql(sc, tok, model):
    out = steps.run_surprisal(sc, model, tok, {"questions": ["q_month", "n_top5"]})
    eot = steps.end_of_turn(tok)
    assert eot == "<turn|>"
    for q in out["questions"]:
        toks = out["tokens"][q["id"]]
        assert "".join(toks) == q["reference_sql"] + eot
        for c, i in out["decision_token"][q["id"]].items():
            assert 0 <= i < len(toks) - 1
        net = out["decision_token"][q["id"]]["net"]
        assert toks[net] == " -" and "".join(toks[:net]).endswith("SUM(price_cents")
    for r in out["rows"]:
        assert len(r["bits"]) == len(out["tokens"][r["question"]]) and min(r["bits"]) > 0
    assert out["summary"]["none"]["saved_bits"] == 0 and out["summary"]["ex5"]["context_tokens"] > 100


def test_perturbations_and_summary(sc, tok, model):
    names = [n for n, _, _ in sc.perturbations("net")]
    car = sc.carriers("net")
    assert car == ["ex_shop", "ex_total", "ex_day", "ex_top"]  # ex_count has no revenue
    assert len(names) == len(set(names)) and names.count("r0|none") == 1
    assert sum(n.startswith("r0|e") for n in names) == 2 ** len(car)
    assert sum(n.startswith("r+|e") for n in names) == 1 + 4 + 4 + 1 + 1  # sizes 0, 1, n-1, n and one chain
    out = steps.run_perturb(sc, model, tok, {"conventions": ["net"], "questions": ["q_month", "n_top5"]})
    s = out["summary"]["net|rule0"]
    assert s["carriers"] == [1, 2, 3, 5] and set(s["dose"]) == {0, 1, 2, 3, 4}
    assert set(s["break_one"]) == set(s["restore_one"]) == {1, 2, 3, 5}
    assert s["near"]["break_nearest"]["n"] == 1 and s["near"]["break_other"]["n"] == 3  # n_top5 is near example 5
    assert set(out["summary"]) == {"net|rule0", "net|rule+", "net|rule-"}


def test_grid_has_every_condition_and_decision(sc, tok, model):
    out = steps.run_grid(sc, model, tok, {})
    n_dec = sum(len(sc.applies(q)) for q in sc.questions)
    assert len(out["rows"]) == len(out["conditions"]) * n_dec
    names = [c["name"] for c in out["conditions"]]
    assert names[:7] == ["none", "rule_one_line", "rules_all", "ex1", "ex3", "ex5", "rules+examples"]
    assert sum(n.startswith("conflict:rule_") for n in names) == 2 * len(sc.conventions)
    assert [n for n in names if n.startswith("conflict:one_line")] == [  # only what the one-line rule states
        "conflict:one_line_clean+examples_broken:net",
        "conflict:one_line_broken+examples_clean:net",
    ]
    assert out["meta"]["scenario_version"]["hash"] == sc.version["hash"] and not out["meta"]["over_window"]
    assert "enable_thinking=False" in out["meta"]["thinking"]


def test_picture_conditions_can_join_the_grid(sc):
    conds = steps.conditions_for(sc, {"picture": "net", "picture_break": 2, "conditions": ["none", "picture:"]})
    names = [c.name for c in conds]
    assert names[0] == "none" and "picture:one_example_broken+rules" in names and "ex5" not in names
    spec = dict((c.name, c.spec) for c in conds)["picture:one_example_broken"]
    assert [sorted(u.corrupt) for u in spec.examples] == [[], ["net"], [], [], []]  # example 2, and only it
    other = dict((c.name, c.spec) for c in conds)["picture:other_example_broken"]
    assert [sorted(u.corrupt) for u in other.examples] == [["net"], [], [], [], []]
    assert sc.picture_facts("net", 2) == {
        "broken_example": 2,
        "broken_example_id": "ex_total",
        "other_example": 1,
        "other_example_id": "ex_shop",
    }
    with pytest.raises(ValueError):
        sc.picture_contexts("net", 4)  # example 4 counts sales: there is no revenue to break


def test_sampled_check_runs_and_checkers_agree_with_the_reference(sc, tok, model):
    q = sc.question("q_month")
    assert sc.check(q, q.sql) == {"net": True, "no_test": True, "euros": True}
    assert sc.check(q, "```sql\n" + sc.td.variant(q, ["net"]) + "\n```")["net"] is False
    assert sc.check(q, "I cannot answer that.") == {"net": None, "no_test": None, "euros": None}
    out = steps.run_sample(
        sc, model, tok, {"n": 3, "max_new_tokens": 6, "questions": ["q_month"], "conditions": ["none", "ex5"]}
    )
    assert len(out["cells"]) == 2 and all(len(c["texts"]) == 3 for c in out["cells"])
    assert len(out["pairs"]) == 2 * 3 and all(0 < p["p_follow"] < 1 for p in out["pairs"])


# ---------------------------------------------------------------------------- head patching


def _pair(sc, tok, q_id="q_month", conv="net"):
    q = sc.question(q_id)
    ctx = dict(sc.picture_contexts(conv))
    return {n: steps.aligned(sc, tok, sc.build(q, ctx[n]), q, conv, {}) for n in ("examples_clean", "examples_broken")}


def _state(model):
    mods = heads.attn_modules(model)
    return (
        [len(m.o_proj._forward_pre_hooks) + len(m._forward_hooks) for m in mods],
        model.config._attn_implementation,
        [p.requires_grad for p in model.parameters()],
    )


def test_patching_leaves_the_model_unchanged(sc, tok, model):
    al = _pair(sc, tok)
    ids = al["examples_clean"].full("follows")
    before, state = full_logprobs(model, ids), _state(model)
    res = heads.patch_pair(model, tok, al["examples_clean"], al["examples_broken"], Head(model), {})
    heads.attention_rows(model, al["examples_clean"].common, al["examples_clean"].decision_pos)
    assert _state(model) == state and state[0] == [0] * len(state[0]) and state[1] == "sdpa"
    torch.testing.assert_close(full_logprobs(model, ids), before)
    assert res["modes"]["last"]["denoise"]["noise_floor"] < 1e-3


def test_cached_tail_matches_plain_forwards(sc, tok, model):
    al = _pair(sc, tok)
    head = Head(model)
    for name, a in al.items():
        f, v = naive(model, a)
        for width in (1, 40):
            _, m = heads.capture(heads.Tail(model, a, width), head)
            np.testing.assert_allclose(m, [f, v, f - v], atol=3e-3)


def test_patching_every_head_over_the_tail_gives_the_donor(sc, tok, model):
    """The tail then reads the donor's context only: the whole effect, in both directions; a head
    patched with its own activations changes nothing; and the effects are not additive."""
    al = _pair(sc, tok)
    head = Head(model)
    clean, corrupt = al["examples_clean"], al["examples_broken"]
    n_tail = heads.common_suffix(clean.common, corrupt.common)
    assert 10 < n_tail < len(corrupt.common) - 50  # after the last example's edit, before the context
    tails = {k: heads.Tail(model, a, n_tail) for k, a in (("clean", clean), ("corrupt", corrupt))}
    acts, base = {}, {}
    for k, t in tails.items():
        acts[k], base[k] = heads.capture(t, head)
    assert abs(base["clean"][2] - base["corrupt"][2]) > 1e-3  # the two prompts do differ for the model
    for rec, don in (("corrupt", "clean"), ("clean", "corrupt")):
        got = heads.patched(tails[rec], acts[don], [("all", "all"), ([], "all")], head)
        np.testing.assert_allclose(got[0], base[don], atol=3e-3)
        np.testing.assert_allclose(got[1], base[rec], atol=1e-4)
        own = heads.patched(tails[rec], acts[rec], [("all", "all"), ([(2, 1)], "all")], head)
        np.testing.assert_allclose(own, np.stack([base[rec]] * 2), atol=1e-4)


def test_single_head_patch_matches_an_independent_full_forward(sc, tok, model):
    """The batched, cached, masked patch == a plain forward of the whole recipient prompt with one
    hand-written hook."""
    al = _pair(sc, tok)
    head = Head(model)
    clean, corrupt = al["examples_clean"], al["examples_broken"]
    n_tail = heads.common_suffix(clean.common, corrupt.common)
    layer, h = 2, 3  # a full-attention layer (one KV head, bigger head_dim)
    mods = heads.attn_modules(model)
    d = mods[layer].head_dim
    assert mods[layer].layer_type == "full_attention" and d == 32
    for where, width in ((1, 1), (n_tail, n_tail)):
        tails = {k: heads.Tail(model, a, width) for k, a in (("clean", clean), ("corrupt", corrupt))}
        donor, _ = heads.capture(tails["clean"], head)
        pos = [width - 1] if width == 1 else "all"
        got = heads.patched(tails["corrupt"], donor, [([(layer, h)], pos), ([(0, 0), (layer, h)], pos)], head)[0]

        store = {}
        hk = mods[layer].o_proj.register_forward_pre_hook(lambda m, a: store.__setitem__("x", a[0].clone()))
        ref = {}
        try:
            for alt in ALTS:
                with torch.no_grad():
                    model(input_ids=torch.tensor([clean.sequence(alt)]))
                ref[alt] = store["x"][0]
        finally:
            hk.remove()
        want = {}
        for alt in ALTS:
            n_c, n_k = len(clean.sequence(alt)), len(corrupt.sequence(alt))
            k = where + len(clean.cont[alt]) - 1 if width > 1 else None  # tail: every tail and continuation position

            def patch(m, a, alt=alt, n_c=n_c, n_k=n_k, k=k):
                x = a[0].clone()
                if k is None:  # last: the decision position only
                    p_k, p_c = len(corrupt.common) - 1, len(clean.common) - 1
                    x[0, p_k, h * d : (h + 1) * d] = ref[alt][p_c, h * d : (h + 1) * d]
                else:
                    x[0, n_k - k :, h * d : (h + 1) * d] = ref[alt][n_c - k :, h * d : (h + 1) * d]
                return (x,)

            hk = mods[layer].o_proj.register_forward_pre_hook(patch)
            try:
                full = corrupt.full(alt)
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=torch.tensor([full[:-1]])).logits[0].float(), -1)
            finally:
                hk.remove()
            want[alt] = sum(float(lp[p - 1, full[p]]) for p in corrupt.label_positions(alt))
        np.testing.assert_allclose(got[:2], [want["follows"], want["violates"]], atol=3e-3)


def test_tail_is_right_beyond_the_sliding_window(sc, tok):
    """Prompt far longer than the window: the cached prefix keeps every position and the sliding
    mask comes from the config, as in a plain forward."""
    small = tiny_model(tok, sliding_window=32)
    al = _pair(sc, tok)["examples_clean"]
    assert len(al.common) > 300
    f, v = naive(small, al)
    _, m = heads.capture(heads.Tail(small, al, 50), Head(small))
    np.testing.assert_allclose(m, [f, v, f - v], atol=3e-3)
    att = heads.attention_rows(small, al.common, al.decision_pos)
    sliding = [i for i, m_ in enumerate(heads.attn_modules(small)) if m_.is_sliding]
    assert (
        np.all(att[sliding][:, :, : len(al.common) - 32] == 0) and att[2, :, :100].sum() > 0
    )  # full layers see it all


def test_attention_rows_are_the_models(sc, tok, model):
    q = sc.question("q_month")
    built = sc.build(q, some_spec(sc))
    al = steps.aligned(sc, tok, built, q, "net", {})
    att = heads.attention_rows(model, al.common, al.decision_pos)
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    assert att.shape == (L, H, len(al.common))
    np.testing.assert_allclose(att.sum(-1), 1.0, atol=1e-4)
    with heads.attn_implementation(model, "eager"), torch.no_grad():
        ref = base_model(model)(input_ids=torch.tensor([al.common]), output_attentions=True).attentions
    np.testing.assert_allclose(att, np.stack([a[0, :, al.decision_pos].numpy() for a in ref]), atol=1e-5)
    enc = al.enc["follows"]
    labels = token_labels(enc, unit_spans(enc, built), special_token_ids(tok))[: len(al.common)]
    order = [u.id for u in built.units] + ["other", "special", "output"]
    np.testing.assert_allclose(heads.unit_mass(att, labels, order).sum(-1), 1.0, atol=1e-4)


def test_heads_and_attention_steps(sc, tok, model, tmp_path):
    import json

    opts = {"conventions": ["net"], "questions_per_convention": 2}
    out = heads.run_heads(sc, model, tok, dict(opts), tmp_path)
    assert [(r["convention"], r["question"]) for r in out["pairs"]] == [
        ("net", "q_best"),
        ("net", "n_top5"),
    ]  # earliest decision first
    L, H = model.config.num_hidden_layers, model.config.num_attention_heads
    for r in out["pairs"]:
        for mode, directions in (("last", {"denoise", "noise"}), ("tail", {"denoise"})):
            assert set(r["modes"][mode]) == directions
            eff = np.array(r["modes"][mode]["denoise"]["effect"])
            assert eff.shape == (L, H)
        t = r["modes"]["tail"]["denoise"]
        assert t["all_heads"] == pytest.approx(r["total"], abs=5e-3)  # the whole effect goes through the tail
        assert len(t["per_position"]) == r["tail_len"] and t["per_position"][-1]["rel"] == 0
        assert t["cumulative"][-1]["k"] < L * H
    assert out["ranking"]["net"]["n_pairs"] == 2
    (tmp_path / "heads.json").write_text(json.dumps(out))
    att = heads.run_attention(sc, model, tok, {**opts, "top_heads": 3}, tmp_path)
    cell = att["cells"][0]
    assert (
        cell["selection"] == "leave this question out"
        and cell["selected_on"] == ["n_top5"]
        and len(cell["selected_heads"]) == 3
    )
    assert set(cell["contexts"]) == {
        "examples_clean",
        "one_example_broken",
        "one_example_broken+rules",
        "other_example_broken",
        "rule_only",
        "rule_wrong_only",
        "rule_vs_examples",
        "rule_wrong_vs_examples",
        "one_line_vs_examples",
        "one_line_wrong_vs_examples",
        "examples_broken",
    }
    ctx = cell["contexts"]["one_example_broken"]
    assert len(ctx["tokens"]) == len(ctx["labels"]) == ctx["n_tokens"] == len(ctx["token_maps"]["selected_heads"])
    assert sum(ctx["unit_mass"]["all_heads_all_layers"]) == pytest.approx(1.0, abs=1e-3)
    npz = np.load(tmp_path / "attention.npz")
    assert npz["net|q_best|rule_only"].shape[:2] == (L, H)

    # the centrepiece: which heads carry the effect of ONE changed example (here example 2)
    focus = {**opts, "corrupt_context": "one_example_broken", "picture_break": 2, "questions": ["q_best"]}
    one = heads.run_heads(sc, model, tok, dict(focus), tmp_path)["pairs"][0]
    assert one["corrupt_context"] == "one_example_broken" and one["broken_example"] == 2
    assert one["modes"]["tail"]["denoise"]["all_heads"] == pytest.approx(one["total"], abs=5e-3)
    assert one["tail_len"] > out["pairs"][0]["tail_len"]  # the last edit is further from the question
    few = heads.run_attention(sc, model, tok, {**focus, "contexts": ["one_example_broken", "rule_only"]}, tmp_path)
    assert list(few["cells"][0]["contexts"]) == ["one_example_broken", "rule_only"]
    assert few["cells"][0]["broken_example"] == 2


# ---------------------------------------------------------------------------- gradients


def test_integrated_gradients_are_complete_and_leave_the_model_alone(sc, tok, model):
    q = sc.question("q_month")
    built = sc.build(q, sc.spec(examples=sc.examples[:2]))
    state = _state(model)
    filler = tok(" ", add_special_tokens=False)["input_ids"][0]
    for conv in ("net", "no_test"):  # one-token and several-token continuations
        al = steps.aligned(sc, tok, built, q, conv, {})
        enc = al.enc["follows"]
        labels = token_labels(enc, unit_spans(enc, built), special_token_ids(tok))[: len(al.common)]
        keep = [lab == "special" for lab in labels]
        # a random-weight model is rough along the path: many steps (the real run reports its own gap)
        res = integrated_gradients(model, al, Head(model), keep=keep, steps=256, baseline_id=filler)
        f, v = naive(model, al)
        assert res["m"] == pytest.approx(f - v, abs=3e-3)
        assert abs(res["completeness_gap"]) < 0.03
        assert np.all(res["ig"][np.array(keep)] == 0) and np.abs(res["ig"][~np.array(keep)]).sum() > 0
        assert res["ig"].shape == res["grad_x_input"].shape == (len(al.common),)
    assert _state(model) == state


def test_convention_tokens_are_the_ones_an_edit_touches(sc, tok):
    from wb.grads import convention_tokens

    q = sc.question("q_best")
    for name, spec in sc.picture_contexts("net", 2):
        built = sc.build(q, spec)
        enc = encode(tok, built.messages, q.sql)
        spans = unit_spans(enc, built)
        got = convention_tokens(sc, built, enc, spans, "net")
        text = lambda uid: "".join(enc.pieces(*spans[uid])[i - spans[uid][0]] for i in got if spans[uid][0] <= i < spans[uid][1])  # noqa: E731
        if name == "examples_clean":
            assert text("ex1.sql").strip() == "- refund_cents" and text("ex4.sql") == ""  # example 4 has no revenue
            assert not any(spans["question"][0] <= i < spans["question"][1] for i in got)
        if name == "one_example_broken":
            assert text("ex2.sql") == "price_cents)" and text("ex1.sql").strip() == "- refund_cents"  # the absence, by its neighbours
        if name == "rule_only":
            assert got == list(range(*spans["rule.net"]))
        if name == "one_line_wrong_vs_examples":
            assert set(range(*spans["rule.one_line"])) <= set(got)
