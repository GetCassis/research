"""Prompt rendering, unit spans and decision-token alignment (tokenizer only, no model).

Rendering: the builder's chat messages go through the HF chat template of the model, with the
generation prompt and `enable_thinking=False`. For Gemma 4 the prompt then ends with an EMPTY thought
channel, `<|turn>model\\n<|channel>thought\\n<channel|>`: this is how its template turns thinking off,
so what the model writes next is the answer itself and the reference SQL can be teacher-forced right
after the prompt. `output_prefix` (default empty) is text put between the prompt and the SQL, for a
model that always opens a code fence.

Decision alignment is done on TOKENS, not characters. The scenario gives a character prefix of the
reference SQL and two continuations; the tokenizer does not have to cut at that character (Gemma:
`COUNT(DISTINCT` is `COUNT`, `(`, `DIST`, `INCT` but `COUNT(*)` is `COUNT`, `(*)`). So both
alternatives are tokenised whole (prompt + prefix + continuation), the longest common token prefix
is the shared context, the first token after it is the decision token, and each alternative's
continuation is its tokens from there to the end of its text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ALTS = ("follows", "violates")


class PrefixError(ValueError):
    """One continuation is a token prefix of the other: the caller extends them and retries."""


def render_prompt(tok, messages: list[dict], enable_thinking: bool = False) -> str:
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )


@dataclass
class Encoded:
    text: str  # prompt + output_prefix + output
    ids: list[int]
    offsets: list[tuple[int, int]]  # char span of each token in `text`
    prompt_chars: int  # len(rendered prompt)
    prompt_len: int  # tokens of the rendered prompt
    output_start: int  # char where the SQL starts (after output_prefix)

    def pieces(self, lo: int = 0, hi: int | None = None) -> list[str]:
        return [self.text[a:b] for a, b in self.offsets[lo:hi]]


def encode(tok, messages: list[dict], output: str = "", output_prefix: str = "") -> Encoded:
    prompt = render_prompt(tok, messages)
    text = prompt + output_prefix + output
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)  # the template wrote <bos>
    ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    prompt_len = sum(1 for a, _ in offsets if a < len(prompt))
    if prompt_len and offsets[prompt_len - 1][1] != len(prompt):
        raise ValueError("a token straddles the end of the prompt: the template must end on a special token")
    return Encoded(text, ids, offsets, len(prompt), prompt_len, len(prompt) + len(output_prefix))


def token_span(offsets: list[tuple[int, int]], cs: int, ce: int) -> tuple[int, int]:
    """[first, last + 1) of the tokens that overlap chars [cs, ce)."""
    toks = [i for i, (a, b) in enumerate(offsets) if a < ce and b > cs and b > a]
    if not toks:
        raise ValueError(f"no token covers chars [{cs}, {ce})")
    return toks[0], toks[-1] + 1


def unit_spans(enc: Encoded, built) -> dict[str, tuple[int, int]]:
    """Token span of each unit: its text is searched verbatim in the rendered prompt, in order.

    A token shared by two adjacent units (a merge across the boundary) goes to the first, so the
    spans never overlap."""
    spans: dict[str, tuple[int, int]] = {}
    cursor, tok_floor = 0, 0
    prompt = enc.text[: enc.prompt_chars]
    for u in built.units:
        cs = prompt.find(u.text, cursor)
        if cs < 0:
            raise ValueError(f"unit {u.id!r} not found verbatim in the rendered prompt after char {cursor}")
        s, e = token_span(enc.offsets, cs, cs + len(u.text))
        s = max(s, tok_floor)
        if s >= e:
            raise ValueError(f"unit {u.id!r} has no token of its own")
        spans[u.id] = (s, e)
        cursor, tok_floor = cs + len(u.text), e
    return spans


def token_labels(enc: Encoded, spans: dict[str, tuple[int, int]], special_ids: set[int]) -> list[str]:
    """One label per token, a partition of the sequence: a unit id, `special` (template tokens),
    `output` (the SQL written so far) or `other` (prompt text outside every unit: headings, labels)."""
    labels = ["other"] * len(enc.ids)
    for i, t in enumerate(enc.ids):
        if i >= enc.prompt_len:
            labels[i] = "output"
        elif t in special_ids:
            labels[i] = "special"
    for uid, (s, e) in spans.items():
        for i in range(s, e):
            labels[i] = uid
    return labels


@dataclass
class Aligned:
    """One decision on one prompt, in tokens."""

    common: list[int]  # prompt + the shared SQL prefix: the last token's next-token distribution decides
    prompt_len: int
    cont: dict[str, list[int]]  # alternative -> its continuation tokens (first one = the decision token)
    enc: dict[str, Encoded] = field(repr=False, default_factory=dict)
    nominal_char: int = 0  # where the scenario put the decision, as a char offset in the SQL
    decision_char: int = 0  # where the first differing token starts, same coordinates

    @property
    def decision_pos(self) -> int:
        """Index of the query position: the token whose output predicts the decision token."""
        return len(self.common) - 1

    def sequence(self, alt: str) -> list[int]:
        """Input ids that teacher-force alternative `alt`: every continuation token but the last."""
        return self.common + self.cont[alt][:-1]

    def label_positions(self, alt: str) -> list[int]:
        """Positions (in `full(alt)`) of the continuation tokens; logits at p - 1 predict token p."""
        n = len(self.common)
        return list(range(n, n + len(self.cont[alt])))

    def full(self, alt: str) -> list[int]:
        return self.common + self.cont[alt]

    def describe(self, tok) -> dict:
        """What was aligned, for the JSON and the `info` step."""
        out = {
            "n_prompt_tokens": self.prompt_len,
            "n_common_tokens": len(self.common),
            "shift_chars": self.decision_char - self.nominal_char,
            "shared_sql_tail": "".join(
                self.enc["follows"].pieces(max(self.prompt_len, len(self.common) - 6), len(self.common))
            ),
        }
        for alt in ALTS:
            out[alt] = self.enc[alt].pieces(len(self.common), None)
        return out


def align_decision(tok, built, decision, output_prefix: str = "") -> Aligned:
    """`built`: anything with `.messages`; `decision`: anything with `.prefix`, `.follows`, `.violates`."""
    enc = {
        "follows": encode(tok, built.messages, decision.prefix + decision.follows, output_prefix),
        "violates": encode(tok, built.messages, decision.prefix + decision.violates, output_prefix),
    }
    a, b = enc["follows"].ids, enc["violates"].ids
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    plen = enc["follows"].prompt_len
    if n < plen:
        raise ValueError("the two alternatives differ inside the prompt")
    if n == len(a) or n == len(b):
        raise PrefixError(
            f"{decision.convention}: one continuation is a token prefix of the other "
            f"({decision.follows!r} / {decision.violates!r}): extend the shorter one so the two differ"
        )
    start = enc["follows"].output_start
    return Aligned(
        common=a[:n],
        prompt_len=plen,
        cont={"follows": a[n:], "violates": b[n:]},
        enc=enc,
        nominal_char=len(decision.prefix),
        decision_char=enc["follows"].offsets[n][0] - start,
    )


def special_token_ids(tok) -> set[int]:
    return set(tok.all_special_ids) | {i for i in tok.get_added_vocab().values()}
