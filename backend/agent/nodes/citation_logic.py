import re

# Phrases that signal the answer is a "not found" / information-absence reply.
# Deliberately about *missing information*, not about content (so "OSM-PINN does not
# use symmetric penalties" — a substantive claim — does NOT match).
_NEGATIVE_MARKERS = (
    "does not mention", "doesn't mention", "not mentioned", "no mention of",
    "does not discuss", "doesn't discuss", "does not contain", "doesn't contain",
    "does not provide", "doesn't provide", "does not specify", "doesn't specify",
    "does not appear", "doesn't appear", "not appear in",
    "no information", "no relevant information", "could not find", "couldn't find",
    "not found in", "is not mentioned", "isn't mentioned", "not contain any",
    "i cannot provide", "i can't provide", "i could not find", "unable to find",
    "context does not", "context doesn't", "documents do not", "documents don't",
)


def _is_negative_answer(answer: str) -> bool:
    """True when the answer is an information-absence reply ("X is not mentioned").

    Such answers must not carry citations or high confidence — the cited chunks
    do not *support* the claim, they merely failed to contain the requested fact.
    """
    low = answer.lower()
    return any(marker in low for marker in _NEGATIVE_MARKERS)


def _extract_cited_ids(answer: str, valid_ids: set[str]) -> list[str]:
    """Return the ordered, de-duplicated chunk indices the answer actually cites.

    Handles grouped citations — [3], [2, 4], [2, 4, 6] are all parsed. Any bracketed
    number that is NOT a real retrieved-chunk index is ignored (academic papers embed
    their own bibliography markers like "[7]" in the body text, which must never
    become citations). Returns indices sorted ascending by value.
    """
    out: list[str] = []
    for group in re.findall(r'\[([\d,\s]+)\]', answer):
        for num in re.findall(r'\d+', group):
            if num in valid_ids and num not in out:
                out.append(num)
    out.sort(key=int)
    return out


def _remap_citation_groups(answer: str, local_to_global: dict[str, str]) -> str:
    """Rewrite every [N] / [N, M, ...] group from local context indices to global IDs.

    Numbers that don't map to a retrieved chunk (the source's own references) are
    dropped from the group. A single regex pass with a replacement function avoids
    the partial-replacement bug where rewriting [1] before [10] corrupts "[10]".
    """
    def _sub(match: re.Match) -> str:
        mapped: list[str] = []
        for num in re.findall(r'\d+', match.group(1)):
            g = local_to_global.get(num)
            if g and g not in mapped:
                mapped.append(g)
        return "[" + ", ".join(mapped) + "]" if mapped else ""

    return re.sub(r'\[([\d,\s]+)\]', _sub, answer)
