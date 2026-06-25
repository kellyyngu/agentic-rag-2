import re

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(0-9])')

# Small stopword set so overlap scoring keys on meaningful terms, not filler.
_STOPWORDS = frozenset("""
a an the and or but of to in on at for with from by as is are was were be been
this that these those it its their they them which who what when where how why
we our you your i me my be can could would should may might will not no
into than then over under such only also more most other some any each both
""".split())


# PyPDF2 Symbol-font glyph codes → readable characters.
# These appear when a PDF embeds math/symbol characters in a non-standard font
# that PyPDF2 cannot decode (e.g. APA-style stats: r = .58, p < .001).
_PDF_SYMBOL_MAP: dict[str, str] = {
    '/H11005': '=',   # equals sign
    '/H11021': '<',   # less than
    '/H11022': '>',   # greater than
    '/H11349': '±',   # plus-minus
    '/H11011': '−',   # minus (en-dash)
    '/H11015': '≈',   # approximately equal
    '/H11003': '×',   # multiplication
    '/H11001': '+',   # plus (sometimes encoded)
    '/H11002': '−',   # minus (alternate)
    '/H11032': '′',   # prime
    '/H11018': '≤',   # less-than-or-equal
    '/H11019': '≥',   # greater-than-or-equal
    '/H9251':  'α',   # alpha
    '/H9252':  'β',   # beta
    '/H9254':  'δ',   # delta
    '/H9262':  'μ',   # mu
    '/H9268':  'σ',   # sigma
    '/H9273':  'χ',   # chi
    '/H9274':  'ψ',   # psi
}
# Compiled once — matches any known glyph code surrounded by word boundaries.
_PDF_SYMBOL_RE = re.compile(
    '|'.join(re.escape(k) for k in _PDF_SYMBOL_MAP),
)


def _clean_pdf_text(text: str) -> str:
    """Repair PDF extraction artifacts for readable display.

    Replaces Symbol-font glyph codes (e.g. /H11005 → =, /H11021 → <) that
    PyPDF2 emits when it cannot decode embedded math fonts. Also joins
    line-break hyphenation ("funda- mental" → "fundamental") while keeping
    real compound hyphens ("fixed-weight"), then collapses whitespace.
    """
    text = _PDF_SYMBOL_RE.sub(lambda m: _PDF_SYMBOL_MAP[m.group(0)], text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    return re.sub(r'\s+', ' ', text).strip()


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r'[a-z]+', text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _claim_for(local_id: str, answer: str, query: str) -> str:
    """The text a citation should be matched against: the answer sentence(s) that
    cite this source. Falls back to the user query if none can be isolated."""
    citing = []
    for sent in re.split(r'(?<=[.!?])\s+', answer):
        if any(local_id in re.findall(r'\d+', g) for g in re.findall(r'\[([\d,\s]+)\]', sent)):
            citing.append(sent)
    return " ".join(citing).strip() or query


def _evidence_snippet(content: str, claim: str, max_chars: int = 600) -> str:
    """Return the passage of `content` most relevant to `claim`, with neighbouring
    context — picked by keyword overlap, trimmed to whole sentences (never mid-word).
    """
    clean = _clean_pdf_text(content)
    sentences = [s.strip() for s in _SENT_SPLIT.split(clean) if s.strip()]
    if not sentences:
        return clean[:max_chars]

    kw = _keywords(claim)
    scores = [len(kw & _keywords(s)) for s in sentences] if kw else [0] * len(sentences)
    best = max(range(len(sentences)), key=lambda i: scores[i]) if any(scores) else 0

    # Window: best sentence plus one neighbour on each side (1–3 sentences).
    lo, hi = max(0, best - 1), min(len(sentences), best + 2)
    snippet = " ".join(sentences[lo:hi]).strip()

    if len(snippet) > max_chars:
        cut = snippet[:max_chars]
        boundary = max(cut.rfind('. '), cut.rfind('! '), cut.rfind('? '))
        snippet = (cut[:boundary + 1] if boundary > max_chars // 2 else cut.rsplit(' ', 1)[0]).strip()

    prefix = "… " if lo > 0 else ""
    suffix = " …" if hi < len(sentences) else ""
    return f"{prefix}{snippet}{suffix}".strip()
