"""Corpus Processing & Chunking module for Agentic Code Intelligence.

Role 2: Corpus Processing & Chunking.
Provides:
- Pydantic model `SnippetSchema` (+ `SnippetMetadata`) as the typed contract
  between corpus processing and dense retrieval.
- `StructuralExtractor`: `ast`-based static analysis recovering import
  statements, top-level function/class definitions with their docstrings, and a
  boilerplate-stripped "core logic" region (argparse / input-parsing scaffolding
  and `if __name__ == "__main__"` guards are removed).
- `LongDocChunker`: recursive structural chunking for documents whose estimated
  token count exceeds a configurable budget. Splits on function/class
  boundaries first, then blank-line paragraphs, then physical lines.
- `ChunkCache`: SQLite-backed memoization of chunking results keyed by a
  whitespace-normalized content hash, so re-running over an unchanged corpus
  skips re-processing entirely.
- `aggregate_scores`: max-pooling of per-chunk similarity scores back to one
  score per document, called from `encoder.py` before MTEB ranks documents.

Everything here is deterministic static analysis and hashing: no tokenizer, no
network, no GPU, and no generative calls.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from typing import Any, Dict, List, Literal, NamedTuple, Sequence, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

UnitKind = Literal["function", "async_function", "class"]

DEFAULT_MAX_TOKENS = 256
TOKENS_PER_WORD = 1.3
SUMMARY_MAX_LINES = 15
DEFAULT_CACHE_PATH = os.path.join(".cache", "chunker_cache.sqlite3")

_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t]*\n")
_BLANK_RUN_RE = re.compile(r"\n{3,}")

_PARSE_ERRORS = (SyntaxError, ValueError, TypeError, RecursionError, MemoryError)

_ARG_PARSER_TAIL_NAMES: Set[str] = {
    "ArgumentParser",
    "add_argument",
    "add_parser",
    "add_subparsers",
    "add_argument_group",
    "add_mutually_exclusive_group",
    "parse_args",
    "parse_known_args",
    "set_defaults",
}
_PARSE_ARGS_TAIL_NAMES: Set[str] = {"parse_args", "parse_known_args"}
_PROMPT_INPUT_NAMES: Set[str] = {"input", "raw_input"}


def estimate_tokens(text: str, tokens_per_word: float = TOKENS_PER_WORD) -> int:
    """Estimate the token count of `text` without loading a tokenizer.

    Uses a whitespace split scaled by `tokens_per_word` (1.3 by default), which
    is a close-enough proxy for BPE tokenizers on source code and keeps the
    corpus stage dependency-free and CPU-only.

    Args:
        text: Text to measure.
        tokens_per_word: Estimated tokens emitted per whitespace-separated word.

    Returns:
        Estimated token count (0 for empty input).
    """
    if not text:
        return 0
    return int(len(text.split()) * tokens_per_word)


def compute_content_hash(source: str) -> str:
    """Hash source code by its normalized AST dump, ignoring whitespace/comments.

    `ast.dump` with attributes excluded drops line numbers, comments, and all
    incidental formatting, so two reformattings of the same code hash equally.
    Non-parseable text falls back to a whitespace-normalized hash, which still
    makes the hash insensitive to indentation and line wrapping.

    This is the identity carried by `SnippetSchema.id`. It costs an extra parse
    plus AST dump per call, so use `compute_text_hash` when only an exact cache
    key is needed.

    Args:
        source: Source text to hash.

    Returns:
        Hex sha256 digest.
    """
    return hashlib.sha256(
        _normalise_for_hash(source).encode("utf-8", errors="surrogatepass")
    ).hexdigest()


def compute_text_hash(source: str) -> str:
    """Hash whitespace-normalized text: cheap, exact, and comment sensitive.

    Used as the `ChunkCache` lookup key. Identical keys imply identical chunk
    output, so a hit is always safe, and the key costs one C-level whitespace
    split instead of an AST parse plus dump. Formatting-only edits still hit;
    comment-only edits miss and are simply re-chunked.

    Args:
        source: Source text to hash.

    Returns:
        Hex sha256 digest.
    """
    normalised = " ".join(source.split()) if source else ""
    return hashlib.sha256(normalised.encode("utf-8", errors="surrogatepass")).hexdigest()


def _normalise_for_hash(source: str) -> str:
    """Build the canonical string that `compute_content_hash` digests."""
    if not source:
        return ""
    try:
        tree = ast.parse(source)
        return ast.dump(tree, annotate_fields=True, include_attributes=False)
    except _PARSE_ERRORS:
        return " ".join(source.split())


class CodeUnit(BaseModel):
    """A single top-level function, async function, or class definition."""

    name: str = Field(..., description="Name of the function or class.")
    kind: UnitKind = Field(..., description="Type of the top-level definition.")
    signature: str = Field(
        default="", description="Declaration header without decorators or body."
    )
    docstring: str | None = Field(
        default=None, description="Dedented docstring of the definition, if present."
    )
    source: str = Field(default="", description="Complete text of the definition.")
    lineno: int = Field(default=1, description="1-based start line in the document.")
    end_lineno: int = Field(default=1, description="1-based end line in the document.")


class StructuralExtract(BaseModel):
    """Output of `StructuralExtractor` for a single document."""

    imports: List[str] = Field(
        default_factory=list,
        description="Import statements recovered verbatim, in source order.",
    )
    units: List[CodeUnit] = Field(
        default_factory=list,
        description="Top-level function/class definitions with docstrings.",
    )
    core_logic: str = Field(
        default="",
        description=(
            "Module-level code with imports, definitions, and CLI scaffolding stripped. "
            "Definitions are recovered separately in `units`, so nothing is duplicated."
        ),
    )
    module_docstring: str | None = Field(
        default=None, description="Dedented module docstring, if present."
    )
    parse_failed: bool = Field(
        default=False,
        description="True when the source could not be parsed and raw text was kept.",
    )


class SnippetMetadata(BaseModel):
    """Provenance and positioning metadata carried by every `SnippetSchema`."""

    chunk_index: int = Field(..., ge=0, description="0-based position within the document.")
    num_chunks: int = Field(..., ge=1, description="Total chunks produced for the document.")
    has_imports: List[str] = Field(
        default_factory=list,
        description="Import statements visible in the parent document.",
    )
    name: str = Field(default="", description="Name of the originating definition.")
    kind: str = Field(default="module", description="Definition kind or 'module'/'paragraph'.")
    signature: str = Field(default="", description="Signature of the originating definition.")
    docstring: str | None = Field(default=None, description="Docstring of the originating definition.")
    parse_failed: bool = Field(
        default=False, description="True when the parent document failed to parse."
    )
    structural: bool = Field(
        default=False, description="True when the snippet came from AST structure."
    )
    estimated_tokens: int = Field(
        default=0, description="Estimated token count of the full view."
    )


class SnippetSchema(BaseModel):
    """Standardized embeddable snippet passed from corpus processing to retrieval.

    `summary_view` is the string that gets embedded (signature + docstring +
    first lines). `full_view` holds the complete chunk text and is deliberately
    left unembedded, reserved for a future reranker.
    """

    id: str = Field(
        ...,
        description="sha256 content hash of the normalized AST dump (whitespace/comment invariant).",
    )
    doc_id: str = Field(..., description="Identifier of the original corpus document.")
    summary_view: str = Field(..., description="Compact view that is embedded.")
    full_view: str = Field(..., description="Complete chunk text, never embedded.")
    metadata: SnippetMetadata = Field(
        ..., description="Chunk position, imports, and structural provenance."
    )


class _Piece(NamedTuple):
    """Intermediate chunk candidate before it becomes a `SnippetSchema`."""

    header: str
    body: str
    name: str = ""
    kind: str = "module"
    signature: str = ""
    docstring: str | None = None
    truncate: bool = True


def _dotted_name(node: ast.AST) -> str:
    """Render an attribute/name expression as a dotted string (best effort)."""
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _call_tail(node: ast.AST) -> str:
    """Return the final attribute name of a call target (e.g. 'add_argument')."""
    return _dotted_name(node).split(".")[-1]


def _names_in(node: ast.AST) -> Set[str]:
    """Collect every identifier referenced inside `node`."""
    found: Set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            found.add(child.id)
        elif isinstance(child, ast.Attribute):
            found.add(child.attr)
    return found


def _is_str_constant(node: ast.AST, value: str) -> bool:
    """True when `node` is the literal `value`."""
    return isinstance(node, ast.Constant) and node.value == value


def _is_main_guard(node: ast.If) -> bool:
    """True for `if __name__ == "__main__":` guards (in either operand order)."""
    test = node.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
        return False
    if not isinstance(test.ops[0], (ast.Eq, ast.NotEq)):
        return False
    left, right = test.left, test.comparators[0]
    for name_node, const_node in ((left, right), (right, left)):
        if (
            isinstance(name_node, ast.Name)
            and name_node.id == "__name__"
            and _is_str_constant(const_node, "__main__")
        ):
            return True
    return False


class _BoilerplateScanner:
    """Locates argparse / input-parsing scaffolding at module level."""

    def __init__(self, tree: ast.Module) -> None:
        self._scaffold_names: Set[str] = self._collect_bound_names(tree.body)

    @classmethod
    def _collect_bound_names(cls, body: Sequence[ast.stmt]) -> Set[str]:
        """Names bound by argparse construction or `parse_args()` calls.

        Only module level and `__main__` guard bodies are scanned: scaffolding
        deeper inside a function is real program logic, not CLI boilerplate.
        """
        names: Set[str] = set()
        for stmt in body:
            if isinstance(stmt, ast.If) and _is_main_guard(stmt):
                names |= cls._collect_bound_names(stmt.body)
                names |= cls._collect_bound_names(stmt.orelse)
                continue
            if not isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            value = stmt.value
            if not isinstance(value, ast.Call):
                continue
            if _call_tail(value.func) not in _ARG_PARSER_TAIL_NAMES:
                continue
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        return names

    def _is_argparse_call(self, node: ast.Call) -> bool:
        """True for calls into the argparse builder API."""
        return _call_tail(node.func) in _ARG_PARSER_TAIL_NAMES

    def _is_prompt_call(self, node: ast.Call) -> bool:
        """True for interactive `input()`-style prompts."""
        return isinstance(node.func, ast.Name) and node.func.id in _PROMPT_INPUT_NAMES

    def is_scaffolding(self, stmt: ast.stmt) -> bool:
        """True when a module-level statement is CLI/input scaffolding."""
        if isinstance(stmt, ast.If):
            return _is_main_guard(stmt) or bool(_names_in(stmt.test) & self._scaffold_names)

        if isinstance(stmt, ast.Expr):
            return isinstance(stmt.value, ast.Call) and (
                self._is_argparse_call(stmt.value) or self._is_prompt_call(stmt.value)
            )

        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            if any(isinstance(t, ast.Name) and t.id in self._scaffold_names for t in targets):
                return True
            if not isinstance(stmt.value, ast.Call):
                return False
            tail = _call_tail(stmt.value.func)
            return (
                tail in _PARSE_ARGS_TAIL_NAMES
                or tail == "ArgumentParser"
                or self._is_prompt_call(stmt.value)
            )

        return False


class StructuralExtractor:
    """Role 2: AST-based static structure extraction for Python corpus documents.

    Recovers imports, top-level definitions with docstrings, and the core logic
    region with argparse/input scaffolding and `__main__` guards removed. Never
    raises on malformed input: unparseable sources are returned verbatim as a
    single region flagged with `parse_failed`.
    """

    def extract(self, source: str) -> StructuralExtract:
        """Extract structure from a Python source document.

        Args:
            source: Raw document text (may be malformed or non-Python).

        Returns:
            StructuralExtract with `parse_failed=True` and the raw text kept in
            `core_logic` when the document cannot be parsed.
        """
        text = source if isinstance(source, str) else str(source or "")
        try:
            tree = ast.parse(text)
        except _PARSE_ERRORS as exc:
            logger.debug("StructuralExtractor falling back to raw text: %s", exc)
            return StructuralExtract(core_logic=text, parse_failed=True)

        lines = text.splitlines()
        return StructuralExtract(
            imports=self.extract_imports(tree, lines),
            units=self.extract_units(tree, lines),
            core_logic=self.extract_core_logic(tree, lines),
            module_docstring=ast.get_docstring(tree),
            parse_failed=False,
        )

    @staticmethod
    def extract_imports(tree: ast.Module, lines: Sequence[str]) -> List[str]:
        """Collect import statements verbatim, deduplicated, in source order."""
        imports: List[str] = []
        seen: Set[str] = set()
        for node in tree.body:
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            segment = _slice_lines(lines, node.lineno, node.end_lineno)
            if not segment:
                try:
                    segment = ast.unparse(node)
                except Exception:  # pragma: no cover - defensive
                    segment = None
            if segment and segment not in seen:
                seen.add(segment)
                imports.append(segment)
        return imports

    def extract_units(self, tree: ast.Module, lines: Sequence[str]) -> List[CodeUnit]:
        """Collect top-level function/async-function/class definitions."""
        units: List[CodeUnit] = []
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                kind: UnitKind = "function"
            elif isinstance(node, ast.AsyncFunctionDef):
                kind = "async_function"
            elif isinstance(node, ast.ClassDef):
                kind = "class"
            else:
                continue

            start = max(node.lineno, 1) - 1
            end = min(max(node.end_lineno or node.lineno, node.lineno), len(lines))
            units.append(
                CodeUnit(
                    name=node.name,
                    kind=kind,
                    signature=self.build_signature(node, lines),
                    docstring=ast.get_docstring(node),
                    source="\n".join(lines[start:end]),
                    lineno=node.lineno,
                    end_lineno=node.end_lineno or node.lineno,
                )
            )
        return units

    def extract_core_logic(self, tree: ast.Module, lines: Sequence[str]) -> str:
        """Return module-level code with imports, definitions, and CLI scaffolding removed."""
        scanner = _BoilerplateScanner(tree)
        dropped_types = (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        )
        dropped: Set[int] = set()
        for stmt in tree.body:
            if scanner.is_scaffolding(stmt) or isinstance(stmt, dropped_types):
                end = stmt.end_lineno or stmt.lineno
                dropped.update(range(stmt.lineno, end + 1))

        kept = [line.rstrip() for number, line in enumerate(lines, start=1) if number not in dropped]
        return _normalise_block("\n".join(kept))

    @staticmethod
    def build_signature(node: ast.AST, lines: Sequence[str]) -> str:
        """Build a definition header (decorators and body excluded).

        Uses the literal source header when it can be sliced cleanly and falls
        back to `ast.unparse` for one-line bodies or multi-line signatures whose
        body starts on the closing line.
        """
        body = getattr(node, "body", None)
        lineno = getattr(node, "lineno", None)
        if body and isinstance(lineno, int):
            start = lineno - 1
            end = body[0].lineno - 1
            if end > start:
                header = "\n".join(lines[start:end]).strip()
                if header.endswith(":"):
                    return header
        return StructuralExtractor._signature_from_ast(node)

    @staticmethod
    def _signature_from_ast(node: ast.AST) -> str:
        """Reconstruct a signature via `ast.unparse` with an empty body."""
        name = getattr(node, "name", "")
        try:
            stub = copy.deepcopy(node)
            stub.decorator_list = []
            stub.body = [ast.Pass()]
            ast.fix_missing_locations(stub)
            return ast.unparse(stub)
        except Exception:  # pragma: no cover - defensive
            return name


class LongDocChunker:
    """Role 2: structural chunker producing embeddable `SnippetSchema` entries.

    Documents whose estimated token count exceeds `max_tokens` are recursively
    split on structural boundaries: function/class definitions first, then
    blank-line paragraphs, then physical lines, then words as a last resort.
    Every chunk keeps its parent `doc_id`, so `aggregate_scores` can max-pool
    per-chunk similarity back to document level.
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        tokens_per_word: float = TOKENS_PER_WORD,
        summary_max_lines: int = SUMMARY_MAX_LINES,
    ) -> None:
        """Initialize the chunker.

        Args:
            max_tokens: Estimated token budget per chunk before splitting.
            tokens_per_word: Multiplier applied to the whitespace word count.
            summary_max_lines: Lines of a chunk kept in `summary_view`.
        """
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if summary_max_lines < 1:
            raise ValueError("summary_max_lines must be >= 1")
        self.max_tokens = max_tokens
        self.tokens_per_word = tokens_per_word
        self.summary_max_lines = summary_max_lines
        self.extractor = StructuralExtractor()

    def estimate_tokens(self, text: str) -> int:
        """Instance-level token estimate using this chunker's word multiplier."""
        return estimate_tokens(text, self.tokens_per_word)

    def _body_budget(self, header: str) -> int:
        """Token budget left for the body once header context is accounted for."""
        if not header:
            return self.max_tokens
        reserved = estimate_tokens(header, self.tokens_per_word)
        return max(self.max_tokens // 2, self.max_tokens - reserved)

    def chunk(self, source: str, doc_id: str) -> List[SnippetSchema]:
        """Split one corpus document into embeddable snippets.

        Args:
            source: Raw document text.
            doc_id: Identifier of the document in the corpus.

        Returns:
            One `SnippetSchema` per chunk, all sharing `doc_id`. Never returns an
            empty list, so every corpus document stays rankable.
        """
        text = source if isinstance(source, str) else str(source or "")
        extract = self.extractor.extract(text)

        if extract.parse_failed:
            piece = _Piece("", text, truncate=True)
            return [self._to_snippet(doc_id, 0, 1, piece, extract)]

        if self.estimate_tokens(text) <= self.max_tokens:
            piece = _Piece(
                "", text, docstring=extract.module_docstring, truncate=False
            )
            return [self._to_snippet(doc_id, 0, 1, piece, extract)]

        pieces = self._structural_pieces(extract)
        if not pieces:
            pieces = self._paragraph_pieces(
                text, kind="module", docstring=extract.module_docstring
            )
        if not pieces:
            pieces = [_Piece("", text, docstring=extract.module_docstring, truncate=False)]

        return [
            self._to_snippet(doc_id, index, len(pieces), piece, extract)
            for index, piece in enumerate(pieces)
        ]

    def chunk_corpus(self, corpus: Dict[str, str]) -> Dict[str, List[SnippetSchema]]:
        """Chunk a `{doc_id: source}` corpus mapping into snippets per document."""
        return {doc_id: self.chunk(source, doc_id) for doc_id, source in corpus.items()}

    def _structural_pieces(self, extract: StructuralExtract) -> List[_Piece]:
        """Split on definition boundaries, preceded by the core logic region."""
        pieces: List[_Piece] = []
        if extract.core_logic.strip():
            pieces.extend(
                self._paragraph_pieces(
                    extract.core_logic, name="<module>", kind="module", docstring=extract.module_docstring
                )
            )
        for unit in extract.units:
            pieces.extend(self._unit_pieces(unit))
        return pieces

    def _unit_pieces(self, unit: CodeUnit) -> List[_Piece]:
        """Emit one piece per definition, recursively splitting oversized ones."""
        if not unit.source.strip():
            return []
        if self.estimate_tokens(unit.source) <= self.max_tokens:
            return [
                _Piece(
                    "",
                    unit.source,
                    name=unit.name,
                    kind=unit.kind,
                    signature=unit.signature,
                    docstring=unit.docstring,
                    truncate=False,
                )
            ]
        return self._paragraph_pieces(
            unit.source,
            header=_unit_header(unit, self.summary_max_lines),
            name=unit.name,
            kind=unit.kind,
            signature=unit.signature,
            docstring=unit.docstring,
        )

    def _paragraph_pieces(
        self,
        text: str,
        header: str = "",
        name: str = "",
        kind: str = "module",
        signature: str = "",
        docstring: str | None = None,
    ) -> List[_Piece]:
        """Group blank-line-separated paragraphs up to the token budget."""
        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return []
        budget = self._body_budget(header)
        if len(paragraphs) == 1:
            return self._fit_pieces(
                paragraphs[0], header, name, kind, signature, docstring
            )

        groups: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        for paragraph in paragraphs:
            tokens = self.estimate_tokens(paragraph) + 1
            if current and current_tokens + tokens > budget:
                groups.append(current)
                current = [paragraph]
                current_tokens = tokens
            else:
                current.append(paragraph)
                current_tokens += tokens
        if current:
            groups.append(current)

        pieces: List[_Piece] = []
        for group in groups:
            pieces.extend(
                self._fit_pieces("\n\n".join(group), header, name, kind, signature, docstring)
            )
        return pieces

    def _fit_pieces(
        self,
        body: str,
        header: str,
        name: str,
        kind: str,
        signature: str,
        docstring: str | None,
    ) -> List[_Piece]:
        """Return the body as one piece, or fall back to line/word splitting."""
        if self.estimate_tokens(body) <= self._body_budget(header):
            return [_Piece(header, body, name, kind, signature, docstring)]
        return self._line_pieces(body, header, name, kind, signature, docstring)

    def _line_pieces(
        self,
        body: str,
        header: str,
        name: str,
        kind: str,
        signature: str,
        docstring: str | None,
    ) -> List[_Piece]:
        """Group physical lines up to the token budget as the final fallback."""
        lines = body.splitlines()
        if len(lines) <= 1:
            return self._word_pieces(body, header, name, kind, signature, docstring)

        budget = self._body_budget(header)
        groups: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        for line in lines:
            tokens = self.estimate_tokens(line) + 1
            if current and current_tokens + tokens > budget:
                groups.append(current)
                current = [line]
                current_tokens = tokens
            else:
                current.append(line)
                current_tokens += tokens
        if current:
            groups.append(current)

        pieces: List[_Piece] = []
        for group in groups:
            joined = "\n".join(group)
            if len(group) == 1 and self.estimate_tokens(joined) > budget:
                pieces.extend(
                    self._word_pieces(joined, header, name, kind, signature, docstring)
                )
            else:
                pieces.append(_Piece(header, joined, name, kind, signature, docstring))
        return pieces

    def _word_pieces(
        self,
        body: str,
        header: str,
        name: str,
        kind: str,
        signature: str,
        docstring: str | None,
    ) -> List[_Piece]:
        """Hard-split a single unbreakable line into token-bounded pieces."""
        words = body.split()
        if len(words) <= 1:
            return [_Piece(header, body, name, kind, signature, docstring)]

        budget = self._body_budget(header)
        groups: List[List[str]] = []
        current: List[str] = []
        current_tokens = 0
        for word in words:
            tokens = self.estimate_tokens(word) + 1
            if current and current_tokens + tokens > budget:
                groups.append(current)
                current = [word]
                current_tokens = tokens
            else:
                current.append(word)
                current_tokens += tokens
        if current:
            groups.append(current)
        return [
            _Piece(header, " ".join(group), name, kind, signature, docstring)
            for group in groups
        ]

    def _summarize(self, piece: _Piece) -> str:
        """Build the embedded view: signature/docstring header plus leading lines.

        Pieces already inside the token budget are embedded verbatim, so a
        document that needs no chunking is represented exactly as the baseline
        encoder sees it today.
        """
        if not piece.truncate:
            return "\n".join(part for part in (piece.header, piece.body) if part).strip("\n")
        lines = piece.body.splitlines()
        body_view = "\n".join(lines[: self.summary_max_lines])
        if len(lines) > self.summary_max_lines:
            body_view = f"{body_view}\n... ({len(lines) - self.summary_max_lines} more lines)"
        if piece.header:
            return f"{piece.header}\n{body_view}"
        return body_view

    def _to_snippet(
        self,
        doc_id: str,
        chunk_index: int,
        num_chunks: int,
        piece: _Piece,
        extract: StructuralExtract,
    ) -> SnippetSchema:
        """Materialize one chunk into a `SnippetSchema` with a content-hash id."""
        full_view = "\n".join(part for part in (piece.header, piece.body) if part).strip("\n")
        return SnippetSchema(
            id=compute_content_hash(full_view),
            doc_id=doc_id,
            summary_view=self._summarize(piece),
            full_view=full_view,
            metadata=SnippetMetadata(
                chunk_index=chunk_index,
                num_chunks=num_chunks,
                has_imports=list(extract.imports),
                name=piece.name,
                kind=piece.kind,
                signature=piece.signature,
                docstring=piece.docstring,
                parse_failed=extract.parse_failed,
                structural=bool(extract.units) and not extract.parse_failed,
                estimated_tokens=estimate_tokens(full_view, self.tokens_per_word),
            ),
        )


class ChunkCache:
    """SQLite-backed memoization of chunking results keyed by content hash.

    `get_or_compute` re-chunks a document only when its content hash (and the
    chunker's token budget) is not already stored, so re-running the pipeline
    over an unchanged corpus skips parsing, splitting, and serialization.

    The lookup key is `compute_text_hash`, a whitespace-normalized sha256 whose
    cost is one C-level split instead of the AST parse plus `ast.dump` that
    `SnippetSchema.id` requires; identical keys always imply identical chunk
    output, so a hit is safe. Database errors degrade to recomputation rather
    than failing the run.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS chunks (
        content_hash TEXT NOT NULL,
        max_tokens INTEGER NOT NULL,
        doc_id TEXT NOT NULL,
        num_chunks INTEGER NOT NULL,
        payload TEXT NOT NULL,
        created_at REAL NOT NULL,
        PRIMARY KEY (content_hash, max_tokens)
    )
    """

    def __init__(
        self,
        db_path: str = DEFAULT_CACHE_PATH,
        chunker: LongDocChunker | None = None,
    ) -> None:
        """Initialize the cache.

        Args:
            db_path: SQLite file path, or ':memory:' for an ephemeral cache.
            chunker: Chunker used on cache misses; defaults to a 256-token
                `LongDocChunker`.
        """
        self.db_path = db_path
        self.chunker = chunker if chunker is not None else LongDocChunker()
        self._conn: sqlite3.Connection | None = None
        self.hits = 0
        self.misses = 0

    @property
    def conn(self) -> sqlite3.Connection:
        """Lazily opened SQLite connection with the cache table ensured."""
        if self._conn is None:
            directory = os.path.dirname(os.path.abspath(self.db_path))
            if self.db_path != ":memory:" and directory:
                os.makedirs(directory, exist_ok=True)
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute(self._SCHEMA)
            self._conn.commit()
        return self._conn

    def get_or_compute(self, source: str, doc_id: str) -> List[SnippetSchema]:
        """Return cached snippets for `source`, chunking only on a cache miss.

        Args:
            source: Raw document text.
            doc_id: Identifier of the document in the corpus.

        Returns:
            List of `SnippetSchema`, re-stamped with the requested `doc_id`.
        """
        content_hash = compute_text_hash(source)
        budget = self.chunker.max_tokens

        payload = self._read(content_hash, budget)
        if payload is not None:
            self.hits += 1
            return [
                SnippetSchema.model_validate(item).model_copy(update={"doc_id": doc_id})
                for item in payload
            ]

        self.misses += 1
        snippets = self.chunker.chunk(source, doc_id)
        self._write(content_hash, budget, doc_id, snippets)
        return snippets

    def _read(self, content_hash: str, budget: int) -> List[Dict[str, Any]] | None:
        try:
            row = self.conn.execute(
                "SELECT payload FROM chunks WHERE content_hash = ? AND max_tokens = ?",
                (content_hash, budget),
            ).fetchone()
        except sqlite3.Error as exc:
            logger.warning("ChunkCache read failed, recomputing: %s", exc)
            return None
        if row is None:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            logger.warning("ChunkCache payload corrupted for %s, recomputing", content_hash)
            return None

    def _write(
        self, content_hash: str, budget: int, doc_id: str, snippets: Sequence[SnippetSchema]
    ) -> None:
        payload = json.dumps([snippet.model_dump(mode="json") for snippet in snippets])
        try:
            self.conn.execute(
                "INSERT OR REPLACE INTO chunks "
                "(content_hash, max_tokens, doc_id, num_chunks, payload, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    content_hash,
                    budget,
                    doc_id,
                    len(snippets),
                    payload,
                    time.time(),
                ),
            )
            self.conn.commit()
        except sqlite3.Error as exc:
            logger.warning("ChunkCache write failed: %s", exc)

    def clear(self) -> None:
        """Delete every cached row."""
        try:
            self.conn.execute("DELETE FROM chunks")
            self.conn.commit()
        except sqlite3.Error as exc:
            logger.warning("ChunkCache clear failed: %s", exc)

    def stats(self) -> Dict[str, int]:
        """Return hit/miss counters and the number of cached rows."""
        try:
            rows = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        except sqlite3.Error:
            rows = 0
        return {"hits": self.hits, "misses": self.misses, "entries": int(rows)}

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "ChunkCache":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


def aggregate_scores(
    chunk_scores: Dict[str, float], chunk_to_doc: Dict[str, str]
) -> Dict[str, float]:
    """Max-pool per-chunk similarity scores back to one score per document.

    A document split into several chunks should rank on its single best-matching
    chunk, so chunk scores are collapsed with a max over `chunk_to_doc`.

    Args:
        chunk_scores: Mapping of chunk id to similarity score.
        chunk_to_doc: Mapping of chunk id to parent document id.

    Returns:
        Mapping of document id to its max chunk score, in first-seen order.
        Chunks without a document mapping are ignored.
    """
    doc_scores: Dict[str, float] = {}
    for chunk_id, score in chunk_scores.items():
        doc_id = chunk_to_doc.get(chunk_id)
        if doc_id is None:
            continue
        current = doc_scores.get(doc_id)
        if current is None or score > current:
            doc_scores[doc_id] = score
    return doc_scores


def build_chunk_to_doc(snippets: Sequence[SnippetSchema]) -> Dict[str, str]:
    """Build the chunk key to document id map consumed by `aggregate_scores`.

    `SnippetSchema.id` is a pure content hash, so byte-identical chunks appearing
    in two different documents would collide and one document would silently lose
    its score. Colliding keys are therefore suffixed with `#<doc_id>`, which keeps
    the mapping injective (one entry per snippet, in snippet order) without
    changing `SnippetSchema.id`. Callers score snippets positionally against
    `list(build_chunk_to_doc(snippets))`.
    """
    mapping: Dict[str, str] = {}
    for snippet in snippets:
        key = snippet.id
        if key in mapping and mapping[key] != snippet.doc_id:
            key = f"{key}#{snippet.doc_id}"
        mapping[key] = snippet.doc_id
    return mapping


def _slice_lines(lines: Sequence[str], lineno: int | None, end_lineno: int | None) -> str:
    """Slice a 1-based inclusive line range out of pre-split source lines."""
    if not lineno:
        return ""
    start = max(lineno, 1) - 1
    end = min(max(end_lineno or lineno, lineno), len(lines))
    return "\n".join(lines[start:end]).strip()


def _split_paragraphs(text: str) -> List[str]:
    """Split text on runs of blank lines, preserving code indentation."""
    paragraphs: List[str] = []
    for raw in _PARAGRAPH_BREAK_RE.split(text):
        block = raw.strip("\n")
        if block.strip():
            paragraphs.append(block)
    return paragraphs


def _unit_header(unit: CodeUnit, max_doc_lines: int) -> str:
    """Build the signature + docstring context prepended to fragment summaries."""
    parts: List[str] = []
    if unit.signature:
        parts.append(unit.signature.rstrip())
    if unit.docstring:
        doc_lines = unit.docstring.strip().splitlines()
        truncated = doc_lines[: max_doc_lines]
        parts.append("\n".join(truncated))
        if len(doc_lines) > len(truncated):
            parts.append(f"... ({len(doc_lines) - len(truncated)} more docstring lines)")
    return "\n".join(parts).strip()


def _normalise_block(text: str) -> str:
    """Trim trailing whitespace and collapse runs of blank lines."""
    collapsed = _BLANK_RUN_RE.sub("\n\n", text.strip("\n"))
    return "\n".join(line.rstrip() for line in collapsed.splitlines()).strip("\n")
