"""Query Understanding preprocessor module for Agentic Code Intelligence.

Role 1: Query Understanding.
Provides:
- Pydantic model `StructuredQuery` for standardizing downstream query metadata.
- Rule-based regex intent classifier (definition-lookup, usage-lookup, behavior-lookup).
- Identifier expander generating camelCase, PascalCase, and snake_case variants.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Pattern, Set, Tuple
from pydantic import BaseModel, Field

IntentType = Literal["definition-lookup", "usage-lookup", "behavior-lookup"]


class StructuredQuery(BaseModel):
    """Standardized structured query passed to downstream retrieval roles."""

    raw_query: str = Field(..., description="The original natural language query.")
    intent: IntentType = Field(..., description="Categorized query intent.")
    intent_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the categorized intent."
    )
    matched_pattern: str | None = Field(
        default=None, description="Regex pattern name or rule that matched the intent."
    )
    extracted_terms: List[str] = Field(
        default_factory=list,
        description="Key terms and phrases extracted from the query.",
    )
    expanded_identifiers: List[str] = Field(
        default_factory=list,
        description="Expanded camelCase, PascalCase, and snake_case code identifiers.",
    )
    processed_query: str = Field(
        ...,
        description="Enriched query string incorporating expanded identifiers for retrieval.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary downstream metadata."
    )


class IdentifierExpander:
    """Expands natural language technical terms into code casing variants.

    Generates camelCase, PascalCase, and snake_case representations for downstream
    sparse and dense retrieval against codebases.
    """

    STOP_WORDS: Set[str] = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
        "by", "about", "how", "what", "where", "why", "which", "who", "when",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "should", "would", "before", "after",
        "into", "from", "of", "before", "during", "then", "this", "that"
    }

    @classmethod
    def split_into_tokens(cls, term: str) -> List[str]:
        """Split camelCase, PascalCase, snake_case, or space-separated words into lowercase tokens."""
        # Handle snake_case and kebab-case
        term = term.replace("-", "_")
        subterms = term.split("_")
        tokens: List[str] = []
        for st in subterms:
            # Split camelCase / PascalCase
            parts = re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z][a-z0-9]|\b)", st)
            if parts:
                tokens.extend([p.lower() for p in parts if p])
            elif st.strip():
                tokens.append(st.lower().strip())
        return [t for t in tokens if t]

    @classmethod
    def to_camel_case(cls, tokens: List[str]) -> str:
        """Convert a list of tokens to camelCase."""
        if not tokens:
            return ""
        return tokens[0].lower() + "".join(t.capitalize() for t in tokens[1:])

    @classmethod
    def to_pascal_case(cls, tokens: List[str]) -> str:
        """Convert a list of tokens to PascalCase."""
        if not tokens:
            return ""
        return "".join(t.capitalize() for t in tokens)

    @classmethod
    def to_snake_case(cls, tokens: List[str]) -> str:
        """Convert a list of tokens to snake_case."""
        if not tokens:
            return ""
        return "_".join(t.lower() for t in tokens)

    def expand_token_group(self, tokens: List[str]) -> List[str]:
        """Generate camelCase, PascalCase, and snake_case variants for a group of tokens."""
        clean = [re.sub(r"[^a-zA-Z0-9]", "", t) for t in tokens]
        clean = [t for t in clean if t]
        if not clean:
            return []

        variants = [
            self.to_camel_case(clean),
            self.to_pascal_case(clean),
            self.to_snake_case(clean),
        ]
        # Return unique non-empty variants
        seen: Set[str] = set()
        res: List[str] = []
        for v in variants:
            if v and v not in seen:
                seen.add(v)
                res.append(v)
        return res

    def extract_and_expand(self, query: str) -> Tuple[List[str], List[str]]:
        """Extract candidate term sequences from query and expand them.

        Returns:
            Tuple of (extracted_terms, expanded_identifiers).
        """
        # Find explicit code tokens like `foo_bar` or `fooBar` or words in quotes
        quoted = re.findall(r"['\"`]([^'\"`]+)['\"`]", query)
        raw_words = re.findall(r"[a-zA-Z0-9_]+", query)

        # Filter meaningful words (exclude single characters & common stop words)
        content_words = [
            w for w in raw_words
            if len(w) > 1 and w.lower() not in self.STOP_WORDS
        ]

        extracted_terms: List[str] = []
        for q in quoted:
            if q not in extracted_terms:
                extracted_terms.append(q)

        # Add multi-word n-grams from content words (bigrams and trigrams)
        token_groups: List[List[str]] = []

        # If quoted terms exist, split them into token groups
        for q in quoted:
            toks = self.split_into_tokens(q)
            if toks and toks not in token_groups:
                token_groups.append(toks)

        # Single tokens that already look like identifiers (snake_case or camelCase)
        for w in content_words:
            if "_" in w or (re.search(r"[a-z][A-Z]", w) is not None):
                toks = self.split_into_tokens(w)
                if toks and toks not in token_groups:
                    token_groups.append(toks)
                if w not in extracted_terms:
                    extracted_terms.append(w)

        # Adjacent content word pairs (bigrams) e.g., "input preprocessed" -> ['input', 'preprocessed']
        if len(content_words) >= 2:
            for i in range(len(content_words) - 1):
                pair = [content_words[i], content_words[i + 1]]
                phrase = " ".join(pair)
                if phrase not in extracted_terms:
                    extracted_terms.append(phrase)
                token_groups.append([t.lower() for t in pair])

        # Trigrams for richer concepts e.g., "main function normalize"
        if len(content_words) >= 3:
            for i in range(min(2, len(content_words) - 2)):
                triplet = [content_words[i], content_words[i + 1], content_words[i + 2]]
                token_groups.append([t.lower() for t in triplet])

        # Individual significant content words (if 3+ characters)
        for w in content_words:
            if len(w) >= 3 and [w.lower()] not in token_groups:
                token_groups.append([w.lower()])
            if w not in extracted_terms:
                extracted_terms.append(w)

        expanded_identifiers: List[str] = []
        seen_identifiers: Set[str] = set()

        for group in token_groups:
            expanded = self.expand_token_group(group)
            for exp in expanded:
                if exp not in seen_identifiers:
                    seen_identifiers.add(exp)
                    expanded_identifiers.append(exp)

        return extracted_terms, expanded_identifiers


class QueryParser:
    """Role 1: Query Understanding engine.

    Parses natural language user queries, applies rule-based intent classification,
    and enriches queries with expanded identifier variants for downstream retrieval.
    """

    def __init__(self) -> None:
        self.expander = IdentifierExpander()
        self._init_rules()

    def _init_rules(self) -> None:
        """Initialize regex rules for query classification."""
        self.definition_patterns: List[Tuple[str, Pattern[str]]] = [
            (
                "def_explicit_keyword",
                re.compile(
                    r"\b(?:where\s+is|find|search\s+for|show|get|locate)\s+(?:the\s+)?(?:definition|declaration|signature|class|function|method|interface|struct|enum)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "def_declaration_of",
                re.compile(
                    r"\b(?:definition|declaration|signature|interface)\s+of\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "def_code_syntax",
                re.compile(
                    r"\b(?:def|class|interface|type|struct)\s+[a-zA-Z_][a-zA-Z0-9_]*",
                    re.IGNORECASE,
                ),
            ),
            (
                "def_where_defined",
                re.compile(
                    r"\bwhere\s+(?:is|are)\s+[a-zA-Z0-9_]+\s+(?:defined|declared|implemented)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "def_what_is_concept",
                re.compile(
                    r"\bwhat\s+is\s+(?:the\s+)?(?:definition|signature|type\s+of|class\s+of)\b",
                    re.IGNORECASE,
                ),
            ),
        ]

        self.usage_patterns: List[Tuple[str, Pattern[str]]] = [
            (
                "usage_how_to_use",
                re.compile(
                    r"\bhow\s+(?:to|can\s+I|do\s+I|should\s+we)\s+(?:use|call|invoke|instantiate|execute|import|run|apply|initialize)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "usage_example_sample",
                re.compile(
                    r"\b(?:usage|example|sample|snippet|demonstration)\s+(?:of|for|showing)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "usage_passive_call",
                re.compile(
                    r"\bhow\s+is\s+[a-zA-Z0-9_]+\s+(?:used|called|invoked|instantiated|imported)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "usage_how_do_i_call",
                re.compile(
                    r"\b(?:call|invoke|use|instantiate)\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(",
                    re.IGNORECASE,
                ),
            ),
        ]

        self.behavior_patterns: List[Tuple[str, Pattern[str]]] = [
            (
                "behavior_how_does_work",
                re.compile(
                    r"\bhow\s+(?:does|is|are|do|would)\s+.*?\s+(?:work|behave|preprocess|process|handle|execute|compute|transform|validate|convert|return|evaluate|check|normalize)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "behavior_what_happens",
                re.compile(
                    r"\bwhat\s+(?:happens|occurs|does)\s+(?:when|if|during|before|after)\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "behavior_lifecycle_logic",
                re.compile(
                    r"\b(?:behavior|working|logic|flow|algorithm|mechanism|lifecycle|internals|pipeline)\s+of\b",
                    re.IGNORECASE,
                ),
            ),
            (
                "behavior_why_does",
                re.compile(r"\bwhy\s+(?:does|is|do)\b", re.IGNORECASE),
            ),
            (
                "behavior_explanation",
                re.compile(
                    r"\b(?:explain|describe)\s+(?:the\s+)?(?:logic|flow|step|process|behavior)\b",
                    re.IGNORECASE,
                ),
            ),
        ]

    def classify_intent(self, query: str) -> Tuple[IntentType, float, str | None]:
        """Classify query intent into definition-lookup, usage-lookup, or behavior-lookup.

        Returns:
            Tuple of (intent, confidence, matched_rule_name).
        """
        # 1. Check definition lookup
        for name, pattern in self.definition_patterns:
            if pattern.search(query):
                return "definition-lookup", 0.95, name

        # 2. Check usage lookup
        for name, pattern in self.usage_patterns:
            if pattern.search(query):
                return "usage-lookup", 0.95, name

        # 3. Check behavior lookup
        for name, pattern in self.behavior_patterns:
            if pattern.search(query):
                return "behavior-lookup", 0.95, name

        # Fallback heuristic based on keywords
        q_lower = query.lower()
        if any(k in q_lower for k in ["class", "def ", "signature", "interface", "type of"]):
            return "definition-lookup", 0.70, "keyword_fallback_def"
        if any(k in q_lower for k in ["use", "call", "example", "import", "invoke"]):
            return "usage-lookup", 0.70, "keyword_fallback_usage"
        if any(k in q_lower for k in ["how", "process", "work", "handle", "logic", "flow", "before", "after"]):
            return "behavior-lookup", 0.75, "keyword_fallback_behavior"

        # Default fallback for general code search
        return "behavior-lookup", 0.50, "default_fallback"

    def parse(self, query: str) -> StructuredQuery:
        """Parse a single natural language query into a StructuredQuery.

        Args:
            query: The natural language search query.

        Returns:
            StructuredQuery instance ready for downstream retrieval modules.
        """
        stripped_query = query.strip()
        intent, confidence, rule_name = self.classify_intent(stripped_query)
        extracted_terms, expanded_identifiers = self.expander.extract_and_expand(stripped_query)

        # Enrich query for downstream dense/sparse retrieval
        # Format: "<intent_prefix> <raw_query> | identifiers: <expanded_tokens>"
        intent_prefix_map: Dict[IntentType, str] = {
            "definition-lookup": "Definition of code structure:",
            "usage-lookup": "Code usage and implementation example:",
            "behavior-lookup": "Code execution flow and behavior:",
        }
        intent_prefix = intent_prefix_map.get(intent, "")

        if expanded_identifiers:
            # Append top identifier variants to the query text
            top_identifiers = " ".join(expanded_identifiers[:12])
            processed_query = f"{intent_prefix} {stripped_query} | tokens: {top_identifiers}".strip()
        else:
            processed_query = f"{intent_prefix} {stripped_query}".strip()

        return StructuredQuery(
            raw_query=stripped_query,
            intent=intent,
            intent_confidence=confidence,
            matched_pattern=rule_name,
            extracted_terms=extracted_terms,
            expanded_identifiers=expanded_identifiers,
            processed_query=processed_query,
            metadata={"rule": rule_name},
        )

    def parse_batch(self, queries: List[str]) -> List[StructuredQuery]:
        """Parse a batch of queries."""
        return [self.parse(q) for q in queries]
