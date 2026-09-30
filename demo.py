"""Retrieval demo: rank code snippets for natural-language queries, with timings.

    python demo.py                                   # interactive, full CoIR Apps corpus
    python demo.py -q "count pairs whose sum is divisible by k" -k 5
    python demo.py --dir path/to/project             # index any folder of source files
    python demo.py --versions --limit 400            # P1 + Bonus: multi-version demo

Uses configs/best.json (tuned on the dev split) when present. Corpus embeddings are
cached by content hash in .cache/, so only the first run pays the embedding cost.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.retriever import HybridRetriever, RetrieverConfig

CODE_EXTS = {".py", ".js", ".ts", ".java", ".go", ".rb", ".rs", ".c", ".cc", ".cpp", ".h", ".cs", ".php"}


def load_config(path: str) -> RetrieverConfig:
    return RetrieverConfig.load(path) if os.path.exists(path) else RetrieverConfig()


def apps_corpus(limit: int | None) -> Dict[str, str]:
    from src.evaluation import load_apps

    data = load_apps()
    pairs = list(zip(data.doc_ids, data.doc_texts))
    return dict(pairs[:limit] if limit else pairs)


def dir_corpus(root: str) -> Dict[str, str]:
    docs: Dict[str, str] = {}
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith((".", "__")) and d not in {"node_modules", "venv"}]
        for name in files:
            if os.path.splitext(name)[1] in CODE_EXTS:
                path = os.path.join(base, name)
                with open(path, encoding="utf-8", errors="ignore") as f:
                    docs[os.path.relpath(path, root)] = f.read()
    return docs


def preview(code: str, lines: int) -> str:
    body = code.strip().splitlines()
    shown = "\n".join(f"      {line}" for line in body[:lines])
    return shown + (f"\n      ... ({len(body) - lines} more lines)" if len(body) > lines else "")


def print_results(results: List[Tuple[str, float]], texts: Dict[str, str], lines: int) -> None:
    for rank, (doc_id, score) in enumerate(results, 1):
        print(f"  #{rank:<2} {doc_id}  (score {score:.4f})")
        print(preview(texts[doc_id], lines))
        print()


def run_query(retr: HybridRetriever, texts: Dict[str, str], query: str, k: int, lines: int) -> None:
    t0 = time.perf_counter()
    parsed = retr.expander.parse(query)
    t1 = time.perf_counter()
    results = retr.retrieve([query], top_k=k)[0]
    t2 = time.perf_counter()
    print(f"\nIntent: {parsed.intent} ({parsed.intent_confidence:.2f})   "
          f"parse {1000 * (t1 - t0):.1f} ms | retrieve {1000 * (t2 - t1):.0f} ms "
          f"over {len(retr.doc_ids)} snippets\n")
    print_results(results, texts, lines)


def mutate(code: str, rng: random.Random) -> str:
    """Simulated commit: rename a variable, or add a comment/guard line."""
    names = sorted(set(re.findall(r"\b([a-z])\b\s*=", code)))
    if names and rng.random() < 0.6:
        old = rng.choice(names)
        return re.sub(rf"\b{old}\b", f"{old}_val", code)
    return "# refactored: input handling\n" + code


def versions_demo(config: RetrieverConfig, limit: int, k: int, lines: int, query: str | None) -> None:
    from src.versioning import VersionedCodeIndex

    base = apps_corpus(None)
    ids = list(base)
    rng = random.Random(0)
    v1 = {d: base[d] for d in ids[:limit]}
    spare = ids[limit:]

    def next_version(prev: Dict[str, str], frac: float = 0.05) -> Dict[str, str]:
        docs = dict(prev)
        keys = list(docs)
        for d in rng.sample(keys, max(1, int(frac * len(keys)))):
            docs[d] = mutate(docs[d], rng)
        for d in rng.sample(keys, max(1, int(0.02 * len(keys)))):
            docs.pop(d, None)
        for _ in range(max(1, int(0.02 * len(keys)))):
            d = spare.pop()
            docs[d] = base[d]
        return docs

    index = VersionedCodeIndex(config)
    versions = {"v1": v1}
    versions["v2"] = next_version(v1)
    versions["v3"] = next_version(versions["v2"])
    print(f"\n{'version':<8}{'docs':>6}{'added':>7}{'modified':>10}{'removed':>9}"
          f"{'embedded':>10}{'reused':>8}{'seconds':>9}")
    for name, docs in versions.items():
        d = index.add_version(name, docs)
        print(f"{d.version:<8}{len(docs):>6}{d.added:>7}{d.modified:>10}{d.removed:>9}"
              f"{d.embedded:>10}{d.reused:>8}{d.seconds:>9.1f}")

    changed = [d for d in versions["v3"] if d in v1 and versions["v3"][d] != v1[d]]
    q = query or "Given n integers, output the number of pairs whose sum is divisible by k."
    print(f"\nQuery: {q}")
    for name in versions:
        t = time.perf_counter()
        hits = index.search(q, version=name, k=3)
        print(f"\n[{name}] top-3 in {1000 * (time.perf_counter() - t):.0f} ms: "
              + ", ".join(f"{d} ({s:.4f})" for d, s in hits))

    t = time.perf_counter()
    hits = index.search_all(q, k=k)
    print(f"\n[all versions] evolutionary top-{k} in {1000 * (time.perf_counter() - t):.0f} ms "
          f"(near-duplicate versions collapsed per snippet):\n")
    for rank, h in enumerate(hits, 1):
        print(f"  #{rank:<2} {h.doc_id} best@{h.version} (score {h.score:.4f})  "
              f"present in {','.join(h.versions_present)}"
              + (f"; changed in {','.join(h.versions_changed)}" if h.versions_changed else ""))
        print(preview(h.text, lines))
        print()
    print(f"({len(changed)} snippets changed between v1 and v3; e.g. {', '.join(changed[:5])})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", "-q")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="Index only the first N Apps documents.")
    ap.add_argument("--dir", help="Index a folder of source files instead of CoIR Apps.")
    ap.add_argument("--versions", action="store_true", help="Run the multi-version (P1/Bonus) demo.")
    ap.add_argument("--config", default="configs/best.json")
    ap.add_argument("--lines", type=int, default=10, help="Code lines shown per result.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    config = load_config(args.config)
    if args.versions:
        versions_demo(config, args.limit or 400, args.k, args.lines, args.query)
        return

    t0 = time.perf_counter()
    retr = HybridRetriever(config)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s ({config.model_name}, "
          f"{'dense+BM25 RRF' if config.use_bm25 else 'dense'})")
    texts = dir_corpus(args.dir) if args.dir else apps_corpus(args.limit)
    stats = retr.index_corpus(list(texts), list(texts.values()))
    print(f"Indexed {stats['docs']} snippets in {stats['total_seconds']:.1f}s "
          f"({stats['embedded']} embedded, {stats['reused']} reused from cache)")

    if args.query:
        run_query(retr, texts, args.query, args.k, args.lines)
        return
    print("\nEnter a query (empty line to quit).")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            break
        run_query(retr, texts, query, args.k, args.lines)


if __name__ == "__main__":
    main()
