#!/usr/bin/env python
"""
Laya + IAB Content Taxonomy 3.0
Hierarchical / beam-search benchmark for NVIDIA.

WHY THIS VERSION EXISTS
-----------------------
The earlier 700 x NOUL test treated every IAB category as an independent
yes/no question. That is useful as a throughput stress test, but it is not
the right primitive for ranking a taxonomy.

This script instead:

    IAB ROOT
       |
       +--> CHOICE over Tier-1 candidates
               |
               +--> CHOICE over children
                       |
                       +--> CHOICE over children
                               |
                               +--> CHOICE over children

It keeps multiple paths alive with beam search, so NVIDIA can remain
multi-domain (Technology & Computing, Video Gaming, Automotive, etc.).

IMPORTANT:
- Laya's current config has a special "choice:11+" calibration bucket.
- To avoid relying on very-large-choice calibration, this script NEVER sends
  more than --max-choice-options options in one Choice question (default 10).
- If a taxonomy node has >10 children, a tournament reducer is used.
- Final NOUL verification is reported separately and does NOT determine the
  hierarchical rank.

Default:
    exactly 700 IAB v3.0 categories
    max 10 options / Choice
    beam width 10
    top 3 children kept per parent
    verify final top 20 paths

Run:
    python test_iab700_hierarchical.py

Useful:
    python test_iab700_hierarchical.py --beam-width 15 --children-per-parent 4
    python test_iab700_hierarchical.py --verify-top-k 0
    python test_iab700_hierarchical.py --count 704
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
import sys
import time
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from rl_agent_api import RLAgent


IAB_V3_URL = (
    "https://raw.githubusercontent.com/InteractiveAdvertisingBureau/Taxonomies/"
    "main/Content%20Taxonomies/Content%20Taxonomy%203.0.tsv"
)

DEFAULT_COUNT = 700
DEFAULT_MAX_CHOICE_OPTIONS = 10
DEFAULT_BEAM_WIDTH = 10
DEFAULT_CHILDREN_PER_PARENT = 3
DEFAULT_VERIFY_TOP_K = 20


# ---------------------------------------------------------------------------
# Compact "top 10 search results" corpus
# ---------------------------------------------------------------------------

SEARCH_RESULTS = [
    "NVIDIA is described as a technology company centered on GPUs, accelerated computing and artificial intelligence.",
    "Results discuss NVIDIA data-center AI infrastructure for model training and inference.",
    "GeForce and RTX products are associated with PC gaming, graphics and professional visualization.",
    "NVIDIA DGX and HGX platforms combine GPUs, CPUs, networking and memory for AI data centers.",
    "CUDA is described as NVIDIA's software platform and programming ecosystem for accelerated computing.",
    "NVIDIA networking products support high-speed data-center and AI-cluster interconnects.",
    "Jetson products are used for edge AI, robotics and embedded autonomous systems.",
    "NVIDIA DRIVE technology is associated with autonomous and driverless vehicle computing.",
    "Results discuss digital twins, simulation and physical-AI systems for industrial and robotics applications.",
    "NVIDIA works with cloud providers such as AWS to deploy GPU-based AI and compute infrastructure.",
]


def make_state() -> Dict[str, Any]:
    return {
        "entity": "NVIDIA",
        "task": (
            "Classify the topical aboutness of these aggregated search results "
            "using IAB Content Taxonomy 3.0."
        ),
        "search_results": SEARCH_RESULTS,
        "rules": [
            "Classify the aggregate corpus, not an isolated keyword.",
            "Use only information in the supplied results.",
            "The corpus may legitimately belong to multiple independent IAB branches.",
            "Prefer the most specific supported category when walking a branch.",
        ],
    }


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def download_iab_tsv(path: Path) -> None:
    print("[taxonomy] downloading official IAB Content Taxonomy 3.0...")
    try:
        with urllib.request.urlopen(IAB_V3_URL, timeout=30) as response:
            data = response.read()
        path.write_bytes(data)
        print(f"[taxonomy] cached: {path} ({len(data):,} bytes)")
    except Exception as exc:
        raise RuntimeError(
            "Could not download IAB v3.0 TSV.\n"
            "Download 'Content Taxonomy 3.0.tsv' manually from the IAB Tech Lab "
            "Taxonomies repository and save it next to this script as:\n"
            f"  {path}\n\nOriginal error: {exc}"
        ) from exc


def load_iab_categories(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        download_iab_tsv(path)

    rows = list(
        csv.reader(
            path.read_text(encoding="utf-8-sig").splitlines(),
            delimiter="\t",
        )
    )

    categories: List[Dict[str, Any]] = []

    for row in rows[2:]:
        if not row:
            continue

        row = row + [""] * max(0, 7 - len(row))
        uid, parent, name, tier1, tier2, tier3, tier4 = [
            x.strip() for x in row[:7]
        ]

        if not uid or not name:
            continue

        tiers = [x for x in (tier1, tier2, tier3, tier4) if x]

        categories.append(
            {
                "id": uid,
                "parent_id": parent,
                "name": name,
                "tier": len(tiers),
                "tier1": tier1,
                "tier2": tier2,
                "tier3": tier3,
                "tier4": tier4,
                "path": tiers,
            }
        )

    return categories


def build_tree(
    categories: List[Dict[str, Any]],
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, List[Dict[str, Any]]],
    List[Dict[str, Any]],
]:
    by_id = {c["id"]: c for c in categories}
    children: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for cat in categories:
        if cat["parent_id"] in by_id:
            children[cat["parent_id"]].append(cat)

    for values in children.values():
        values.sort(key=lambda c: (c["name"].lower(), c["id"]))

    roots = [
        c for c in categories
        if not c["parent_id"] or c["parent_id"] not in by_id
    ]
    roots.sort(key=lambda c: (c["name"].lower(), c["id"]))

    return by_id, children, roots


def option_description(
    cat: Dict[str, Any],
    children: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    kids = children.get(cat["id"], [])

    payload: Dict[str, Any] = {
        "name": cat["name"],
        "tier": cat["tier"],
        "path": " > ".join(cat["path"]),
        "what": (
            f"Content materially about '{cat['name']}' in the IAB taxonomy."
        ),
    }

    if kids:
        payload["contains"] = [c["name"] for c in kids[:8]]
        if len(kids) > 8:
            payload["contains_more"] = len(kids) - 8

    if cat["tier"] > 1:
        payload["boundary"] = (
            "Choose this only when this category is more directly supported than "
            "its siblings; parent-topic relevance alone is insufficient."
        )

    return payload


# ---------------------------------------------------------------------------
# Runtime statistics
# ---------------------------------------------------------------------------

@dataclass
class RuntimeStats:
    calls: int = 0
    questions: int = 0
    input_tokens: int = 0
    inference_seconds: float = 0.0
    batches: List[Dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        label: str,
        n_questions: int,
        n_tokens: int,
        seconds: float,
    ) -> None:
        self.calls += 1
        self.questions += n_questions
        self.input_tokens += n_tokens
        self.inference_seconds += seconds
        self.batches.append(
            {
                "label": label,
                "questions": n_questions,
                "input_tokens": n_tokens,
                "seconds": seconds,
            }
        )


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def clear_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def run_questions(
    agent: RLAgent,
    state: Dict[str, Any],
    questions: Dict[str, Any],
    stats: RuntimeStats,
    label: str,
) -> Dict[str, Any]:
    if not questions:
        return {}

    cuda_sync()
    t0 = time.perf_counter()

    result = agent.system_one(
        state=state,
        questions=questions,
    )

    cuda_sync()
    dt = time.perf_counter() - t0

    tokens = int(result.get("usage", {}).get("input_tokens", 0))

    stats.add(
        label=label,
        n_questions=len(questions),
        n_tokens=tokens,
        seconds=dt,
    )

    qps = len(questions) / dt if dt else float("inf")
    avg_tok = tokens / len(questions) if questions else 0.0

    print(
        f"[run] {label:<30} | "
        f"{len(questions):3d} q | "
        f"{dt:7.3f} s | "
        f"{qps:7.1f} q/s | "
        f"{avg_tok:6.1f} tok/q"
    )

    return result["answers"]


# ---------------------------------------------------------------------------
# Choice helpers
# ---------------------------------------------------------------------------

def extract_choice_probabilities(
    answer: Dict[str, Any],
    allowed_ids: Iterable[str],
) -> Dict[str, float]:
    allowed = {str(x) for x in allowed_ids}

    raw = answer.get("probabilities")
    probs: Dict[str, float] = {}

    if isinstance(raw, dict):
        for key, value in raw.items():
            key = str(key)
            if key in allowed:
                try:
                    probs[key] = float(value)
                except (TypeError, ValueError):
                    pass

    # Fallback for unexpected response shapes.
    if not probs:
        selected = answer.get("choice")
        if selected is not None and str(selected) in allowed:
            probs[str(selected)] = float(answer.get("confidence", 1.0))

    # Give absent options an epsilon so sorting/code stays stable.
    eps = 1e-12
    for option_id in allowed:
        probs.setdefault(option_id, eps)

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    return probs


def make_choice_question(
    question_text: str,
    option_cats: List[Dict[str, Any]],
    children: Dict[str, List[Dict[str, Any]]],
    parent_path: Optional[str] = None,
) -> Dict[str, Any]:
    instructions: Dict[str, Any] = {
        "question": question_text,
        "focus": (
            "Choose the best-supported category from these candidates. "
            "Judge the supplied search-result corpus, not general world knowledge."
        ),
    }

    if parent_path:
        instructions["current_parent_path"] = parent_path

    return {
        "type": "choice",
        "instructions": instructions,
        "criteria": {
            cat["id"]: option_description(cat, children)
            for cat in option_cats
        },
    }


@dataclass
class TournamentCandidate:
    cat: Dict[str, Any]
    log_score: float = 0.0
    rounds: List[Dict[str, Any]] = field(default_factory=list)


def chunked(seq: List[Any], size: int) -> List[List[Any]]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def tournament_rank(
    agent: RLAgent,
    state: Dict[str, Any],
    option_cats: List[Dict[str, Any]],
    children: Dict[str, List[Dict[str, Any]]],
    stats: RuntimeStats,
    label: str,
    max_options: int,
    keep: int,
) -> List[Tuple[Dict[str, Any], float, List[Dict[str, Any]]]]:
    """
    Rank arbitrarily many sibling categories without ever sending >max_options
    options in one Choice.

    Returns:
        [(category, normalized_tournament_score, round_trace), ...]

    This is an APPROXIMATE reducer. It intentionally avoids the current
    choice:11+ calibration bucket.
    """
    if not option_cats:
        return []

    if len(option_cats) == 1:
        return [(option_cats[0], 1.0, [])]

    candidates = [
        TournamentCandidate(cat=c)
        for c in option_cats
    ]

    round_no = 0

    while len(candidates) > max_options:
        round_no += 1
        groups = chunked(candidates, max_options)

        questions: Dict[str, Any] = {}
        group_map: Dict[str, List[TournamentCandidate]] = {}

        for gi, group in enumerate(groups, start=1):
            qid = f"{label}__r{round_no}__g{gi}"
            cats = [x.cat for x in group]

            questions[qid] = make_choice_question(
                question_text=(
                    "Which candidate is the best IAB category for the supplied corpus "
                    "within this tournament group?"
                ),
                option_cats=cats,
                children=children,
            )
            group_map[qid] = group

        answers = run_questions(
            agent=agent,
            state=state,
            questions=questions,
            stats=stats,
            label=f"{label} tournament r{round_no}",
        )

        survivors: List[TournamentCandidate] = []

        for qid, group in group_map.items():
            probs = extract_choice_probabilities(
                answers[qid],
                [x.cat["id"] for x in group],
            )

            ranked_group = sorted(
                group,
                key=lambda x: probs[x.cat["id"]],
                reverse=True,
            )

            n_keep = min(keep, len(ranked_group))

            for item in ranked_group[:n_keep]:
                p = max(probs[item.cat["id"]], 1e-12)

                survivors.append(
                    TournamentCandidate(
                        cat=item.cat,
                        log_score=item.log_score + math.log(p),
                        rounds=item.rounds + [
                            {
                                "round": round_no,
                                "group_probability": p,
                                "group_size": len(group),
                            }
                        ],
                    )
                )

        candidates = survivors

    # Final head-to-head among survivors.
    cats = [x.cat for x in candidates]

    question = {
        f"{label}__final": make_choice_question(
            question_text=(
                "Which of these surviving IAB categories is the best-supported "
                "category for the supplied corpus?"
            ),
            option_cats=cats,
            children=children,
        )
    }

    answers = run_questions(
        agent=agent,
        state=state,
        questions=question,
        stats=stats,
        label=f"{label} final",
    )

    ans = answers[f"{label}__final"]
    probs = extract_choice_probabilities(
        ans,
        [x.cat["id"] for x in candidates],
    )

    scored: List[Tuple[TournamentCandidate, float]] = []

    for item in candidates:
        p = max(probs[item.cat["id"]], 1e-12)
        scored.append(
            (
                TournamentCandidate(
                    cat=item.cat,
                    log_score=item.log_score + math.log(p),
                    rounds=item.rounds + [
                        {
                            "round": "final",
                            "group_probability": p,
                            "group_size": len(candidates),
                        }
                    ],
                ),
                item.log_score + math.log(p),
            )
        )

    # Normalize accumulated tournament scores over survivors.
    max_log = max(s for _, s in scored)
    weights = [math.exp(s - max_log) for _, s in scored]
    denom = sum(weights)

    out = []

    for (item, _), weight in zip(scored, weights):
        out.append(
            (
                item.cat,
                weight / denom if denom else 0.0,
                item.rounds,
            )
        )

    out.sort(key=lambda x: x[1], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Hierarchical beam search
# ---------------------------------------------------------------------------

@dataclass
class BeamPath:
    cats: List[Dict[str, Any]]
    log_score: float
    local_probs: List[float]
    tournament_traces: List[List[Dict[str, Any]]]

    @property
    def leaf(self) -> Dict[str, Any]:
        return self.cats[-1]

    @property
    def depth(self) -> int:
        return len(self.cats)

    @property
    def path_text(self) -> str:
        return " > ".join(c["name"] for c in self.cats)

    @property
    def joint_probability(self) -> float:
        return math.exp(self.log_score)

    @property
    def geometric_mean_probability(self) -> float:
        if self.depth == 0:
            return 0.0
        return math.exp(self.log_score / self.depth)


def rank_paths_for_beam(paths: List[BeamPath]) -> List[BeamPath]:
    # At the same taxonomy depth, joint probability is a useful branch score.
    # Across mixed depths, geometric mean is less biased toward shallow leaves.
    return sorted(
        paths,
        key=lambda p: (
            p.geometric_mean_probability,
            p.joint_probability,
        ),
        reverse=True,
    )


def hierarchical_beam_search(
    agent: RLAgent,
    state: Dict[str, Any],
    roots: List[Dict[str, Any]],
    children: Dict[str, List[Dict[str, Any]]],
    stats: RuntimeStats,
    max_options: int,
    beam_width: int,
    children_per_parent: int,
) -> Tuple[List[BeamPath], List[BeamPath]]:
    print()
    print("=" * 116)
    print("HIERARCHICAL CHOICE / BEAM SEARCH")
    print("=" * 116)

    # Tier 1 / roots.
    root_rank = tournament_rank(
        agent=agent,
        state=state,
        option_cats=roots,
        children=children,
        stats=stats,
        label="root",
        max_options=max_options,
        keep=max(children_per_parent, 2),
    )

    active: List[BeamPath] = []

    for cat, p, trace in root_rank[:beam_width]:
        p = max(p, 1e-12)
        active.append(
            BeamPath(
                cats=[cat],
                log_score=math.log(p),
                local_probs=[p],
                tournament_traces=[trace],
            )
        )

    print()
    print("[beam] root survivors")
    for i, path in enumerate(active, 1):
        print(
            f"  {i:2d}. {path.local_probs[-1]:.4f} | "
            f"{path.path_text}"
        )

    terminals: List[BeamPath] = []

    # IAB v3 has at most Tier 4.
    for target_depth in (2, 3, 4):
        next_active: List[BeamPath] = []

        print()
        print(f"[beam] expanding to Tier {target_depth}")

        for parent_path in active:
            kids = children.get(parent_path.leaf["id"], [])

            if not kids:
                terminals.append(parent_path)
                continue

            ranked_children = tournament_rank(
                agent=agent,
                state=state,
                option_cats=kids,
                children=children,
                stats=stats,
                label=f"node_{parent_path.leaf['id']}",
                max_options=max_options,
                keep=max(children_per_parent, 2),
            )

            for child, local_p, trace in ranked_children[:children_per_parent]:
                local_p = max(local_p, 1e-12)

                next_active.append(
                    BeamPath(
                        cats=parent_path.cats + [child],
                        log_score=parent_path.log_score + math.log(local_p),
                        local_probs=parent_path.local_probs + [local_p],
                        tournament_traces=parent_path.tournament_traces + [trace],
                    )
                )

        next_active = rank_paths_for_beam(next_active)[:beam_width]

        if not next_active:
            break

        active = next_active

        for i, path in enumerate(active, 1):
            print(
                f"  {i:2d}. geom={path.geometric_mean_probability:.4f} "
                f"joint={path.joint_probability:.6f} | "
                f"{path.path_text}"
            )

    all_paths = rank_paths_for_beam(terminals + active)

    # Dedupe by leaf ID; the taxonomy should already be a tree, but be defensive.
    seen = set()
    deduped: List[BeamPath] = []

    for path in all_paths:
        if path.leaf["id"] in seen:
            continue
        seen.add(path.leaf["id"])
        deduped.append(path)

    return deduped, terminals


# ---------------------------------------------------------------------------
# Optional final NOUL verification
# ---------------------------------------------------------------------------

def verify_paths(
    agent: RLAgent,
    state: Dict[str, Any],
    paths: List[BeamPath],
    stats: RuntimeStats,
    top_k: int,
) -> Dict[str, float]:
    if top_k <= 0 or not paths:
        return {}

    selected = paths[:top_k]
    questions: Dict[str, Any] = {}
    path_by_qid: Dict[str, BeamPath] = {}

    for i, path in enumerate(selected, start=1):
        qid = f"verify_{i:03d}_{path.leaf['id']}"
        path_by_qid[qid] = path

        questions[qid] = {
            "type": "noul",
            "instructions": {
                "question": (
                    "Is the supplied search-result corpus materially about this "
                    "complete IAB category path?"
                ),
                "iab_path": path.path_text,
                "leaf_category": path.leaf["name"],
                "focus": (
                    "Return true only when the leaf category itself is substantively "
                    "supported. Do not return true merely because an ancestor is relevant."
                ),
            },
            "criteria": {
                "true": (
                    "The complete path, including the leaf category, is materially "
                    "supported by the corpus."
                ),
                "false": (
                    "The leaf is unsupported, incidental, overly specific, or only "
                    "an ancestor category is relevant."
                ),
            },
        }

    answers = run_questions(
        agent=agent,
        state=state,
        questions=questions,
        stats=stats,
        label="final NOUL verification",
    )

    verified: Dict[str, float] = {}

    for qid, answer in answers.items():
        path = path_by_qid[qid]
        verified[path.leaf["id"]] = float(answer.get("noul", 0.0))

    return verified


# ---------------------------------------------------------------------------
# Main / reporting
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--taxonomy-path",
        type=Path,
        default=Path(__file__).resolve().parent / "iab_content_taxonomy_3_0.tsv",
    )
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument(
        "--max-choice-options",
        type=int,
        default=DEFAULT_MAX_CHOICE_OPTIONS,
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=DEFAULT_BEAM_WIDTH,
    )
    parser.add_argument(
        "--children-per-parent",
        type=int,
        default=DEFAULT_CHILDREN_PER_PARENT,
    )
    parser.add_argument(
        "--verify-top-k",
        type=int,
        default=DEFAULT_VERIFY_TOP_K,
    )

    args = parser.parse_args()

    if args.max_choice_options < 2 or args.max_choice_options > 10:
        raise ValueError(
            "--max-choice-options must be between 2 and 10 in this benchmark. "
            "This intentionally avoids the current choice:11+ bucket."
        )

    if args.children_per_parent < 1:
        raise ValueError("--children-per-parent must be >= 1")

    print("=" * 116)
    print("LAYA x IAB CONTENT TAXONOMY 3.0 — HIERARCHICAL CHOICE BENCHMARK")
    print("=" * 116)
    print(f"Python:             {sys.version.split()[0]}")
    print(f"PyTorch:            {torch.__version__}")
    print(f"CUDA build:         {torch.version.cuda}")
    print(f"CUDA available:     {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")
        print(f"GPU memory:          {props.total_memory / 1024**3:.2f} GiB")

    print()

    all_categories = load_iab_categories(args.taxonomy_path)

    if not (1 <= args.count <= len(all_categories)):
        raise ValueError(
            f"--count must be 1..{len(all_categories)}; got {args.count}"
        )

    categories = all_categories[:args.count]
    by_id, children, roots = build_tree(categories)

    tier_counts = defaultdict(int)
    for cat in categories:
        tier_counts[cat["tier"]] += 1

    max_children = 0
    max_child_parent = None
    for parent_id, kids in children.items():
        if len(kids) > max_children:
            max_children = len(kids)
            max_child_parent = by_id[parent_id]

    print(f"[taxonomy] official categories parsed: {len(all_categories)}")
    print(f"[taxonomy] categories used:            {len(categories)}")
    print(f"[taxonomy] roots:                      {len(roots)}")
    print(
        "[taxonomy] tier counts:                "
        + ", ".join(
            f"T{tier}={tier_counts[tier]}"
            for tier in sorted(tier_counts)
        )
    )

    if max_child_parent:
        print(
            f"[taxonomy] widest node:               "
            f"{max_child_parent['name']} -> {max_children} children"
        )

    print(f"[config] max Choice options:          {args.max_choice_options}")
    print(f"[config] beam width:                  {args.beam_width}")
    print(f"[config] children / parent kept:      {args.children_per_parent}")
    print(f"[config] final NOUL verify top-k:     {args.verify_top_k}")
    print()

    print(f"[model] loading from: {args.model_dir}")
    t0 = time.perf_counter()
    agent = RLAgent(str(args.model_dir))
    cuda_sync()
    load_seconds = time.perf_counter() - t0

    print(f"[model] device:       {agent.device}")
    print(f"[model] load time:    {load_seconds:.3f} s")
    print(f"[model] max_len:      {agent.cfg.get('max_len')}")
    print(f"[model] head_max_len: {agent.cfg.get('head_max_len')}")
    print(
        f"[model] choice 11+ T: "
        f"{agent.cfg.get('temperature_by_options', {}).get('choice:11+')}"
    )
    print()

    state = make_state()

    # Warm-up with 3 options.
    warmup_cats = roots[:3]
    warmup_q = {
        "warmup": make_choice_question(
            "Which category best fits the corpus?",
            warmup_cats,
            children,
        )
    }

    warmup_stats = RuntimeStats()
    _ = run_questions(
        agent,
        state,
        warmup_q,
        warmup_stats,
        "warmup",
    )

    clear_cuda()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    stats = RuntimeStats()
    wall_start = time.perf_counter()

    ranked_paths, terminal_paths = hierarchical_beam_search(
        agent=agent,
        state=state,
        roots=roots,
        children=children,
        stats=stats,
        max_options=args.max_choice_options,
        beam_width=args.beam_width,
        children_per_parent=args.children_per_parent,
    )

    verification = verify_paths(
        agent=agent,
        state=state,
        paths=ranked_paths,
        stats=stats,
        top_k=args.verify_top_k,
    )

    cuda_sync()
    wall_seconds = time.perf_counter() - wall_start

    peak_allocated = None
    peak_reserved = None

    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    print()
    print("=" * 116)
    print("FINAL HIERARCHICAL PATHS")
    print("=" * 116)

    for i, path in enumerate(ranked_paths[:30], start=1):
        verify_p = verification.get(path.leaf["id"])

        local = " × ".join(f"{p:.3f}" for p in path.local_probs)

        verify_text = (
            f" | verify={verify_p:.4f}"
            if verify_p is not None
            else ""
        )

        print(
            f"{i:3d}. "
            f"geom={path.geometric_mean_probability:.4f} | "
            f"joint={path.joint_probability:.6f}"
            f"{verify_text}\n"
            f"     local: {local}\n"
            f"     {path.path_text}"
        )

    print()
    print("=" * 116)
    print("BENCHMARK")
    print("=" * 116)

    mean_tokens = (
        stats.input_tokens / stats.questions
        if stats.questions
        else 0.0
    )

    print(f"Model load time:          {load_seconds:.3f} s")
    print(f"system_one() calls:       {stats.calls}")
    print(f"Choice/Noul questions:    {stats.questions}")
    print(f"Pure inference time:      {stats.inference_seconds:.3f} s")
    print(f"Wall benchmark time:      {wall_seconds:.3f} s")

    if stats.inference_seconds:
        print(
            f"Question throughput:      "
            f"{stats.questions / stats.inference_seconds:.2f} q/s"
        )
        print(
            f"Token throughput:         "
            f"{stats.input_tokens / stats.inference_seconds:,.0f} tok/s"
        )

    print(f"Input tokens processed:   {stats.input_tokens:,}")
    print(f"Mean tokens/question:     {mean_tokens:.1f}")

    if peak_allocated is not None:
        print(f"Peak CUDA allocated:      {peak_allocated:.2f} GiB")
        print(f"Peak CUDA reserved:       {peak_reserved:.2f} GiB")

    # Save everything.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = (
        Path(__file__).resolve().parent
        / f"laya_iab_hierarchical_{timestamp}.json"
    )

    payload = {
        "benchmark": "Laya IAB v3 hierarchical Choice beam search",
        "generated_at": datetime.now().isoformat(),
        "entity": "NVIDIA",
        "search_results": SEARCH_RESULTS,
        "taxonomy": {
            "source": IAB_V3_URL,
            "official_categories": len(all_categories),
            "used_categories": len(categories),
            "roots": len(roots),
            "tier_counts": dict(tier_counts),
            "widest_node": {
                "name": max_child_parent["name"] if max_child_parent else None,
                "children": max_children,
            },
        },
        "config": {
            "max_choice_options": args.max_choice_options,
            "beam_width": args.beam_width,
            "children_per_parent": args.children_per_parent,
            "verify_top_k": args.verify_top_k,
        },
        "runtime": {
            "model_load_seconds": load_seconds,
            "system_one_calls": stats.calls,
            "questions": stats.questions,
            "input_tokens": stats.input_tokens,
            "mean_tokens_per_question": mean_tokens,
            "inference_seconds": stats.inference_seconds,
            "wall_seconds": wall_seconds,
            "batches": stats.batches,
            "peak_cuda_allocated_gib": peak_allocated,
            "peak_cuda_reserved_gib": peak_reserved,
        },
        "paths": [
            {
                "leaf_id": p.leaf["id"],
                "leaf_name": p.leaf["name"],
                "tier": p.leaf["tier"],
                "path": [c["name"] for c in p.cats],
                "path_ids": [c["id"] for c in p.cats],
                "local_probabilities": p.local_probs,
                "joint_probability": p.joint_probability,
                "geometric_mean_probability": p.geometric_mean_probability,
                "verification_noul": verification.get(p.leaf["id"]),
                "tournament_traces": p.tournament_traces,
            }
            for p in ranked_paths
        ],
    }

    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
