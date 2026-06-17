#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import pickle
import random
import re
import shutil
from dataclasses import dataclass
from itertools import zip_longest
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class Candidate:
    item_id: str
    label: float
    orig_caption: str
    resample_caption: str
    orig_len: int
    resample_len: int
    len_diff: int


def parse_args() -> argparse.Namespace:
    root = "/data/Lushanhr/popularity/CopyGRPO"
    exp_root = os.path.join(root, "SocialMediaExp")
    parser = argparse.ArgumentParser(description="Generate paired A/B feeds without background posts")
    parser.add_argument("--merged-pkl", default=os.path.join(root, "data/ICIP/merged.pkl"))
    parser.add_argument("--split-json", default=os.path.join(root, "data/ICIP/split_811_seed2026.json"))
    parser.add_argument("--split-key", default="test")
    parser.add_argument("--resample-csv", default=os.path.join(root, "output/eval_resample_earlystop_fair_test.csv"))
    parser.add_argument("--resample-id-col", default="image_id")
    parser.add_argument("--resample-text-col", default="rewritten_caption")
    parser.add_argument("--num-posts", type=int, default=40)
    parser.add_argument("--group-size", type=int, default=20)
    parser.add_argument("--max-pop-fluct", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--media-base", default="https://media.githubusercontent.com/media/Lushanhr/SocialMediaExp/refs/heads/main/test_images_811")
    parser.add_argument("--img-src-dir", default=os.path.join("/data/Lushanhr/popularity", "data/ICIP/train_imgs"))
    parser.add_argument("--img-dst-dir", default=os.path.join(exp_root, "test_images_811_paired"))
    parser.add_argument("--copy-images", action="store_true", help="Copy selected images into dst dir")
    parser.add_argument("--output-selected-csv", default=os.path.join(exp_root, "selected_targets_paired.csv"))
    parser.add_argument("--output-feed-csv", default=os.path.join(exp_root, "social_feed_paired_test.csv"))
    parser.add_argument("--output-summary-json", default=os.path.join(exp_root, "paired_feed_summary.json"))
    return parser.parse_args()


def is_meaningful_caption(text: str) -> bool:
    if not isinstance(text, str):
        return False
    text = text.strip()
    if len(text) < 10:
        return False
    cleaned = re.sub(r"https?://\S+", "", text)
    cleaned = re.sub(r"#\S+", "", cleaned)
    cleaned = re.sub(r"@\S+", "", cleaned).strip()
    if len(cleaned) < 8:
        return False
    words = re.findall(r"[a-zA-Z]+", cleaned)
    if len(words) < 3:
        return False
    alpha_count = sum(c.isalpha() for c in cleaned)
    digit_count = sum(c.isdigit() for c in cleaned)
    if alpha_count < 5:
        return False
    if digit_count > alpha_count * 3:
        return False
    return True


def pop_fluct(labels: List[float]) -> float:
    if not labels:
        return 0.0
    lo = min(labels)
    hi = max(labels)
    return (hi - lo) / max(lo, 1e-9)


def select_posts(candidates: List[Candidate], num_posts: int, max_pop_fluct: float) -> Tuple[List[Candidate], str]:
    cand_sorted = sorted(candidates, key=lambda x: (x.len_diff, x.label))

    selected: List[Candidate] = []
    labels: List[float] = []
    for cand in cand_sorted:
        trial_labels = labels + [cand.label]
        if pop_fluct(trial_labels) <= max_pop_fluct:
            selected.append(cand)
            labels.append(cand.label)
        if len(selected) >= num_posts:
            return selected, "greedy"

    top_k = min(400, len(cand_sorted))
    by_label = sorted(cand_sorted[:top_k], key=lambda x: x.label)
    best_window: List[Candidate] = []
    left = 0
    for right in range(len(by_label)):
        while left < right and pop_fluct([c.label for c in by_label[left : right + 1]]) > max_pop_fluct:
            left += 1
        window = by_label[left : right + 1]
        if len(window) > len(best_window):
            best_window = window

    if len(best_window) >= num_posts:
        best_window_sorted = sorted(best_window, key=lambda x: (x.len_diff, x.label))
        return best_window_sorted[:num_posts], "sliding_window"

    # Fallback: keep best len_diff first, allow slight pop violation if unavoidable.
    fallback = cand_sorted[:num_posts]
    return fallback, "fallback_relaxed"


def _imbalance_score(g1: List[Candidate], g2: List[Candidate]) -> float:
    """Lower is better — measures total statistical imbalance between two groups."""
    if not g1 or not g2:
        return float("inf")

    def avg(arr: List[float]) -> float:
        return float(np.mean(arr))

    a_labels = [x.label for x in g1]
    b_labels = [x.label for x in g2]
    a_ol = [x.orig_len for x in g1]
    b_ol = [x.orig_len for x in g2]
    a_rl = [x.resample_len for x in g1]
    b_rl = [x.resample_len for x in g2]
    a_ld = [x.len_diff for x in g1]
    b_ld = [x.len_diff for x in g2]

    return (
        abs(avg(a_labels) - avg(b_labels))
        + 0.2 * abs(pop_fluct(a_labels) - pop_fluct(b_labels))
        + 0.3 * abs(avg(a_ol) - avg(b_ol))
        + 0.3 * abs(avg(a_rl) - avg(b_rl))
        + 0.4 * abs(avg(a_ld) - avg(b_ld))
    )


def split_balanced(selected: List[Candidate], group_size: int) -> Tuple[List[Candidate], List[Candidate]]:
    if len(selected) != group_size * 2:
        raise ValueError("selected size must be exactly 2 * group_size")

    n = len(selected)
    best_g1: List[Candidate] = []
    best_g2: List[Candidate] = []
    best_score = float("inf")

    # Try multiple random starts + the zigzag seed, keep the best split.
    rng = random.Random(12345)
    orderings: List[List[Candidate]] = []

    # Seed 0: zigzag by popularity (deterministic)
    ordered = sorted(selected, key=lambda x: x.label)
    zigzag: List[Candidate] = []
    for lo, hi in zip_longest(ordered[: n // 2], reversed(ordered[n // 2 :])):
        if hi is not None:
            zigzag.append(hi)
        if lo is not None:
            zigzag.append(lo)
    orderings.append(zigzag)

    # Seeds 1..49: random shuffles
    for _ in range(49):
        perm = list(selected)
        rng.shuffle(perm)
        orderings.append(perm)

    for ordering in orderings:
        g1: List[Candidate] = []
        g2: List[Candidate] = []
        for cand in ordering:
            if len(g1) >= group_size:
                g2.append(cand)
                continue
            if len(g2) >= group_size:
                g1.append(cand)
                continue
            s1 = _imbalance_score(g1 + [cand], g2)
            s2 = _imbalance_score(g1, g2 + [cand])
            if s1 <= s2:
                g1.append(cand)
            else:
                g2.append(cand)
        sc = _imbalance_score(g1, g2)
        if sc < best_score:
            best_score = sc
            best_g1 = list(g1)
            best_g2 = list(g2)

    return best_g1, best_g2


def stable_rng_from_id(item_id: str) -> random.Random:
    seed = int(hashlib.md5(item_id.encode("utf-8")).hexdigest()[:8], 16)
    return random.Random(seed)


def gen_item_meta(item_id: str, slot: int) -> Dict[str, object]:
    rng = stable_rng_from_id(item_id)
    username = f"User{slot:02d}"
    return {
        "likes": rng.randint(15, 85),
        "reposts": rng.randint(0, 7),
        "replies": rng.randint(0, 4),
        "datetime": f"{rng.randint(1,28):02d}.{rng.randint(1,12):02d}.26 {rng.randint(0,23):02d}:{rng.choice([0,15,30,45]):02d}",
        "username": username,
        "handle": f"@{username.lower()}",
        "user_description": "Sharing moments that matter",
        "user_image": f"https://ui-avatars.com/api/?name={username}&size=150&background=random",
        "user_followers": rng.randint(100, 15000),
    }


def detect_image_ext(item_id: str, src_dir: str) -> str:
    for ext in [".jpg", ".jpeg", ".png"]:
        if os.path.exists(os.path.join(src_dir, f"{item_id}{ext}")):
            return ext
    return ".jpg"


def build_feed_rows(group: List[Candidate], group_idx: int, seed: int, media_base: str, img_src_dir: str) -> List[Dict[str, object]]:
    rng = random.Random(seed + group_idx * 1000)
    order = list(range(len(group)))
    rng.shuffle(order)
    sequence_map = {idx: pos + 1 for pos, idx in enumerate(order)}

    half = len(group) // 2
    origin_in_a = set(rng.sample(range(len(group)), half))

    rows: List[Dict[str, object]] = []
    condition_a = f"pair_g{group_idx + 1}_A"
    condition_b = f"pair_g{group_idx + 1}_B"

    for idx, cand in enumerate(group):
        ext = detect_image_ext(cand.item_id, img_src_dir)
        media = f"{media_base}/{cand.item_id}{ext}"
        meta = gen_item_meta(cand.item_id, slot=group_idx * len(group) + idx + 1)
        seq = sequence_map[idx]
        doc_id = idx + 1

        a_is_origin = idx in origin_in_a
        text_a = cand.orig_caption if a_is_origin else cand.resample_caption
        text_b = cand.resample_caption if a_is_origin else cand.orig_caption

        base = {
            "doc_id": doc_id,
            "datetime": meta["datetime"],
            "media": media,
            "alt_text": "",
            "likes": meta["likes"],
            "reposts": meta["reposts"],
            "replies": meta["replies"],
            "username": meta["username"],
            "handle": meta["handle"],
            "user_description": meta["user_description"],
            "user_image": meta["user_image"],
            "user_followers": meta["user_followers"],
            "commented_post": 0,
            "sponsored": 0,
            "target": "",
            "sequence": seq,
            "item_id": cand.item_id,
            "version": "origin" if a_is_origin else "resample",
        }

        row_a = dict(base)
        row_a["condition"] = condition_a
        row_a["text"] = text_a.strip().replace(" nan", "").replace("nan", "")

        row_b = dict(base)
        row_b["condition"] = condition_b
        row_b["text"] = text_b.strip().replace(" nan", "").replace("nan", "")
        row_b["version"] = "resample" if a_is_origin else "origin"

        rows.append(row_a)
        rows.append(row_b)

    return rows


def maybe_copy_images(selected: List[Candidate], src_dir: str, dst_dir: str) -> int:
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(dst_dir):
        path = os.path.join(dst_dir, name)
        if os.path.isfile(path):
            os.remove(path)
    copied = 0
    for cand in selected:
        for ext in [".jpg", ".jpeg", ".png"]:
            src = os.path.join(src_dir, f"{cand.item_id}{ext}")
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
                copied += 1
                break
    return copied


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.merged_pkl, "rb") as f:
        merged = pickle.load(f)
    with open(args.split_json, "r", encoding="utf-8") as f:
        split = json.load(f)

    resample_df = pd.read_csv(args.resample_csv)
    if args.resample_id_col not in resample_df.columns or args.resample_text_col not in resample_df.columns:
        raise ValueError(
            f"resample csv must contain columns: {args.resample_id_col}, {args.resample_text_col}; got {resample_df.columns.tolist()}"
        )

    merged_dict: Dict[str, Dict[str, object]] = {}
    for item in merged:
        iid = str(item["item_id"])
        cap = item.get("cban_text", "") or item.get("skapp_text", "") or item.get("image_caption", "")
        merged_dict[iid] = {"label": float(item["label"]), "caption": str(cap) if cap else ""}

    test_ids = set(str(x) for x in split[args.split_key])
    resample_map = dict(
        zip(resample_df[args.resample_id_col].astype(str), resample_df[args.resample_text_col].astype(str))
    )

    candidates: List[Candidate] = []
    for iid in test_ids:
        if iid not in merged_dict or iid not in resample_map:
            continue
        orig = (merged_dict[iid]["caption"] or "").strip()
        resample = (resample_map[iid] or "").strip()
        if not orig or not resample:
            continue
        if not is_meaningful_caption(orig) or not is_meaningful_caption(resample):
            continue
        candidates.append(
            Candidate(
                item_id=iid,
                label=float(merged_dict[iid]["label"]),
                orig_caption=orig,
                resample_caption=resample,
                orig_len=len(orig),
                resample_len=len(resample),
                len_diff=abs(len(orig) - len(resample)),
            )
        )

    if len(candidates) < args.num_posts:
        raise RuntimeError(f"Not enough candidates after filtering: {len(candidates)} < {args.num_posts}")

    selected, strategy = select_posts(candidates, args.num_posts, args.max_pop_fluct)
    selected_labels = [x.label for x in selected]

    if len(selected) != args.num_posts:
        raise RuntimeError(f"Selection size mismatch: got {len(selected)} expected {args.num_posts}")

    if args.group_size * 2 != args.num_posts:
        raise ValueError("group-size must be num-posts/2 for this paired setup")

    g1, g2 = split_balanced(selected, args.group_size)

    rows = []
    rows.extend(build_feed_rows(g1, 0, args.seed, args.media_base, args.img_src_dir))
    rows.extend(build_feed_rows(g2, 1, args.seed, args.media_base, args.img_src_dir))

    df_rows = pd.DataFrame(rows)

    # Hard checks: exact mirrored versions in A/B within each group and same sequence.
    for group_idx in [1, 2]:
        a = df_rows[df_rows["condition"] == f"pair_g{group_idx}_A"].sort_values("doc_id")
        b = df_rows[df_rows["condition"] == f"pair_g{group_idx}_B"].sort_values("doc_id")
        if not np.array_equal(a["item_id"].values, b["item_id"].values):
            raise RuntimeError(f"Group {group_idx}: item_id mismatch between A and B")
        if not np.array_equal(a["sequence"].values, b["sequence"].values):
            raise RuntimeError(f"Group {group_idx}: sequence mismatch between A and B")
        if np.any(a["version"].values == b["version"].values):
            raise RuntimeError(f"Group {group_idx}: found non-swapped versions")

    os.makedirs(os.path.dirname(args.output_selected_csv), exist_ok=True)
    selected_df = pd.DataFrame(
        [
            {
                "item_id": x.item_id,
                "label": x.label,
                "orig_len": x.orig_len,
                "resample_len": x.resample_len,
                "len_diff": x.len_diff,
                "orig_caption": x.orig_caption,
                "resample_caption": x.resample_caption,
                "group": 1 if x.item_id in {c.item_id for c in g1} else 2,
            }
            for x in selected
        ]
    )
    selected_df.to_csv(args.output_selected_csv, index=False)

    feed_cols = [
        "doc_id",
        "datetime",
        "text",
        "media",
        "alt_text",
        "likes",
        "reposts",
        "replies",
        "username",
        "handle",
        "user_description",
        "user_image",
        "user_followers",
        "commented_post",
        "sponsored",
        "target",
        "condition",
        "sequence",
        "item_id",
        "version",
    ]
    df_rows[feed_cols].to_csv(args.output_feed_csv, sep=";", index=False)

    copied = None
    if args.copy_images:
        copied = maybe_copy_images(selected, args.img_src_dir, args.img_dst_dir)

    def group_stats(group: List[Candidate]) -> Dict[str, float]:
        labels = [x.label for x in group]
        return {
            "size": len(group),
            "pop_min": float(np.min(labels)),
            "pop_max": float(np.max(labels)),
            "pop_fluct": pop_fluct(labels),
            "orig_len_mean": float(np.mean([x.orig_len for x in group])),
            "resample_len_mean": float(np.mean([x.resample_len for x in group])),
            "len_diff_mean": float(np.mean([x.len_diff for x in group])),
        }

    summary = {
        "selection_strategy": strategy,
        "num_candidates": len(candidates),
        "num_selected": len(selected),
        "global_pop_fluct": pop_fluct(selected_labels),
        "global_pop_min": float(np.min(selected_labels)),
        "global_pop_max": float(np.max(selected_labels)),
        "group_1": group_stats(g1),
        "group_2": group_stats(g2),
        "feeds": {
            "pair_g1_A": int((df_rows["condition"] == "pair_g1_A").sum()),
            "pair_g1_B": int((df_rows["condition"] == "pair_g1_B").sum()),
            "pair_g2_A": int((df_rows["condition"] == "pair_g2_A").sum()),
            "pair_g2_B": int((df_rows["condition"] == "pair_g2_B").sum()),
        },
    }
    if copied is not None:
        summary["copied_images"] = copied
        summary["image_dst_dir"] = args.img_dst_dir

    with open(args.output_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Paired feed generation complete")
    print(f"  selected csv: {args.output_selected_csv}")
    print(f"  feed csv: {args.output_feed_csv}")
    print(f"  summary json: {args.output_summary_json}")
    print(f"  selection strategy: {strategy}")
    print(f"  global pop fluctuation: {summary['global_pop_fluct']*100:.2f}%")
    if copied is not None:
        print(f"  copied images: {copied}")


if __name__ == "__main__":
    main()
