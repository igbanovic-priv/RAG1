import hashlib
import html
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

DEV_N = 500
TUNE_N = 600
DEV_SEED = 42
TUNE_SEED = 43

UNIQUE_POOL_N = 1200
UNIQUE_POOL_SEED = 41

OUT_DIR = Path(__file__).resolve().parent.parent / "data"


def stratified_indices(ds, n, seed, allowed_indices=None):
    rng = random.Random(seed)

    if allowed_indices is None:
        allowed_indices = list(range(len(ds)))

    groups = defaultdict(list)
    for i in allowed_indices:
        row = ds[i]
        groups[(row["type"], row["level"])].append(i)

    keys = sorted(groups.keys())
    total = len(allowed_indices)

    chosen = []
    for k in keys:
        idxs = groups[k]
        take = int(n * len(idxs) / total)  # floor
        rng.shuffle(idxs)
        chosen.extend(idxs[:take])

    need = n - len(chosen)
    if need > 0:
        chosen_set = set(chosen)
        leftovers = [i for i in allowed_indices if i not in chosen_set]
        rng.shuffle(leftovers)
        chosen.extend(leftovers[:need])

    rng.shuffle(chosen)
    return chosen[:n]


def write_jsonl(ds, indices, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in indices:
            f.write(json.dumps(ds[idx], ensure_ascii=False) + "\n")


def write_jsonl_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def verify_subset_file(path, expected_n):
    n_lines = 0
    dist = Counter()

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            obj = json.loads(line)
            dist[(obj.get("type"), obj.get("level"))] += 1

    ok = (n_lines == expected_n)
    return ok, n_lines, dist


def _norm_title(title: str) -> str:
    # Unescape HTML entities in titles too: &quot; -> "
    return html.unescape(title).strip()


def _norm_sentences(sentences):
    # Unescape HTML entities like &quot; -> "
    return [html.unescape(s).strip() for s in sentences]


def _sent_signature(sentences):
    sentences = _norm_sentences(sentences)
    blob = "\n".join(sentences).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def main():
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")

    # 1) Identify multi-variant titles across the whole dataset (after normalizing titles+sentences)
    title_to_sigs = defaultdict(set)
    for i in range(len(ds)):
        row = ds[i]
        titles = row["context"]["title"]
        sent_lists = row["context"]["sentences"]
        for title, sentences in zip(titles, sent_lists):
            title = _norm_title(title)
            title_to_sigs[title].add(_sent_signature(sentences))

    multi_variant_titles = {t for t, sigs in title_to_sigs.items() if len(sigs) > 1}

    # 2) Find "unique examples": examples whose titles are all single-variant
    unique_example_indices = []
    for i in range(len(ds)):
        row = ds[i]
        titles_in_example = {_norm_title(t) for t in row["context"]["title"]}
        if titles_in_example.isdisjoint(multi_variant_titles):
            unique_example_indices.append(i)

    if len(unique_example_indices) < UNIQUE_POOL_N:
        raise ValueError(
            f"Not enough unique examples to sample {UNIQUE_POOL_N}. Found only {len(unique_example_indices)}."
        )

    # 3) Choose 10,000 unique examples (pool)
    rng = random.Random(UNIQUE_POOL_SEED)
    rng.shuffle(unique_example_indices)
    pool_indices = unique_example_indices[:UNIQUE_POOL_N]

    # 4) Make a title-dedup corpus from those 10,000 (store normalized titles+sentences)
    title_to_sentences = {}
    for ex_i in pool_indices:
        row = ds[ex_i]
        titles = row["context"]["title"]
        sent_lists = row["context"]["sentences"]
        for title, sentences in zip(titles, sent_lists):
            title = _norm_title(title)

            # defensive: should never occur in the pool
            if title in multi_variant_titles:
                raise ValueError(f"Multi-variant title unexpectedly present in pool: {title!r}")

            sentences = _norm_sentences(sentences)

            if title not in title_to_sentences:
                title_to_sentences[title] = sentences
            else:
                # defensive: keep the longer one (for single-variant titles this should be identical)
                if len(sentences) > len(title_to_sentences[title]):
                    title_to_sentences[title] = sentences

    corpus_rows = [
        {"id": title, "title": title, "sentences": sents, "text": " ".join(sents)}
        for title, sents in title_to_sentences.items()
    ]
    corpus_rows.sort(key=lambda x: x["id"])

    corpus_path = OUT_DIR / f"hotpotqa_unique_pool{UNIQUE_POOL_N}_corpus_titles.jsonl"
    write_jsonl_rows(corpus_rows, corpus_path)

    # 5) Out of these 10,000 make dev500 and tune2000 (disjoint)
    dev_idx = stratified_indices(ds, DEV_N, DEV_SEED, allowed_indices=pool_indices)
    dev_set = set(dev_idx)
    remaining = [i for i in pool_indices if i not in dev_set]
    tune_idx = stratified_indices(ds, TUNE_N, TUNE_SEED, allowed_indices=remaining)

    dev_path = OUT_DIR / f"hotpotqa_unique_pool{UNIQUE_POOL_N}_dev{DEV_N}.jsonl"
    tune_path = OUT_DIR / f"hotpotqa_unique_pool{UNIQUE_POOL_N}_tune{TUNE_N}.jsonl"

    write_jsonl(ds, dev_idx, dev_path)
    write_jsonl(ds, tune_idx, tune_path)

    dev_ok, dev_lines, dev_dist = verify_subset_file(dev_path, DEV_N)
    tune_ok, tune_lines, tune_dist = verify_subset_file(tune_path, TUNE_N)

    print("Produced files:")
    print(" -", corpus_path, f"({len(corpus_rows)} title-docs)")
    print(" -", dev_path, f"({dev_lines} lines)")
    print(" -", tune_path, f"({tune_lines} lines)")
    print()

    print("Dev distribution (type, level):")
    for k, v in dev_dist.most_common():
        print(" ", k, ":", v)
    print()

    print("Tune distribution (type, level):")
    for k, v in tune_dist.most_common():
        print(" ", k, ":", v)
    print()

    if dev_ok and tune_ok:
        print("Verification OK: correct number of JSONL entries and stratification looks reasonable.")
    else:
        print("Verification FAILED:")
        if not dev_ok:
            print(f" - dev expected {DEV_N} lines, got {dev_lines}")
        if not tune_ok:
            print(f" - tune expected {TUNE_N} lines, got {tune_lines}")


if __name__ == "__main__":
    main()
