"""Build the corpus and evaluation set from the SQuAD v1.1 training file.

Source: SQuAD v1.1 (Stanford Question Answering Dataset), Wikipedia
articles with human written questions and gold answer spans.
Download page: https://rajpurkar.github.io/SQuAD-explorer/
License: CC BY-SA 4.0.

What this script does:
- Loads data/raw/train-v1.1.json (download it first, see README).
- Picks a seeded random sample of whole articles. Every paragraph of a
  picked article goes into the corpus, so picked articles contribute
  both the paragraphs questions were written about and distractor
  paragraphs from the same article.
- Writes data/corpus.jsonl: one document per article, paragraphs joined
  with a blank line.
- Writes data/eval_questions.jsonl: a seeded random sample of questions
  from the picked articles, with the gold answer converted to character
  offsets inside the article document.
- Writes data/dataset_info.json with the source, the seed, and counts.
"""

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=str(ROOT / "data/raw/train-v1.1.json"))
    parser.add_argument("--articles", type=int, default=60)
    parser.add_argument("--eval-questions", type=int, default=800)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    with open(args.raw, encoding="utf-8") as f:
        squad = json.load(f)

    articles = sorted(squad["data"], key=lambda a: a["title"])
    rng = random.Random(args.seed)
    picked = rng.sample(articles, min(args.articles, len(articles)))

    corpus = []
    all_questions = []
    for article in picked:
        title = article["title"]
        doc_id = "doc-" + title.lower().replace(" ", "_")
        paragraphs = []
        para_offsets = []
        offset = 0
        for para in article["paragraphs"]:
            para_offsets.append(offset)
            paragraphs.append(para["context"])
            offset += len(para["context"]) + 2  # "\n\n" separator
        corpus.append({
            "doc_id": doc_id,
            "title": title,
            "text": "\n\n".join(paragraphs),
        })
        for para, para_offset in zip(article["paragraphs"], para_offsets):
            for qa in para["qas"]:
                if not qa["answers"]:
                    continue
                answer = qa["answers"][0]
                start = para_offset + answer["answer_start"]
                all_questions.append({
                    "qid": qa["id"],
                    "doc_id": doc_id,
                    "title": title,
                    "question": qa["question"],
                    "answer_text": answer["text"],
                    "answer_start": start,
                    "answer_end": start + len(answer["text"]),
                })

    rng.shuffle(all_questions)
    eval_questions = all_questions[:args.eval_questions]

    data_dir = ROOT / "data"
    with (data_dir / "corpus.jsonl").open("w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    with (data_dir / "eval_questions.jsonl").open("w", encoding="utf-8") as f:
        for q in eval_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    info = {
        "dataset": "SQuAD v1.1 (Stanford Question Answering Dataset), "
                   "training split",
        "source_url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
                      "train-v1.1.json",
        "license": "CC BY-SA 4.0",
        "seed": args.seed,
        "articles_in_full_train_split": len(articles),
        "articles_sampled": len(corpus),
        "questions_available_for_sampled_articles": len(all_questions),
        "eval_questions_sampled": len(eval_questions),
        "article_titles_sampled": [d["title"] for d in corpus],
    }
    with (data_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(json.dumps({k: v for k, v in info.items()
                      if k != "article_titles_sampled"}, indent=2))


if __name__ == "__main__":
    main()
