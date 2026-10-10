# Public decision benchmarks beyond the KEV suites

Verified on 2026-09-21. JevBench's 231 public decisions are integrated into the preparation tool and default H200 matrix. The other entries below are research candidates, not automatically downloaded.

The primary evaluation combines KEV and **JevBench's 231 public decisions** for explicit rule following and probability quality. **Typed Decisions' 400 test cases / 2,000 questions** is a secondary shared-state diagnostic because its labels measure teacher agreement. Image support is not a selection requirement. Additional objective intent/OOS and evidence-grounded candidates are assessed in [DECISION_BENCHMARK_SELECTION.md](DECISION_BENCHMARK_SELECTION.md). Keep each suite's results separate.

## Official TypeSafe data versus community benchmarks

No downloadable, labeled benchmark released by TypeSafe itself was located in its [official documentation](https://docs.typesafe.ai/introduction), [announcement](https://typesafe.ai/blog/introducing-system-one-models-and-jev), or [public GitHub organization](https://github.com/typesafe-ai). This is a search finding, not proof that no such data exists. TypeSafe publishes an [LLM comparison adapter](https://github.com/typesafe-ai/system-one-adapter-python), but an adapter is not an evaluation dataset.

All Jev-specific datasets below are independent community work. A repository name containing `Jev` or `TypeSafe` does not make it official. The community [TypeSafeAI playground](https://github.com/TypeSafeAI/typesafe-playground) explicitly says it is independent and its examples are not validated accuracy benchmarks.

KEV's current [4B model card](https://github.com/jaredpalmer/kev/blob/4f8110a3f8620cc3a182ae9a708e4398492c4b1a/docs/model-cards/kev-4b.md) already reports:

- `decision-v7`, `transfer-v4`, and newer `transfer-v9` evaluations;
- SemIf's 144 authored examples;
- scienthoon's 900 support tickets;
- ekzhang's 1,000-question MMLU-Pro sample.

SemIf, scienthoon, and that MMLU-Pro sample are therefore **external to KEV's original training suite, but not additional benchmarks that KEV has never run**. The referenced KEV suites are text based; they do not measure the vision encoder. “Never trained” in KEV's terminology concerns its own fine-tuning sources and is not evidence that a foundation model or Jev never saw the data.

## 1. JevBench public subset — first addition

[JevBench](https://github.com/fstandhartinger/jevbench) covers routing, policy checks, intent, enum extraction, ordinal severity, answer judging, and harder reasoning over supplied state. It explicitly disclaims TypeSafe affiliation. The full published evaluation has 534 decisions, while the downloadable public subset has **231**. Use a separate public-subset result rather than claiming reproduction of the full leaderboard.

Pinned repository: [`fd51755eb0c0b546ca206d764faf3302feca913e`](https://github.com/fstandhartinger/jevbench/tree/fd51755eb0c0b546ca206d764faf3302feca913e). Counts below were checked by parsing the pinned JSONL bytes.

| Download | Decisions | Choice | Noul | Score | License |
|---|---:|---:|---:|---:|---|
| [original.jsonl](https://raw.githubusercontent.com/fstandhartinger/jevbench/fd51755eb0c0b546ca206d764faf3302feca913e/datasets/public/original.jsonl) | 72 | 36 | 24 | 12 | MIT |
| [easy.jsonl](https://raw.githubusercontent.com/fstandhartinger/jevbench/fd51755eb0c0b546ca206d764faf3302feca913e/datasets/public/easy.jsonl) | 48 | 36 | 12 | 0 | MIT |
| [hard.jsonl](https://raw.githubusercontent.com/fstandhartinger/jevbench/fd51755eb0c0b546ca206d764faf3302feca913e/datasets/public/hard.jsonl) | 111 | 67 | 38 | 6 | MIT |
| Total | 231 | 139 | 74 | 18 | |

SHA-256 checksums, in the same order:

```text
5c2414edb3006b8bfcb70fda433f0f9ca015759433849f8d3104328a1f7c4180
231df3c2c8e88a1a8c137ebe85de96ba70fabd330849098ac7b3c52c70b7172b
89e9e6becb33ed88c1de7d42dcc87531b2fb64cfaef4e1986faf7c37b3f80ebb
```

Each row has `state`, one `question`, ordered `labels`, `expected`, `id`, `group`, and `provenance`. Send only `state` and `question`. Noul gold is `"no"`/`"yes"`; Score gold is an integer rubric index. The hard tier's `provenance` can contain the answer rationale and `gold_probs`: neither belongs in model input. Some hard cases have exact reference probabilities, unlike ordinary hard-label classification. The [hard-tier protocol](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/datasets/HARD-TIER.md) describes synthetic authoring, cross-model review, and the freeze.

The 303 unavailable decisions comprise 157 deliberately private items and 146 imported items whose task text the maintainers do not redistribute. Hashes and aggregate results do not reconstruct those inputs or grant redistribution rights. See [third-party terms](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/THIRD-PARTY.md) and the [license](https://github.com/fstandhartinger/jevbench/blob/fd51755eb0c0b546ca206d764faf3302feca913e/LICENSE).

Implementation notes: retain paraphrase `group` identifiers for grouped uncertainty estimates. Label our adapter's output as **token-logprob-derived**, since JevBench's published ordinary-LLM adapter uses verbalized probabilities. Its leaderboard also applies assumed latency adjustments to some endpoints; our H200 results should report actual local measurements without adopting those adjustments.

## 2. LocalLLaMA/typed-decisions — shared-state and soft labels

The [dataset card](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/blob/ea9306458d6e9563628369a3d1e72e362fb381d2/README.md) specifies four workflows: agent traces, customer service, invoices, and security incidents. Each has 300 train and 100 test cases, with five Choice/Noul/Score questions per case. Evaluate **400 test cases / 2,000 questions**. The `all` configuration repeats the four workflow configurations; do not concatenate both.

- Dataset revision: `ea9306458d6e9563628369a3d1e72e362fb381d2`.
- [Combined test Parquet](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/resolve/ea9306458d6e9563628369a3d1e72e362fb381d2/all/test-00000-of-00001.parquet), SHA-256 `4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c`.
- License: Apache-2.0, as declared in the pinned card.
- Parse JSON-string columns `state`, `questions`, and `gold`. Only the first two are model input; `factors` reveals the generating variables.

Gold averages three teacher-model probability samples. Consequently, this measures **agreement with a synthetic teacher**, not independently established correctness. Report hard-label agreement separately from soft-target Brier/KL and Score MAE. The teacher can be wrong, so a stronger model can receive a lower score. Its published Jev result is measured on this test set; its specialist baselines were trained on the corresponding training workflows.

## 3. NPC addressee benchmark — application behavior

[wondertwins/jev-benchmark](https://github.com/wondertwins/jev-benchmark/tree/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b) provides **79 hand-labeled utterances**, each with clean, punctuation-free STT, and misheard-name variants. That is 237 case variants, not 237 independent utterances. Its strict addressee metrics exclude four explicitly ambiguous originals. Questions combine per-NPC Noul, primary-addressee Choice, intent, and urgency Score.

- Pin: `1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b`.
- Fixtures: [npcaddress/dataset.py](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/npcaddress/dataset.py).
- Request construction: [npcaddress/questions.py](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/npcaddress/questions.py).
- [MIT license](https://github.com/wondertwins/jev-benchmark/blob/1c2509ac7d5508b8a7ce00ae4df6d7652d05de8b/LICENSE).

The same repository contains 30 chess positions and 25 mate-in-one puzzles, with cached Stockfish references. Treat chess as a separate reasoning stress test. Keep raw-board, code-enriched, and filtered hybrid variants distinct. These inputs are text/structured state, not screenshots.

## 4. Prompt-injection decisions — domain-specific addition

[jev-sec-bench](https://github.com/Gaurav-Gosain/jev-sec-bench/tree/fdb16b94d37535db9bad77f8ef0faa971bd7d69a) evaluated all **662** public `deepset/prompt-injections` messages. The original data has 546 train and 116 test rows. Reproducing the 662-row community experiment requires explicitly naming the combined split; it must not be reported as 662 held-out test examples. Its policy context matters: the source labels concern a news assistant, not an unrestricted chatbot.

[Dataset and Apache-2.0 declaration](https://huggingface.co/datasets/deepset/prompt-injections/tree/4f61ecb038e9c3fb77e21034b22511b523772cdd), pinned revision `4f61ecb038e9c3fb77e21034b22511b523772cdd`. Schema: `text`, binary `label`; map to Noul using the documented deployment context. Report F1, false positives/negatives, ROC-AUC, Brier, and latency. This is an application-specific classification benchmark, not proof of general guardrail security.

## Lower-priority or overlapping sources

[AbdelStark/jev-benchmarks](https://github.com/AbdelStark/jev-benchmarks/tree/0d610cc53e79bcbec691312b0c4adb4a0e371642) has a reproducible 300-example BTZSC pilot: AG News, Banking77, and Emotion. Those source families already occur in KEV training/evaluation, so the new harness is useful, but the data source is not independent of KEV's source selection. It pins BTZSC to `fef2a2ac62b69c58670047dddf045c53d7c3cb5e`; its Banking configuration has 72 available labels and excludes rows with no positive candidate. The harness is Apache-2.0; underlying datasets retain their own terms.

Small scenario collections such as [souvikr/jev-test](https://github.com/souvikr/jev-test) are useful smoke tests, but 17 cases / 23 checks are too small and easy to serve as the main accuracy benchmark. Documentation-derived instruction corpora and unlabeled live Jev outputs are not independent gold evaluation data.

## Future vision additions — general benchmarks, not Jev releases

### POPE: object-presence Noul

The [original POPE project](https://github.com/AoiDragon/POPE/tree/08d957b917e5a378a2f99d35b6293c536a66298b) supplies image filenames, questions, and yes/no labels. At the pinned revision, each of [random](https://raw.githubusercontent.com/AoiDragon/POPE/08d957b917e5a378a2f99d35b6293c536a66298b/output/coco/coco_pope_random.json), [popular](https://raw.githubusercontent.com/AoiDragon/POPE/08d957b917e5a378a2f99d35b6293c536a66298b/output/coco/coco_pope_popular.json), and [adversarial](https://raw.githubusercontent.com/AoiDragon/POPE/08d957b917e5a378a2f99d35b6293c536a66298b/output/coco/coco_pope_adversarial.json) contains **3,000 questions over 500 images**; counts were checked directly. These are related sampling variants, not three independent image datasets. Schema: `question_id`, `image`, `text`, `label`.

Use Noul for “is this object present?” and report each variant separately. The [repository license](https://github.com/AoiDragon/POPE/blob/08d957b917e5a378a2f99d35b6293c536a66298b/LICENSE) is MIT; COCO image rights remain separate. Obtain the named COCO 2014 images according to the [COCO terms](https://cocodataset.org/#termsofuse). Start with a deterministic image-grouped subset before a full run.

### VSR: spatial-relation Noul

The original [VSR benchmark](https://github.com/cambridgeltl/visual-spatial-reasoning/tree/b27a0af0ee1462d2b6b92c8c83e869d9254a241a) asks whether a caption describes the spatial relationship in an image. Recommend the [random test file](https://huggingface.co/datasets/cambridgeltl/vsr_random/resolve/b2053328fafdd018ff56cf1dfa9643caaa4e69b8/test.jsonl): **2,195** image/caption pairs, SHA-256 `8ade82a0b93ac9dc1e53f6cf1f11e9d5536776a3715102b6b4d27d4f81d551cc`. Its [official dataset card](https://huggingface.co/datasets/cambridgeltl/vsr_random/tree/b2053328fafdd018ff56cf1dfa9643caaa4e69b8) declares CC-BY-4.0; the implementation repository is Apache-2.0, and COCO image terms remain separate.

Schema includes `image`, `image_link`, `caption`, binary `label`, and `relation`. Fetch images using the original project's [image instructions](https://github.com/cambridgeltl/visual-spatial-reasoning/blob/b27a0af0ee1462d2b6b92c8c83e869d9254a241a/data/README.md); only image and caption belong in input. Score by relation and overall.

Version pitfall: the repository README's zero-shot table says 616 test examples, but the pinned GitHub and [HF zero-shot](https://huggingface.co/datasets/cambridgeltl/vsr_zeroshot/tree/148b3777ceef4a1bfe46377614980836db7d12f2) `test.jsonl` both contain **1,222** rows, SHA-256 `914d9156723b912d8f794b179248af865a68c5f81c9d8e3a21d007778cacaad4`. Do not copy the README count into a report for those files.

For vision timing, record image dimensions, processed image-token counts, resize policy, and cache condition. Measure the same fixed examples on every model. A no-image ablation tests image dependence but does not make the text-only task equally answerable. Keep prompt tuning, temperature calibration, and final test evaluation separate; public availability does not establish absence from any model's pretraining.
