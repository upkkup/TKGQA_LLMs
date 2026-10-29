# Reviewed BIO training and joint inference

This addition connects the reviewed BIO labels to the existing PyTorch BiLSTM-CRF, question-type classifier, template catalog, and event-centric SPARQL generator.

## Important data-quality gate

The BIO files produced automatically are silver labels. Do not treat them as gold labels. Review the token-level CSV first. Only questions explicitly marked as reviewed in a separate manifest are used for training or evaluation.

## 1. Create a review manifest

From the repository root:

\`\`\`bat
set PYTHONPATH=%CD%\template_based_kgqa\src
python template_based_kgqa\tools\prepare_bio_review_manifest.py --annotations template_based_kgqa\bio_annotations.jsonl --output template_based_kgqa\bio_review_manifest.csv
\`\`\`

The input path above is an example: place your downloaded \`bio_annotations.jsonl\` at that location or substitute its actual path.

Open \`bio_review_manifest.csv\`. After you have checked and corrected every token label for a question in the token review CSV, set that question's \`review_status\` to \`reviewed\`. Leave unreviewed questions blank. The manifest has one row per question.

## 2. Train/evaluate BiLSTM-CRF

\`\`\`bat
python -m template_kgqa.train_ner_reviewed --annotations PATH_TO_bio_annotations_token_review.csv --manifest PATH_TO_bio_review_manifest.csv --output template_based_kgqa\models\entity_relation_crf.pt --metrics-output template_based_kgqa\models\entity_relation_crf.metrics.json --epochs 15 --test-fraction 0.2 --seed 42
\`\`\`

The script:
- uses only questions marked reviewed;
- validates BIO sequences and token indices;
- splits by question before training;
- builds the vocabulary from training questions only;
- reports token accuracy, non-O macro F1, per-tag precision/recall/F1, and exact-span precision/recall/F1;
- saves the checkpoint and held-out metrics.

The test set is a random question-level split, not an external benchmark. If only a small number of questions are reviewed, regard metrics as preliminary. Do not tune against the held-out split.

## 3. Run joint inference

Use the trained type checkpoint from the held-out classifier evaluation and the newly trained CRF checkpoint:

\`\`\`bat
python -m template_kgqa.predict_pipeline --question "When did John Garang praise Sudan People's Liberation Army in 2005?" --type-checkpoint template_based_kgqa\models\question_type_heldout.pt --ner-checkpoint template_based_kgqa\models\entity_relation_crf.pt
\`\`\`

Output includes the predicted question type and confidence, selected template, BIO tags, extracted spans, candidate slots, answer role, and SPARQL when sufficient slots are available. Add \`--force-query\` to generate a query when slots are incomplete.

## 4. Optional local RDF execution

\`\`\`bat
python -m template_kgqa.predict_pipeline --question "When did John Garang praise Sudan People's Liberation Army in 2005?" --type-checkpoint template_based_kgqa\models\question_type_heldout.pt --ner-checkpoint template_based_kgqa\models\entity_relation_crf.pt --rdf data\events.ttl --schema-json data\event_schema.json
\`\`\`

The RDF file must use the event schema configured in \`event_schema.json\`. The default predicates in the existing SPARQL module are placeholders under \`example.org\`; they will not query Wikidata, Freebase, or an arbitrary RDF file correctly without a matching schema and entity/predicate mapping.

## Current limitations

- The question-type classifier predicts the six dataset labels: \`equal\`, \`first_last\`, \`before_after\`, \`before_last\`, \`after_first\`, and \`equal_multi\`.
- The CRF labels are \`ENTITY\`, \`RELATION\`, \`TIME\`, and \`TEMPORAL_OPERATOR\` in BIO format.
- Assigning actor/target roles is a transparent positional heuristic, not a separately trained role classifier.
- Entity linking, predicate normalization, and a verified RDF schema remain necessary for reliable real-world graph queries.
- Exact-span metrics are only meaningful as gold evaluation when the held-out labels have been manually reviewed.
