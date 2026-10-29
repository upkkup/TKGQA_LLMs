# Template-based Knowledge Graph Q&A

A research-oriented KGQA subproject using question-template induction, deep-learning question-type classification, BiLSTM-CRF sequence tagging, SPARQL generation, and RDF execution.

## Dataset inspection

The existing repository dataset at ../data/questions.json contains 497 records with fields such as quid, question, answers, answer_type, time_level, qtype, and qlabel. The supplied dataset has 340 entity-answer and 157 time-answer questions. Observed qtype counts are:

| qtype | Count | Intended template |
|---|---:|---|
| equal | 198 | Exact fact / event lookup |
| first_last | 93 | Earliest or latest matching event |
| before_after | 79 | Event constrained by a date or temporal anchor |
| before_last | 56 | Latest event before an anchor |
| after_first | 56 | Earliest event after an anchor |
| equal_multi | 15 | Multiple answers in the same temporal context |

The data does not contain token-level BIO labels or explicit entity/relation/property spans. Therefore, the question-type classifier can be trained directly from qtype, but a BiLSTM-CRF tagger requires a separately annotated JSONL file. The code intentionally does not fabricate NER training labels from answer strings.

## Pipeline

1. Inspect questions and induce a template catalog from observed qtype labels.
2. Train a BiLSTM question-type classifier on the dataset.
3. Train the BiLSTM-CRF span tagger from manually reviewed token/tag annotations.
4. Extract actor, target, relation, property, date, and temporal-anchor spans; resolve entity mentions to graph resources.
5. Match the predicted question type to a template and fill its slots.
6. Generate a parameter-safe SPARQL query using a configurable event RDF schema.
7. Execute against a local RDF graph and return answer bindings, query text, and execution status.

Fact lookup and temporal-order templates are aligned to labels actually present in the dataset. Comparison and causal-inference templates are included as extension points, not as trained dataset classes: this dataset has no explicit comparison or causal-inference qtype labels.

## Setup (Windows / Anaconda)

From the repository root:

    conda create -n template-kgqa python=3.11 -y
    conda activate template-kgqa
    python -m pip install -r template_based_kgqa/requirements.txt
    set PYTHONPATH=template_based_kgqa/src

PowerShell equivalent: $env:PYTHONPATH="template_based_kgqa/src".

## Inspect questions and induce templates

    python -m template_kgqa.cli inspect --question "When did the al-Shabaab insurgency use unconventional violence against Muslims in the United Kingdom?"
    python -m template_kgqa.cli induce --dataset data/questions.json --output template_based_kgqa/templates.generated.json

## Train the question-type classifier

    python -m template_kgqa.train type --dataset data/questions.json --output template_based_kgqa/models/question_type.pt --epochs 12

## Train the BiLSTM-CRF tagger

Create a reviewed annotation file, one JSON object per line. Tags should use BIO labels such as B-ACTOR, I-ACTOR, B-TARGET, B-RELATION, B-PROPERTY, B-DATE, and B-ANCHOR.

    {"tokens":["When","did","John","praise","Mary","in","2012","?"],"tags":["O","O","B-ACTOR","B-RELATION","B-TARGET","O","B-DATE","O"]}

Then run:

    python -m template_kgqa.train ner --annotations data/ner_train.jsonl --output template_based_kgqa/models/entity_relation_tagger.pt --epochs 12

This example is illustrative only. Review labels against the annotation guide before training. Do not use test answers to create input features or tune a held-out evaluation.

## Configure and query an RDF graph

The initial query generator assumes an event-centric RDF schema with four configurable predicates: actor, relation, target, and date. The defaults use example.org and are placeholders, not claims about the schema of Wikidata, Freebase, or your local corpus.

    python -m template_kgqa.cli query --question "Who praised the government of Nigeria on 27 September 2012?" --rdf data/your_graph.ttl --actor "Ministry of Nigeria" --relation "praise" --target "government of Nigeria" --date "2012-09-27" --answer-role actor

Supported RDF formats depend on RDFLib (for example Turtle, N-Triples, RDF/XML, and JSON-LD). Pass the actual graph file and configure its predicate URIs in code/config before querying it. Freebase requires a local RDF dump or converted dataset. Wikipedia pages are not themselves a SPARQL endpoint; use a suitable RDF extraction or Wikidata representation. Wikidata's public endpoint has its own schema and requires a dedicated mapping adapter rather than reusing the example event predicates unchanged.

## Template families

- equal: direct fact lookup with optional exact date/month/year constraints.
- first_last: earliest/latest matching fact, sorted by event date.
- before_after: temporal filtering before/after a date.
- before_last: latest matching event before a resolved anchor event.
- after_first: earliest matching event after a resolved anchor event.
- equal_multi: retrieve all matching answers in the requested temporal context.
- comparison and causal inference: extension-only templates requiring suitable schema support and labeled examples.

## Current limitations

- Entity linking is a separate step; extracting a span does not prove that it maps to the correct graph URI.
- The supplied dataset alone cannot train a reliable entity/relation/property tagger because it lacks token-level span labels.
- Event schema predicates and date representations must match the actual RDF graph.
- Temporal first/last questions require sufficiently complete graph coverage; incomplete RDF data can produce misleading extrema.
- The code returns evidence bindings and query text where available. It does not claim that answers are correct when the graph or schema is missing.
