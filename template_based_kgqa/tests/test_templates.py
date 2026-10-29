import unittest

from template_kgqa.pipeline import question_to_sparql
from template_kgqa.sparql import build_event_query
from template_kgqa.templates import infer_qtype, template_for


class TemplateTests(unittest.TestCase):
    def test_temporal_types(self):
        self.assertEqual(infer_qtype("Who was the last to praise X before Y?"), "before_last")
        self.assertEqual(infer_qtype("Who was the first to visit X after Y?"), "after_first")
        self.assertEqual(infer_qtype("Who praised X before 2012?"), "before_after")
        self.assertEqual(infer_qtype("When did X praise Y?"), "equal")

    def test_catalog_has_dataset_templates(self):
        for qtype in ("equal", "first_last", "before_after", "before_last", "after_first", "equal_multi"):
            self.assertTrue(template_for(qtype)["name"])

    def test_query_uses_safe_literal_and_configured_predicates(self):
        query = build_event_query(
            "equal",
            {"actor": 'A "quoted" actor', "relation": "praise", "target": "government", "date": "2012"},
            answer_role="actor",
            schema={"actor_predicate": "https://example.org/p/actor"},
        )
        self.assertIn("https://example.org/p/actor", query)
        self.assertIn(r'a \"quoted\" actor', query)
        self.assertIn('STRSTARTS(STR(?date), "2012")', query)

    def test_question_to_sparql_returns_template_and_query(self):
        result = question_to_sparql("When did John praise Mary in 2012?", {"actor": "John", "target": "Mary"}, "date")
        self.assertEqual(result["qtype"], "equal")
        self.assertIn("SELECT DISTINCT ?date", result["sparql"])
        self.assertEqual(result["slots"]["date"], "2012")


if __name__ == "__main__":
    unittest.main()
