import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path
import sys
import benchmark as b

class Tests(unittest.TestCase):
    def test_sets(self):
        self.assertEqual(b.score([' A ','b','b'],['B','a'])['exact_match'],1)
        self.assertAlmostEqual(b.score(['a'],['a','b'])['f1'],2/3)
    def test_prompt_no_gold(self):
        row = dict(question='When?',answers=['SECRET'],answer_type='time',time_level='day')
        self.assertNotIn('SECRET',json.dumps(b.messages(row,'cot',[])))
    def test_parse(self):
        self.assertEqual(b.parse_answer('```json\n{"answers":["2005"]}\n```')['answers'],['2005'])
        with self.assertRaises(ValueError): b.parse_answer('{"answers":"2005"}')
    def test_retrieval(self):
        idx=b.build_retriever([dict(id='a',text='Garang praised Sudan in 2005'),dict(id='b',text='Brazil signed agreement')])
        self.assertEqual(b.retrieve(idx,'Garang Sudan',1)[0]['id'],'a')
    def test_end_to_end_mock(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds=Path(tmp)/'q.json'; ds.write_text(json.dumps([dict(quid=1,question='Year?',answers=['2005'])]))
            args=['benchmark.py','--dataset',str(ds),'--model','mock','--modes','io','sc','--samples','3','--output',str(Path(tmp)/'run')]
            with patch.object(sys,'argv',args),patch.object(b,'call_api',return_value={'choices':[{'message':{'content':'{"answers":["2005"]}'}}],'usage':{'prompt_tokens':10,'completion_tokens':5}}):
                b.main()
            summary=json.loads((Path(tmp)/'run.summary.json').read_text())
            self.assertEqual(summary['sc']['accuracy'],1)
            rows=[json.loads(s) for s in (Path(tmp)/'run.jsonl').read_text().splitlines()]
            self.assertEqual(rows[1]['prompt_tokens'],30)
            self.assertTrue(rows[1]['strict_majority'])
    def test_errors_are_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds=Path(tmp)/'q.json'; ds.write_text(json.dumps([dict(quid=1,question='Year?',answers=[])]))
            with patch.object(sys,'argv',['benchmark.py','--dataset',str(ds),'--model','mock','--modes','io','--output',str(Path(tmp)/'run')]),patch.object(b,'call_api',side_effect=RuntimeError('failed')):
                b.main()
            self.assertEqual(json.loads((Path(tmp)/'run.summary.json').read_text())['io']['accuracy'],0)

if __name__=='__main__': unittest.main()
