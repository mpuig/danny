import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jev.data import Example, load_training_rows
from jev.rendering import CANDIDATE_READOUT, LETTER_READOUT, combine_views, render_views, resolve_readout
from jev.schema import Question
from jev.serialization import dumps


class CandidateTests(unittest.TestCase):
    def test_score_levels_do_not_see_siblings_or_indices(self):
        state = {"message": "Please help."}
        a = Question("score", "How urgent?", ["Immediate danger", "Needs help soon", "Can wait"])
        b = Question("score", "How urgent?", ["Immediate danger", "Something unrelated"])
        c = Question("score", "How urgent?", ["Can wait", "Immediate danger"])
        av = render_views(state, a, readout=CANDIDATE_READOUT)
        self.assertEqual(av[0], render_views(state,b,readout=CANDIDATE_READOUT)[0])
        self.assertEqual(av[0], render_views(state,c,readout=CANDIDATE_READOUT)[1])
        self.assertNotIn("Needs help soon", av[0][0])
        self.assertIn("Question type: score", av[0][0])

    def test_wide_choice_keeps_joint_alternatives_and_all_candidates(self):
        q = Question("choice", "Which?", {f"option-{i}": None for i in range(255)})
        views = render_views("evidence", q, readout=CANDIDATE_READOUT)
        self.assertEqual(len(views), 255)
        self.assertTrue(all(labels == [" A", " B"] for _, labels in views))
        self.assertIn('"option-254":null', views[0][0])
        probs = combine_views(q, [[0.75,0.25]] * 255, CANDIDATE_READOUT)
        self.assertEqual(len(probs),255)
        self.assertAlmostEqual(sum(probs),1)
        with self.assertRaises(ValueError):
            render_views("evidence",q,readout=LETTER_READOUT)

    def test_binary_targets_preserve_soft_mass_and_semantic_weight(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"data.jsonl"
            example=Example(id="a",group_id="a",source="fixture",state="evidence",
                            question=Question("score","How?",["low","middle","high"]),
                            target=[.2,.3,.5],target_origin="teacher",provenance={})
            path.write_text(dumps(example.to_dict())+'\n')
            rows=load_training_rows(path,readout_version=CANDIDATE_READOUT)
            self.assertEqual([r['target'] for r in rows],[[.8,.2],[.7,.3],[.5,.5]])
            self.assertEqual(sum(r['weight'] for r in rows),1)
            self.assertEqual(len({r['example_id'] for r in rows}),1)
            self.assertEqual(len({r['id'] for r in rows}),3)

    def test_adapter_readout_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'adapter_config.json'
            p.write_text(json.dumps({'renderer_version':'structured-v1','readout_version':CANDIDATE_READOUT}))
            self.assertEqual(resolve_readout(adapter_path=tmp),CANDIDATE_READOUT)
            with self.assertRaises(ValueError):
                resolve_readout(LETTER_READOUT,tmp)

    def test_zero_mass_falls_back_to_uniform(self):
        q=Question('score','?', ['a','b'])
        self.assertEqual(combine_views(q,[[1,0],[1,0]],CANDIDATE_READOUT),[.5,.5])


@unittest.skipUnless(importlib.util.find_spec('mlx') and os.environ.get('JEV_TEST_CANDIDATE'),
                     'set JEV_TEST_CANDIDATE for wide-choice model inference')
class CandidateModelTests(unittest.TestCase):
    def test_candidate_adapter_training_round_trip(self):
        from jev.engine import SystemOneEngine
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            for split, state, target in [('train','The server is down.',[0,0,1]),
                                         ('development','A minor display issue.',[1,0,0])]:
                e=Example(id=split,group_id=split,source='fixture',state=state,
                          question=Question('score','Severity?', ['Minor','Moderate','Severe']),
                          target=target,target_origin='gold',provenance={})
                (directory/f'{split}.jsonl').write_text(dumps(e.to_dict())+'\n')
            adapter=directory/'adapter'
            result=subprocess.run([sys.executable,str(root/'scripts/train_lora.py'),
                '--model','HuggingFaceTB/SmolLM2-135M','--train',str(directory/'train.jsonl'),
                '--val',str(directory/'development.jsonl'),'--out',str(adapter),
                '--readout',CANDIDATE_READOUT,'--batch-size','2','--max-steps','2'],
                cwd=root,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            engine=SystemOneEngine('HuggingFaceTB/SmolLM2-135M',adapter_path=str(adapter))
            self.assertEqual(engine.readout_version,CANDIDATE_READOUT)
            answer=engine.ask('A minor display issue.',{'q':e.question})['q']
            self.assertEqual(len(answer.probabilities),3)
            self.assertAlmostEqual(sum(answer.probabilities.values()),1)

    def test_255_options_are_evaluated_without_truncation(self):
        from jev.engine import SystemOneEngine
        engine=SystemOneEngine('HuggingFaceTB/SmolLM2-135M',readout_version=CANDIDATE_READOUT,
                               precision='float32',execution_mode='shared',max_batch_size=4)
        q=Question('choice','Select the named category.',{str(i):None for i in range(255)})
        answer=engine.ask({'category':'150'},{'wide':q})['wide']
        self.assertEqual(set(answer.probabilities),set(q.criteria))
        self.assertAlmostEqual(sum(answer.probabilities.values()),1)
        self.assertTrue(all(0<=p<=1 for p in answer.probabilities.values()))


if __name__ == '__main__':
    unittest.main()
