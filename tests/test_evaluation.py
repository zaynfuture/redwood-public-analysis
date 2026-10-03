import unittest
from types import SimpleNamespace
import pandas as pd
import torch
from evaluate import evaluate_batches
from geoimpact.core import LABELS

class Batch(dict):
    def to(self,device):return self

class EvaluationTests(unittest.TestCase):
    def test_sample_weighted_loss_and_saved_label_mapping(self):
        mapping={name:5-i for i,name in enumerate(LABELS)}
        class Model:
            config=SimpleNamespace(label2id=mapping)
            calls=0
            def to(self,device):return self
            def eval(self):return self
            def __call__(self,labels,**kwargs):
                self.calls+=1
                self_grad=torch.is_grad_enabled()
                if self_grad:raise AssertionError('Evaluation enabled gradients')
                logits=torch.zeros(len(labels),6)
                logits[range(len(labels)),labels]=1
                return SimpleNamespace(loss=torch.tensor(1. if self.calls==1 else 3.),logits=logits)
        frame=pd.DataFrame({'text':['first','second','third'],'label':LABELS[:3], 'event_id':['a','b','c']})
        result=evaluate_batches(Model(),lambda *args,**kwargs:Batch(),frame,batch_size=2)
        self.assertAlmostEqual(result['eval_loss'],5/3)
        self.assertEqual(result['eval_accuracy'],1.)
        self.assertEqual([r['predicted'] for r in result['predictions']],LABELS[:3])
        self.assertEqual(result['sample_count'],3)
    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):evaluate_batches(None,None,pd.DataFrame({'text':['x']}),batch_size=0)

if __name__=='__main__':unittest.main()
