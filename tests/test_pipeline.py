import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from geoimpact.extract import extract, classify_batch

class Generator:
    tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:[1,2],pad_token_id=0)
    model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192))
    def __init__(self,output): self.output=output
    def __call__(self,*args,**kwargs): return [{'generated_text':self.output}]

class PipelineTests(unittest.TestCase):
    def setUp(self):
        patcher = patch('geoimpact.extract.constrained_prefix', return_value=lambda *args: [0])
        patcher.start()
        self.addCleanup(patcher.stop)
    def test_empty_events(self):
        self.assertEqual(extract('No news here',Generator('{"events":[]}')).events,[])
    def test_malformed(self):
        with self.assertRaises(ValueError): extract('A report about a warehouse closure.',Generator('Ignore the schema'))
    def test_sentence_evidence(self):
        draft={'title':'Flood','event_type':'natural_hazard','location_name':'Singapore',
               'evidence_sentence':0,'location_sentence':0,'reported_scope':'Warehouse closed',
               'possible_impacts':['If deliveries use this warehouse, delays are possible.']}
        text='Flooding in Singapore closed the coastal warehouse.'
        result=extract(text,Generator(json.dumps({'events':[draft]})))
        self.assertEqual(result.events[0].evidence_quote,text)
        draft['evidence_sentence']=99
        with self.assertRaises(ValueError):extract(text,Generator(json.dumps({'events':[draft]})))
    def test_repeated_place_mentions(self):
        draft={'title':'Flood','event_type':'natural_hazard','location_name':'Bangkok',
               'evidence_sentence':0,'location_sentence':None,'reported_scope':'Unknown','possible_impacts':[]}
        text='Flooding in Bangkok interrupted deliveries. Bangkok officials advised residents to remain home.'
        event=extract(text,Generator(json.dumps({'events':[draft]}))).events[0]
        self.assertEqual(event.location_quote,'Flooding in Bangkok interrupted deliveries.')
    def test_unsupported_location(self):
        draft={'title':'Flood','event_type':'natural_hazard','location_name':'Tokyo',
               'evidence_sentence':0,'location_sentence':0,'reported_scope':'Unknown','possible_impacts':[]}
        with self.assertRaises(ValueError):
            extract('Flooding in Singapore closed the coastal warehouse.',Generator(json.dumps({'events':[draft]})))
    def test_batch(self):
        classifier=lambda texts,**kwargs:[{'label':'other','score':.5} for text in texts]
        self.assertEqual(len(classify_batch(['a','b'],classifier)),2)
        self.assertEqual(classify_batch([],classifier),[])

if __name__ == '__main__': unittest.main()
