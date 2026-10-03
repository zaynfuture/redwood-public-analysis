import string
import unittest
from types import SimpleNamespace
from geoimpact.extract import constrained_prefix

class CharacterTokenizer:
    alphabet=list(dict.fromkeys(string.printable))
    all_special_ids=[len(alphabet)]
    eos_token_id=len(alphabet)
    def __len__(self): return len(self.alphabet)+1
    def encode(self,text,**kwargs): return [self.alphabet.index(c) for c in text]
    def decode(self,ids,**kwargs): return ''.join(self.alphabet[i] for i in ids if i<len(self.alphabet))

class ConstraintTests(unittest.TestCase):
    def test_enum_and_array_constraints_are_enforced(self):
        tokenizer=CharacterTokenizer()
        schema={'type':'object','properties':{'kind':{'type':'string','enum':['flood']},
                'ids':{'type':'array','items':{'type':'integer','enum':[0,2]},'minItems':1}},
                'required':['kind','ids'],'additionalProperties':False}
        fn=constrained_prefix(SimpleNamespace(tokenizer=tokenizer),schema)
        ids=tokenizer.encode('prompt')
        def allowed():return fn(0,SimpleNamespace(tolist=lambda:ids.copy()))
        allowed()
        for ch in '{"kind":"':
            token=tokenizer.encode(ch)[0]
            self.assertIn(token,allowed());ids.append(token)
        self.assertNotIn(tokenizer.encode('x')[0],allowed())
        for ch in 'flood","ids":[':
            token=tokenizer.encode(ch)[0]
            self.assertIn(token,allowed());ids.append(token)
        self.assertNotIn(tokenizer.encode('9')[0],allowed())
        for ch in '0]}':
            token=tokenizer.encode(ch)[0]
            self.assertIn(token,allowed());ids.append(token)
        self.assertIn(tokenizer.eos_token_id,allowed())

if __name__=='__main__': unittest.main()
