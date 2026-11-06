import unittest
from types import SimpleNamespace
import numpy as np
from quantization.operators import audit_operators


class OperatorAuditTests(unittest.TestCase):
    def test_unknown_operators_and_micro_quantization_constraints_are_flagged(self):
        tensors = [{'index':i,'dtype':np.int8,'quantization':(.25,0)} for i in range(3)]
        ops = [{'op_name':'CONCATENATION','inputs':[0,1],'outputs':[2]},
               {'op_name':'REDUCE_MAX','inputs':[0,42],'outputs':[2]},
               {'op_name':'DELEGATE'}]
        interpreter = SimpleNamespace(_get_ops_details=lambda:ops,get_tensor_details=lambda:tensors)
        runner = SimpleNamespace(interpreter=interpreter,input={'shape':[1,148,40,1]})
        audit = audit_operators(runner)
        self.assertEqual(audit['status'],'static_checks_passed')
        self.assertNotIn('DELEGATE',audit['operators'])
        self.assertFalse(audit['firmware_invocation_verified'])
        tensors[1]['quantization'] = (.5,0)
        self.assertEqual(audit_operators(runner)['status'],'requires_review')
        tensors[1]['quantization'] = (.25,0)
        ops.append({'op_name':'CUSTOM','inputs':[0],'outputs':[2]})
        audit = audit_operators(runner)
        self.assertEqual(audit['unreviewed_operators'],['CUSTOM'])
        self.assertEqual(audit['status'],'requires_review')
