"""Behavior tests for silent corruption, contract drift, cIoU aggregation, budgets."""
import tempfile
import unittest
from pathlib import Path
from common import aggregate,atomic_json,budget_decision,digest,sha256,valid_record

class SafetyTests(unittest.TestCase):
    def test_global_iou_not_mean_chunk_iou(self):
        records=[dict(sample_id='a',intersection=[1],union=[1]),dict(sample_id='b',intersection=[0],union=[99])]
        result=aggregate(records)
        self.assertEqual(result['cIoU'],1)
        self.assertEqual(result['mIoU'],50)
        with self.assertRaises(ValueError): aggregate(records+[records[0]])

    def test_resume_requires_valid_record_and_mask_and_contract(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root=Path(tmp); mask=root/'mask.npz'; mask.write_bytes(b'compressed prediction fixture')
            record=dict(status='DONE',sample_id='a',fingerprint='model-one',sample_hash='sample-one',
                        intersection=[2],union=[3],expressions=[dict(ref_id=1)],mask_file='mask.npz',mask_sha256=sha256(mask))
            record['record_hash']=digest(record); path=root/'record.json'; atomic_json(path,record)
            self.assertIsNotNone(valid_record(path,'model-one','sample-one'))
            self.assertIsNone(valid_record(path,'model-two','sample-one'))
            self.assertIsNone(valid_record(path,'model-one','different-sample'))
            mask.write_bytes(b'truncated')
            self.assertIsNone(valid_record(path,'model-one','sample-one'))
            atomic_json(path,dict(record,union=[999]))
            self.assertIsNone(valid_record(path,'model-one','sample-one',check_masks=False))

    def test_budget_reserves_shutdown_and_disk(self):
        cfg=dict(SAFETY_FACTOR=1.5,SHUTDOWN_RESERVE_MIN=20,MIN_FREE_DISK_GB=3)
        self.assertEqual(budget_decision(500,2,1800,4500,10*1024**3,10000,100000,cfg)['status'],'REDUCE_SCOPE')
        self.assertEqual(budget_decision(50,2,1000,4500,10*1024**3,10000,100000,cfg)['status'],'BLOCKED')
        self.assertEqual(budget_decision(50,2,7200,4500,2*1024**3,10000,100000,cfg)['status'],'BLOCKED')
        self.assertEqual(budget_decision(500,2,7200,4500,10*1024**3,10000,None,cfg)['selected_max_images'],100)

if __name__=='__main__': unittest.main()
