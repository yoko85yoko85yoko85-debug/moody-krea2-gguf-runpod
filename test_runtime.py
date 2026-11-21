"""CPU-only contract tests. No network, model download, ComfyUI or GPU."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import handler
import runtime


class ContractTests(unittest.TestCase):
    def test_workflow_and_request(self):
        p=runtime.validate({'prompt':'A ceramic mug.', 'width':1024,'height':768,'seed':17})
        g=runtime.graph(p)
        self.assertEqual(g['6']['inputs']['seed'],17)
        self.assertEqual(g['5']['inputs']['width'],1024)
        self.assertEqual(g['3']['inputs']['text'],'A ceramic mug.')
        for node in g.values():
            for value in node['inputs'].values():
                if isinstance(value,list):self.assertIn(value[0],g)
        self.assertEqual(runtime.graph(runtime.validate({'prompt':'other'}))['3']['inputs']['text'],'other')

    def test_invalid_requests_never_start_engine(self):
        with patch.object(runtime,'generate') as generate:
            for p in [{'prompt':''},{'prompt':'x','width':641},{'prompt':'x','seed':True},
                      {'prompt':'x','cfg':float('nan')},{'prompt':'x','width':1536,'height':1536},
                      {'prompt':'x','workflow':{}},{'prompt':'x','steps':100},None]:
                self.assertIn('error',handler.handler({'input':p}))
            generate.assert_not_called()

    def test_diagnostic_never_downloads_or_starts_gpu(self):
        with patch.object(runtime,'ensure_models',side_effect=AssertionError),patch.object(runtime,'ready',side_effect=AssertionError):
            d=handler.handler({'input':{'diagnose':True}})
        self.assertTrue(d['ok']);self.assertFalse(d['downloads_started_by_diagnostic'])
        self.assertEqual(d['planned_model_bytes'],12714045638)

    def test_workflow_output_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'output').mkdir();file=root/'output'/'test.png';file.write_bytes(b'png-test')
            calls=[]
            def call(path,payload=None,**kw):
                calls.append((path,payload))
                if path=='/prompt':return {'prompt_id':'job-1'}
                if path=='/history/job-1':return {'job-1':{'status':{'status_str':'success'},'outputs':{'8':{'images':[{'filename':'test.png'}]}}}}
                return {}
            with patch.object(runtime,'COMFY_DIR',root),patch.object(runtime,'ready'),patch.object(runtime,'call',side_effect=call),patch.object(runtime,'process',SimpleNamespace(poll=lambda:None)):
                out=runtime.generate(runtime.validate({'prompt':'A mug.'}))
            self.assertEqual(out['image_base64'],'cG5nLXRlc3Q=');self.assertFalse(file.exists())
            self.assertIn(('/history',{'delete':['job-1']}),calls)

    def test_diagnostic_error_redacts_secrets(self):
        import os
        with patch.dict(os.environ, {'HF_TOKEN':'hf_examplePrivateToken123', 'RUNPOD_API_KEY':'rpa_examplePrivateKey123'}):
            d=runtime.diagnostic_error(RuntimeError('HF hf_examplePrivateToken123 / rpa_examplePrivateKey123 / https://example.com/?token=secret'))
        self.assertNotIn('examplePrivate',json.dumps(d))
        self.assertNotIn('?token=',json.dumps(d))
        self.assertIn('phase',d);self.assertIn('detail',d);self.assertIn('engine_log_tail',d)

    def test_model_manifest_is_pinned(self):
        m=runtime.load_manifest()['models'];self.assertEqual(len(m),3)
        for item in m:
            self.assertEqual(len(item['sha256']),64);self.assertEqual(len(item['revision']),40)
            self.assertGreater(item['bytes'],1000000)
        self.assertEqual(m[0]['filename'],'moodyKrea2Mix_v40-Q4_K.gguf')

    def test_insufficient_disk_before_download(self):
        import sys
        fake=SimpleNamespace(hf_hub_download=lambda **kw:self.fail('Download should not start'))
        with tempfile.TemporaryDirectory() as temp,patch.dict(sys.modules,{'huggingface_hub':fake}),patch.object(runtime,'MODEL_ROOT',Path(temp)),patch.object(runtime.shutil,'disk_usage',return_value=SimpleNamespace(free=100)):
            with self.assertRaisesRegex(RuntimeError,'Insufficient'):runtime.ensure_models()


if __name__=='__main__':unittest.main()
