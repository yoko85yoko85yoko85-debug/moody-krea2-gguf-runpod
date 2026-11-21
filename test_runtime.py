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

    def test_standalone_modulation_expanded_only_when_needed(self):
        import textwrap
        import patch_gguf
        packed=SimpleNamespace(shape=(2,3456),tensor_shape=(2,6144))
        dense=SimpleNamespace(shape=(2,6144))
        for arch,key,quantized,expected in [('krea2','last.modulation.lin',True,dense),
                    ('krea2','last.modulation.lin',False,packed),
                    ('krea2','blocks.0.attn.wq.weight',True,packed),
                    ('flux','last.modulation.lin',True,packed)]:
            state={key:packed};calls=[]
            def dequantize(value,dtype):
                self.assertIs(value,packed);calls.append(dtype);return dense
            env=dict(arch_str=arch,sd_key=key,state_dict=state,
                     is_quantized=lambda _:quantized,dequantize_tensor=dequantize,
                     torch=SimpleNamespace(float32='f32'))
            exec(textwrap.dedent(patch_gguf.PATCH),env)
            self.assertIs(state[key],expected)
            self.assertEqual(len(calls),int(expected is dense))
        source='before\n'+patch_gguf.ANCHOR+'after\n'
        patched=patch_gguf.patch_source(source)
        self.assertEqual(patch_gguf.patch_source(patched),patched)
        with self.assertRaises(RuntimeError):patch_gguf.patch_source('unsupported loader')
        docker=(runtime.HERE/'Dockerfile').read_text()
        self.assertIn('RUN python /app/patch_gguf.py',docker)

    def test_http_empty_success_response(self):
        import io
        with patch.object(runtime.urllib.request,'urlopen',return_value=io.BytesIO(b'')):
            self.assertEqual(runtime.call('/history',{'delete':['job-1']}),{})
        with patch.object(runtime.urllib.request,'urlopen',return_value=io.BytesIO(b'{"ok":true}')):
            self.assertEqual(runtime.call('/history/job-1'),{'ok':True})
        with patch.object(runtime.urllib.request,'urlopen',return_value=io.BytesIO(b'invalid')):
            with self.assertRaises(ValueError):runtime.call('/history/job-1')

    def test_output_survives_history_cleanup_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'output').mkdir();file=root/'output'/'test.png';file.write_bytes(b'png-test')
            def call(path,payload=None,**kw):
                if path=='/prompt':return {'prompt_id':'job-1'}
                if path=='/history/job-1':return {'job-1':{'outputs':{'8':{'images':[{'filename':'test.png'}]}}}}
                raise ValueError('simulated housekeeping failure')
            with patch.object(runtime,'COMFY_DIR',root),patch.object(runtime,'ready'),patch.object(runtime,'call',side_effect=call),patch.object(runtime,'process',SimpleNamespace(poll=lambda:None)):
                out=runtime.generate(runtime.validate({'prompt':'A mug.'}))
            self.assertEqual(out['image_base64'],'cG5nLXRlc3Q=')
            self.assertFalse(file.exists())

    def source_input(self, **changes):
        import io,base64
        from PIL import Image
        image=Image.new('RGB',(100,200),'red');buf=io.BytesIO();image.save(buf,format='PNG')
        return {'prompt':'A watercolor painting of the subject.', 'mode':'i2i', 'image_base64':base64.b64encode(buf.getvalue()).decode(), **changes}

    def test_i2i_padding_and_graph(self):
        import io
        from PIL import Image
        p=runtime.validate(self.source_input(denoise=.45))
        image=Image.open(io.BytesIO(p['_source_png']))
        self.assertEqual(image.size,(768,768))
        self.assertEqual(image.getpixel((0,384)),(0,0,0))
        self.assertEqual(image.getpixel((384,384)),(255,0,0))
        self.assertEqual(image.getpixel((191,384)),(0,0,0))
        self.assertEqual(image.getpixel((192,384)),(255,0,0))
        g=runtime.graph(p,'frame_i2i_'+'a'*32+'.png')
        self.assertNotIn('5',g)
        self.assertEqual(g['10']['class_type'],'LoadImage')
        self.assertEqual(g['11']['inputs'],{'pixels':['10',0],'vae':['7',0]})
        self.assertEqual(g['6']['inputs']['latent_image'],['11',0])
        self.assertEqual(g['6']['inputs']['denoise'],.45)
        with self.assertRaises(ValueError):runtime.graph(p,'../escape.png')
        t2i=runtime.graph(runtime.validate({'prompt':'A mug.'}))
        self.assertEqual(t2i['6']['inputs']['latent_image'],['5',0])
        self.assertEqual(t2i['6']['inputs']['denoise'],1)

    def test_i2i_validation_before_generation(self):
        for changes in [{'image_base64':''},{'image_base64':'bad!'}, {'denoise':0},{'denoise':1.1},{'denoise':True},{'denoise':float('nan')},{'mode':'unknown'}]:
            with self.assertRaises(Exception):runtime.validate(self.source_input(**changes))
        with self.assertRaises(ValueError):runtime.validate({'prompt':'x','image_base64':'abcd'})
        with self.assertRaises(ValueError):runtime.validate({'prompt':'x','denoise':.5})

    def test_source_file_cleanup_after_success_and_failure(self):
        p=runtime.validate(self.source_input())
        for fail in (False,True):
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                def execute(params,filename):
                    file=root/'input'/filename
                    self.assertEqual(file.read_bytes(),p['_source_png'])
                    if fail:raise RuntimeError('simulated generation failure')
                    return {'mode':'krea2_i2i'}
                with patch.object(runtime,'COMFY_DIR',root),patch.object(runtime,'ready'),patch.object(runtime,'execute',side_effect=execute):
                    if fail:
                        with self.assertRaises(RuntimeError):runtime.generate(p)
                    else:self.assertEqual(runtime.generate(p)['mode'],'krea2_i2i')
                self.assertEqual(list((root/'input').iterdir()),[])

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
