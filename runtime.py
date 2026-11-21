"""Lazy ComfyUI runtime. Importing this file never starts GPU work or downloads."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
VERSION = 'moody-krea2-gguf-v1'
COMFY_DIR = Path(os.environ.get('COMFY_DIR', '/opt/ComfyUI'))
MODEL_ROOT = Path(os.environ.get('MODEL_ROOT', '/models'))
API = 'http://127.0.0.1:8188'
LOCK = threading.Lock()
process = None
startup_error = None


def load_manifest():
    return json.loads((HERE / 'models.json').read_text())


def validate(p):
    if not isinstance(p, dict):
        raise ValueError('input must be an object')
    allowed = {'prompt', 'negative_prompt', 'width', 'height', 'seed', 'steps', 'cfg', 'diagnose'}
    if set(p) - allowed:
        raise ValueError('Unsupported input field')
    if p.get('diagnose') is True:
        return {'diagnose': True}
    prompt = p.get('prompt', '')
    negative = p.get('negative_prompt', '')
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 3000:
        raise ValueError('prompt must contain 1–3000 characters')
    if not isinstance(negative, str) or len(negative) > 3000:
        raise ValueError('negative_prompt must be a string of at most 3000 characters')
    out = {'prompt': prompt.strip(), 'negative_prompt': negative}
    for key, default, low, high in [('width', 768, 256, 1536), ('height', 768, 256, 1536),
                                    ('seed', 42, 0, 2147483647), ('steps', 12, 1, 50)]:
        n = p.get(key, default)
        if type(n) is not int or not low <= n <= high:
            raise ValueError(key + ' is outside the allowed integer range')
        out[key] = n
    if out['width'] % 32 or out['height'] % 32 or out['width'] * out['height'] > 1536 * 1024:
        raise ValueError('Dimensions must be multiples of 32 and at most 1.5 MiPixels')
    cfg = p.get('cfg', 1.0)
    if type(cfg) not in (int, float) or not 0.0 <= cfg <= 10.0:
        raise ValueError('cfg must be between 0 and 10')
    out['cfg'] = cfg
    return out


def graph(p):
    g = copy.deepcopy(json.loads((HERE / 'workflow_api.json').read_text()))
    g['3']['inputs']['text'] = p['prompt']
    g['4']['inputs']['text'] = p['negative_prompt']
    for key in ('width', 'height'):
        g['5']['inputs'][key] = p[key]
    for key in ('seed', 'steps', 'cfg'):
        g['6']['inputs'][key] = p[key]
    return g


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_models():
    from huggingface_hub import hf_hub_download
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    entries = load_manifest()['models']
    # local_dir avoids a second full copy in a separate download cache.
    needed = sum(m['bytes'] for m in entries if not (MODEL_ROOT / m['folder'] / m['remote_path']).exists())
    if shutil.disk_usage(MODEL_ROOT).free < needed + 3 * 1024**3:
        raise RuntimeError('Insufficient model disk space; no download was started')
    for m in entries:
        folder = MODEL_ROOT / m['folder']
        folder.mkdir(parents=True, exist_ok=True)
        path = Path(hf_hub_download(repo_id=m['repo'], filename=m['remote_path'],
                    revision=m['revision'], local_dir=str(folder)))
        if path.stat().st_size != m['bytes']:
            raise RuntimeError('Downloaded model has incorrect size: ' + m['filename'])
        marker = path.with_suffix(path.suffix + '.verified.json')
        fingerprint = {'sha256': m['sha256'], 'size': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}
        try:
            verified = json.loads(marker.read_text()) == fingerprint
        except (OSError, ValueError):
            verified = False
        if not verified:
            if digest(path) != m['sha256']:
                raise RuntimeError('Downloaded model checksum mismatch: ' + m['filename'])
            marker.write_text(json.dumps(fingerprint))
        # Support-model repositories use subdirectories; symlink only, never copy weights.
        target = COMFY_DIR / 'models' / m['folder'] / m['filename']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        if target.exists():
            raise RuntimeError('Model destination already exists: ' + str(target))
        target.symlink_to(path.resolve())


def call(path, payload=None, timeout=15):
    req = urllib.request.Request(API + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def ready():
    global process, startup_error
    if startup_error:
        raise RuntimeError(startup_error)
    if process is not None:
        if process.poll() is not None:
            raise RuntimeError('ComfyUI exited; restart the worker')
        return
    try:
        ensure_models()
        process = subprocess.Popen([sys.executable, str(COMFY_DIR / 'main.py'), '--listen', '127.0.0.1',
                    '--port', '8188', '--disable-auto-launch', '--disable-metadata'], cwd=COMFY_DIR)
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('ComfyUI failed to start; see container logs')
            try:
                info = call('/object_info')
                missing = {node['class_type'] for node in graph(validate({'prompt': 'test'})).values()} - set(info)
                if missing:
                    raise RuntimeError('Required nodes missing: ' + ', '.join(sorted(missing)))
                return
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        raise RuntimeError('ComfyUI startup timed out')
    except Exception as error:
        if process is not None and process.poll() is None:
            process.terminate()
        startup_error = 'Startup failed; restart after fixing configuration: ' + str(error)
        raise


def generate(p):
    ready()
    response = call('/prompt', {'prompt': graph(p)})
    if response.get('node_errors') or not response.get('prompt_id'):
        raise RuntimeError('Workflow rejected: ' + json.dumps(response.get('node_errors', {})))
    prompt_id = response['prompt_id']
    deadline = time.monotonic() + 900
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError('ComfyUI exited during generation')
            item = call('/history/' + prompt_id).get(prompt_id)
            if item:
                if item.get('status', {}).get('status_str') == 'error':
                    raise RuntimeError('ComfyUI execution failed; see container logs')
                images = item.get('outputs', {}).get('8', {}).get('images', [])
                if images:
                    image = images[0]
                    base = (COMFY_DIR / 'output').resolve()
                    file = (base / image.get('subfolder', '') / image['filename']).resolve()
                    if not file.is_relative_to(base) or file.stat().st_size > 12 * 1024**2:
                        raise RuntimeError('Invalid or oversized image output')
                    try:
                        encoded = base64.b64encode(file.read_bytes()).decode()
                    finally:
                        file.unlink(missing_ok=True)
                    call('/history', {'delete': [prompt_id]})
                    return {'image_base64': encoded, 'mime_type': 'image/png', 'mode': 'krea2_t2i',
                            'worker_version': VERSION, 'model': 'Moody Krea 2 Mix V4 Q4_K',
                            **{key: p[key] for key in ('width', 'height', 'seed', 'steps', 'cfg')}}
            time.sleep(1)
        raise TimeoutError('Generation timed out')
    except Exception:
        # Avoid leaving an unobserved GPU task running after returning an error.
        process.terminate()
        process.wait(timeout=15)
        raise


def diagnose():
    manifest = load_manifest()
    return {'ok': True, 'worker_version': VERSION, 'engine_running': process is not None and process.poll() is None,
            'downloads_started_by_diagnostic': False, 'planned_model_bytes': sum(m['bytes'] for m in manifest['models']),
            'model_files': [m['filename'] for m in manifest['models']], 'gpu_inference_verified': False}


if __name__ == '__main__':
    # Build-time smoke check: no torch, GPU, network or model download.
    assert '--check' in sys.argv
    assert len(load_manifest()['models']) == 3
    g = graph(validate({'prompt': 'A ceramic mug on a wooden table.'}))
    assert g['6']['class_type'] == 'KSampler'
    print('BUILD_SMOKE_OK: configuration and workflow; no weights downloaded')
