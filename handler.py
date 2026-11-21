import runtime


def handler(job):
    try:
        p = runtime.validate(job.get('input', {}))
        with runtime.LOCK:
            return runtime.diagnose() if p.get('diagnose') else runtime.generate(p)
    except Exception as error:
        # Never return credentials, local paths, or raw HTTP exceptions to API clients.
        print('Generation failed: ' + type(error).__name__, flush=True)
        return {'error': 'Krea2処理に失敗しました。入力・モデル取得権限・ディスク容量とWorkerログを確認してください。',
                'error_type': type(error).__name__, 'worker_version': runtime.VERSION}


if __name__ == '__main__':
    import runpod
    print(runtime.VERSION + ': registered; model loading deferred until generation', flush=True)
    runpod.serverless.start({'handler': handler})
