import runtime


def handler(job):
    try:
        p = runtime.validate(job.get('input', {}))
        with runtime.LOCK:
            return runtime.diagnose() if p.get('diagnose') else runtime.generate(p)
    except Exception as error:
        detail = runtime.diagnostic_error(error)
        print('Generation failed: ' + runtime.safe_text(detail), flush=True)
        return {'error': 'Krea2処理に失敗しました。detail・phase・engine_log_tailを確認してください。', **detail}


if __name__ == '__main__':
    import runpod
    print(runtime.VERSION + ': registered; model loading deferred until generation', flush=True)
    runpod.serverless.start({'handler': handler})
