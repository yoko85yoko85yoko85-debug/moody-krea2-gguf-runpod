# Moody Krea 2 Mix V4 GGUF — RunPod Serverless

モデルをDockerイメージに同梱せず、最初の生成時に取得する構成です。
GPU Pod・Jupyter・手動の巨大イメージ書き出しは使いません。
既存のFLUX/H3とは別のリポジトリ・Endpointで利用します。

## 現在の状態

- Dockerfile / handler / ComfyUI APIワークフロー / モデル取得・容量確認処理を作成済み。
- CPU上の設定・入力検証・API応答・出力後の片付け・容量不足時の停止テスト：6件成功。
- Dockerイメージの実ビルド、依存パッケージの実インストール、GGUFの実ロード、GPU推論は未確認。
- 初期版はテキストから画像生成のみ。画像編集・LoRA・専用アプリは未実装。
- steps=12 / CFG=1 / euler / simple は最初の動作確認用の仮設定。Moody V4での最適値は未検証。
- 4090・24GBでの動作、速度、画質や最大解像度は未検証。

## 固定した構成

| 役割 | ファイル | 容量（概数） |
| --- | --- | ---: |
| モデル | Henrysk/Moody-Krea-2-Mix-V4-GGUF / moodyKrea2Mix_v40-Q4_K.gguf | 7.22GB |
| テキストエンコーダー | Comfy-Org/Krea-2 / text_encoders/qwen3vl_4b_fp8_scaled.safetensors | 5.24GB |
| VAE | Comfy-Org/Krea-2 / vae/qwen_image_vae.safetensors | 0.254GB |

合計12,714,045,638 bytes（約12.7GB / 11.84GiB）。各ファイルのコミット・サイズ・SHA256を models.json に固定。
ComfyUIとKrea2対応GGUFローダーのコミットもDockerfileで固定。モデルの利用条件は配布元に従ってください。

- https://huggingface.co/Henrysk/Moody-Krea-2-Mix-V4-GGUF
- https://huggingface.co/Comfy-Org/Krea-2
- https://github.com/molbal/ComfyUI-GGUF

## 容量不足を避ける仕組み

1. ビルド中はモデルを一切ダウンロードしません。
2. リポジトリ全体ではなく、指定した3ファイルだけを取得します。
3. local_dirへ取得し、ComfyUI側へはコピーせずシンボリックリンクを張ります。
4. 未取得分＋3GiBの空きがない場合、取得開始前に停止します。
5. ファイルサイズとSHA256を検証し、同一ファイルは検証記録を再利用します。
6. 返却した画像はディスクから削除し、ComfyUI履歴も削除します。

ワーカーが破棄されると一時ディスクのキャッシュも消えます。新しいホストでは再取得が必要です。
必要に応じて永続ボリュームを用意しMODEL_ROOTとHF_HOMEをそこへ設定できますが、ストレージ料金は別途発生します。

## 今日は保存だけ行う場合

空のGitHubリポジトリに以下を保存すれば、Podを起動せずに翌日再開できます。
Dockerfile / handler.py / runtime.py / models.json / workflow_api.json / test_runtime.py / .dockerignore / .gitignore / README.md
モデル本体・トークン・APIキーはGitHubへ保存しません。

## RunPodへ接続する手順（まだ実行していません）

1. Serverless → Deploy from a GitHub repositoryを選択。
2. 新しい専用リポジトリを選択。Dockerfileの場所はルートのDockerfile。
3. 設定値を確認してからBuild/Deployを開始。

注意：公式GitHub連携はビルド後のテスト・デプロイも含みます。ビルド操作だけなら必ず無料とは断言できません。
課金を避けたい日はGitHubへの保存までで止めてください。ビルド/テスト/ストレージの料金条件は実行前にRunPod側で確認します。
公式資料：https://docs.runpod.io/serverless/workers/github-integration
Dockerビルド部分は30分、全体は160分という制限が記載されています。実ビルドで制限に当たる場合は依存構成を見直します。

GPUテスト時の候補設定：RTX 4090・1GPU、Max workers=1、Active workers=0、Idle timeout=5秒、Container disk=60GB、Execution timeout=1800秒。
これは試験用の候補で、未検証です。ワーカー起動・初期化・生成・待機、保存領域は課金対象になり得ます。
RunPodの請求仕様：https://docs.runpod.io/serverless/pricing

## 診断

```json
{"input":{"diagnose":true}}
```

診断自身はモデルダウンロード・ComfyUI起動・推論を行いません。ただしRunPod上のGPUワーカーを起動して診断を実行すれば、その稼働は課金対象になり得ます。
`ok=true` は設定ファイルの存在確認であり、GPUでの生成成功を意味しません。

## 最初の生成テスト（有料、まだ実行していません）

```json
{"input":{"prompt":"A ceramic mug on a wooden table, soft window light.","width":768,"height":768,"seed":42,"steps":12,"cfg":1.0}}
```

返却：image_base64（PNG）、width、height、seed、steps、cfg、worker_version。
初回は3モデルの取得・検証・ロードが入ります。ジョブを自動再送しません。
指定できる入力はprompt、negative_prompt、width、height、seed、steps、cfg、diagnoseだけです。
ユーザー指定のワークフローや任意ファイルパスは受け付けません。

## GPU不要のローカル確認

```sh
python runtime.py --check
python -m unittest -v test_runtime.py
python -m py_compile runtime.py handler.py
```

## GGUF補助パラメーターの互換性修正 v3

実GPUログで6144対3456のエラーを確認。配布GGUFのヘッダーでは
`model.diffusion_model.last.modulation.lin` が形状[6144,2]、Q4_K形式でした。
ComfyUIのSimpleModulationがこのパラメーターを直接加算するため、GGUF Linear用の展開経路を通りません。
ビルド時に固定GGUFローダーへ限定パッチを適用し、この小さなパラメーターだけfloat32に展開します。
大きなLinear重み、元のGGUF、生成寸法、推論設定は変更しません。
CPUテストでは対象限定・二重適用防止・上流ソース変更時の停止を検証。修正後のGPU生成は未確認です。
worker_version: moody-krea2-gguf-v3-modulation-fix
