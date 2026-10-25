# UrbanSound8K CNN／DS-CNN

TensorFlow 2.15 的訓練、共享寬度模型、FP32／INT8 TFLite 匯出與原始音檔層級評估。
目前預設將 **1s／2s 混合訓練同一組權重**，完成後同時匯出兩種固定輸入長度。

## 環境與資料

所有命令從專案根目錄執行，使用既有環境；不會自動下載資料或安裝套件：

```sh
PY=/opt/miniconda3/envs/tinyml-tf/bin/python
"$PY" -m data.loader
"$PY" -m unittest discover -s tests -v
```

資料位置由 `dataset.root` 指定，預設 `/Users/colin/sound_datasets/urbansound8k`：

```text
metadata/UrbanSound8K.csv
 audio/fold1/*.wav ... audio/fold10/*.wav
```

原生類別由 CSV 讀取。train folds 1–8（7079 檔）、val fold 9（816 檔）、
test fold 10（837 檔）互不重疊；訓練與 PTQ calibration 不得使用 folds 9／10。
訓練不會自動評估 test。

`requirements-lock.txt` 保持原樣。既有 Python 3.11／TensorFlow 2.15.1／Keras 2.15.0
可直接使用。其他主機須用相容的 Python 3.10／3.11；Linux 若無 Metal wheel，應明確
建立平台安裝清單，不能默默改原 lock：

```sh
# 僅供另一台 Linux 主機安裝時參考；本次沒有執行安裝。
awk '!/^tensorflow-metal==/' requirements-lock.txt > /tmp/requirements-linux.txt
python -m pip install -r /tmp/requirements-linux.txt
```

## 訓練與長度

```sh
# 使用 baseline 當下選定的 SC／DW、width、裝置設定。
"$PY" -m training.train --config configs/baseline.yaml

# 優先 Apple Metal GPU，無 GPU 則使用 CPU。
"$PY" -m training.train --set runtime.device=auto

# 明確指定 DS-CNN、共享寬度；其他參數仍讀 YAML。
"$PY" -m training.train --set model.conv_type=DW --set model.width=all

# 僅用 1s 訓練，仍可匯出同權重的 1s／2s。
"$PY" -m training.train --set 'training.duration_secs=[1]'

# 僅用 2s 訓練／匯出，64 mel，標準 CNN。
"$PY" -m training.train --set 'training.duration_secs=[2]' \
  --set 'quantization.output_duration_secs=[2]' --set audio.n_mels=64 --set model.conv_type=SC

# 一個 synthetic epoch，真正執行訓練與 FP32／INT8 轉換；非資料集準確率。
"$PY" -m training.train --dry-run --limit 4 --set training.batch_size=2
```

`training.duration_secs: [1, 2]` 會依序輪替 1s／2s batch，所有 batch 更新同一個模型；
每個 batch 內的形狀固定。epoch 起點輪替且音檔重新洗牌，各原始檔每個 epoch 使用一次。
這不代表每個原始檔在同一 epoch 都同時跑過兩種長度。`history.csv` 記錄各長度的樣本數。
空清單則沿用 `audio.duration_sec`，相容舊設定。

每個 epoch 分別驗證兩種長度，先對原始音檔聚合 crop 機率，再計算 accuracy／macro-F1。
預設 checkpoint、early stopping 與 learning-rate schedule 都監控 `val_accuracy`：
對訓練長度等權平均；`width=all` 再對三種寬度等權平均。
也可選 `training.monitor=val_loss` 或 `val_macro_f1`。

終端機與 CSV 會顯示 train accuracy、val accuracy、各 duration 的 val accuracy、
已用／總 patience、剩餘 patience、最佳 epoch 與學習率。未改善時累計 patience；
預設連續 8 次未改善停止，恢復最佳權重。連續 3 次未改善則將學習率乘 0.5，最低 1e-5。
train accuracy 含隨機增強，與固定裁切的 validation accuracy 不是完全相同的評估條件。

`--limit` 是診斷子集，結果會標記；`--dry-run` 的數值明確標記 synthetic。
`--skip-int8` 可明確只做 FP32。最佳權重 checkpoint 不保存 optimizer 狀態，
目前不支援接續訓練；新實驗須使用新 run。

## 前處理、增強與模型

訓練順序：soundfile 正規化浮點讀取 → 多聲道平均 → soxr HQ 重採樣 16 kHz →
50% 機率 librosa 保留音高變速（0.9–1.1）→ 隨機裁切／置中補零 →
80% 機率隨機增益（−6～+6 dB，不 clipping）→ Log-Mel。
變速倍率大於 1 會縮短波形。增益為 `10 ** (dB / 20)`。
增強在每次取樣即時計算，seed 與 epoch 決定隨機序列；不預存增強副本。
關閉 augmentation 仍保留訓練隨機裁切。

`features/waveforms` 只快取單聲道／重採樣波形，以來源路徑、size／mtime、前處理設定
與版本雜湊命名並原子寫入；不快取隨機增強。`dataset.cache_dir=null` 可關閉。

Log-Mel：400-sample periodic Hann、160-sample hop、512 FFT、無 centering，
power spectrum、Slaney mel／area normalization、`ln(max(mel, 1e-6))`，無逐檔正規化。
1s／2s 的輸入分別是 `[98,n_mels,1]`／`[198,n_mels,1]`；TFLite 接受這個特徵張量，非 WAV。

驗證／測試預設 `evaluation.num_crops: 3`，使用均勻分布的固定裁切，按原始音檔
平均 softmax 機率；短音檔只補零一次。設為 1 即中央裁切。這些樣本沒有速度／增益增強。
Calibration 永遠只用 training folds 的單一中央裁切，不增強。

名稱維持 `DSCNN_{40|64}_{SC|DW}_{1|5|25|all}`。`5` 表示 0.5 倍，`25` 表示 0.25 倍，
通道數 `max(1,floor(base*width))`；`5.0` 不是合法 token。
新設定 `model.standard_stem=true` 讓第一層使用 Conv2D，其餘 SC 為標準卷積，
DW 為 depthwise＋pointwise；每個 block stride 2、bias／ReLU、無 BatchNorm，
最後 Global Average Pooling＋Dense logits。舊設定未包含此旗標時保留原始 stem。

`all` 的 SC／DW kernel 與 classifier 都以同一組 Variable 的 channel prefix 切片。
同一 minibatch 在單一 GradientTape 內跑三個寬度，再用加權 loss 做一次更新。
不使用三個獨立模型，也不使用混合寬度的共用 BatchNorm。
靜態 subnet 匯出時會檢查與 supernet 的數值一致性。

## 精簡後的輸出

```text
models/1s_2s/DSCNN_40_DW_5/<run_id>/
  config.yaml         # 唯一設定快照
  history.csv         # 每個 epoch 的指標與 patience
  run.json            # 集中紀錄，取代原本十多個 JSON
  best.weights.h5     # 最佳 checkpoint 權重
  best.keras          # 架構＋最佳權重
  exports/1s/fp32.tflite, int8.tflite
  exports/2s/fp32.tflite, int8.tflite
```

單一長度仍放在 `models/1s/` 或 `models/2s/`。run ID 由 UTC 時間、設定雜湊與隨機碼組成，
可用 `--run-id` 自訂；碰撞會拒絕覆寫。`all` 在根目錄保存共同紀錄／shared checkpoint，
三個 subnet 各自保存 `run.json`／`best.keras`／兩種長度的 TFLite，設定讀取共同根目錄。

`run.json` 分成 `state`、`software_versions`、`dataset_manifest`、`first_training_batch`、
`checkpoint`、`metadata`、`architecture`、`exports`、`evaluation`、`artifacts` 區塊。
匯出資訊含 IO dtype／shape／scale／zero-point、operators、bytes、CPU invoke 結果、
校正檔案清單與 frozen-weight SHA-256；兩種長度相同 hash 表示同一組訓練權重。
評估資訊依 duration／split 保存，包含 confusion matrix、accuracy、macro-F1 與類別對照。
另做不同 crop policy 或子集評估時使用不同 key，保留原結果。

```sh
RUN=models/1s_2s/DSCNN_40_DW_5/實際run_id
"$PY" -m quantization.convert_int8 --model-dir "$RUN"
"$PY" -m evaluation.evaluate --model-dir "$RUN" --split val --duration 2
"$PY" -m evaluation.evaluate --model-dir "$RUN" --split val --num-crops 1
# 僅在準備使用 held-out 結果時執行：
"$PY" -m evaluation.evaluate --model-dir "$RUN" --split test
```

PTQ 強制 `TFLITE_BUILTINS_INT8` 與 INT8 IO，拒絕 float fallback；檢查 CPU interpreter
能執行且無 float tensor。錯誤保存在 `run.json` 並 raise。轉換器的 ARITH constant 摘要
或通用 calibration warning 不等於轉換失敗，以最終 tensors／invocation 驗證為準。

舊版多 JSON 輸出仍可讀取。若要整理既有已完成 run：

```sh
"$PY" -m training.artifacts --model-dir models
```

工具先整併、讀回驗證與檢查來源未變動，再刪除明確列出的舊 JSON；不刪除模型權重、
TFLite、YAML 或 CSV，且拒絕整理仍在訓練中的舊 run。

`models/model_registry.csv` 以 lock＋原子更新記錄各 artifact／duration 的設定、路徑、
參數、MACs、模型大小及驗證指標；未測量的值留空。參數／MACs／檔案大小不是裝置
SRAM、Tensor Arena 或延遲。`profiling/schema.json` 保留後續實測欄位；沒有實作韌體、
能耗／延遲預測與硬體選擇策略。

## GPU 與本次分析

`runtime.device` 支援 `auto`／`cpu`／`gpu`；保留目前 YAML 的明確選擇。
`auto` 偵測 TensorFlow GPU，Apple silicon 使用 Metal，無 GPU 則退回 CPU。
啟動會用實際模型做 throwaway optimizer update，確認 outputs／gradients／weights 的裝置，
之後重設 seed。`deterministic=true` 不會被默默關閉；Metal probe 成功不代表跨 OS／driver
的完整訓練可位元重現。CPU helper ops 仍可使用 CPU，小模型也不保證 GPU 更快。
更改裝置可見性或 determinism 時請重啟 Python。

```sh
"$PY" -m training.runtime --set runtime.device=auto --output /tmp/device_verification.json
```

第一次訓練的 checkpoint 選擇問題、實際診斷與最新測試結果見 `TRAINING_REVIEW.md`。
`IMPLEMENTATION_REPORT.md`／`GPU_RUNTIME_REPORT.md` 是前版本的歷史驗證記錄。
