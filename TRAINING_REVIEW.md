# 第一次訓練分析與混合 duration 改版

使用既有 `/opt/miniconda3/envs/tinyml-tf/bin/python`，未安裝套件或變更 lock。
本次保留工作區目前的 `conv_type: SC`、`runtime.device: cpu` 選擇。
要使用 Metal 可加 `--set runtime.device=auto`。

## 已確認的準確率問題

第一次 run：`20261023T204248Z_2bee67ec_ab4bec`，原本是 1s／40 mel／DW／0.5 width。
讀取當時的完整 history 與 metrics 得到：

| 項目 | 實際結果 |
| --- | --- |
| epoch 12 training accuracy（含增強） | 66.97% |
| history 最高 validation accuracy | epoch 10：52.08% |
| history 最低 validation loss | epoch 4：1.47382 |
| 舊程式實際保存／匯出的 checkpoint | epoch 4 |
| 該 checkpoint 的 Keras／FP32 TFLite validation accuracy | 44.61% |
| 該 checkpoint 的 INT8 TFLite validation accuracy | 44.00% |

所以「training accuracy 只有 0.4」不是 history 的最終 training accuracy。
主要已確認的流程問題是 **checkpoint 監控 val_loss，使用者卻以 val_accuracy 比較**。
epoch 4 之後連續 8 個 epoch 沒刷新最低 loss，程式在 epoch 12 正常 early-stop，
恢復 epoch 4，導致較晚 epoch 較高的 accuracy 沒被保存。
這不是 early stopping 計數錯誤，也不是 TFLite 轉換造成原始 FP32 只有 44.61%。
舊程式只有保存最佳 checkpoint，不能從 history 還原 epoch 10 的權重。

在原始模型仍可讀取時，另外實測了相同 epoch-4 權重：

| 評估條件 | Accuracy | Macro-F1 |
| --- | --- | --- |
| training folds，關閉增強，單一 1s 中央裁切 | 53.85% | 0.5367 |
| validation fold 9，單一 1s 中央裁切 | 44.61% | 0.4154 |
| validation fold 9，三個均勻 1s 裁切，原檔平均機率 | 45.59% | 0.4255 |

三裁切只有約 0.98 個百分點改善；因此不能宣稱只靠 multi-crop 就能解決準確率。
training／validation 存在差距，但應比較同一 checkpoint 與一致裁切條件；
不能直接把 epoch-12 augmented training accuracy 和 epoch-4 validation 混在一起判斷。
較短裁切可能漏掉標籤事件是合理推測，並非已證明每個錯誤都由此造成。
UrbanSound8K 原始檔最長約 4 秒，參見[資料集說明](https://urbansounddataset.weebly.com/urbansound8k.html)。

## 目前實作的修正

- `training.monitor: val_accuracy` 成為新 baseline 的共同監控目標；亦支援
  `val_macro_f1`／`val_loss`。checkpoint、early stopping 與 LR schedule 使用同一 monitor。
- 每個 epoch 輸出 train／val accuracy、1s／2s 各自的 val accuracy、已用與剩餘 patience、
  best epoch 和 LR；同樣寫入 history.csv。
- 使用 Keras 既有 callbacks 集中控制訓練；停滯 3 次則 LR ×0.5，最低 1e-5；
  early-stop patience 預設 8。參見 [EarlyStopping](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping)
  與 [ReduceLROnPlateau](https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ReduceLROnPlateau)。
- 依使用者確認，`training.duration_secs: [1,2]` 依 batch 輪替長度，共同更新同一模型。
  不把不同形狀直接堆成同一 batch；每個原始檔每個 epoch 仍取一次，epoch 間洗牌。
- 驗證同時跑兩種長度，按原檔平均 crop 的 softmax 機率。監控指標在兩個長度上等權平均；
  `all` 模式再對三個寬度等權平均，避免只依某一長度／寬度選 checkpoint。
- 新 baseline 使用三個固定驗證裁切；不做隨機驗證增強，也不使用 test fold。
- 新設定 `model.standard_stem=true`：第一層使用標準 Conv2D。
  原始 DW stem 在單聲道輸入上只有一個 depthwise 空間濾波器，再經線性 pointwise 擴張，
  這在結構上限制初始特徵多樣性。新 stem 讓第一層直接學習多個空間濾波器。
  SC／DW 使用相同 stem，後續 block 才區分卷積類型；仍保持 BN-free 與 shared prefix。
  舊 config 沒有此旗標時維持原始架構，避免舊權重形狀失配。
- 同一組最佳權重預設輸出 1s／2s 的 FP32／INT8 TFLite，各自有固定輸入形狀。
  metadata 明確記錄 training durations，並用 frozen-weight SHA-256 驗證來源相同。

這些改動需要新的完整訓練才能測量最終效果；本次沒有宣稱新設定已超過 52.08%。
更大的 width／SC 模型是否改善泛化，仍應做受控比較，不能單靠換 CPU／GPU 推論準確率。

## 輸出與程式整理

新 run 使用 `config.yaml`、`history.csv`、一個 `run.json`、最佳權重／Keras 模型，
以及 `exports/1s/`／`exports/2s/` 的 TFLite。JSON 的 state、環境、資料來源、checkpoint、
架構、匯出、評估、artifact 清單都集中到 run.json，透過共用 artifact API 讀寫。
共享寬度的 parent 與三個 subnet 各有一個 run.json；subnet 共用 parent 的 config.yaml。
新 registry 依 artifact／duration 分別記錄 MACs、大小與該長度的驗證結果。

舊版資料夾仍可讀取，`python -m training.artifacts --model-dir models` 可以整併其 JSON。
工具會讀回驗證、確認來源沒有變動，再刪除列出的舊 JSON；不刪除權重或 CSV。
此次檢查期間，原始 1s run 資料夾已不在工作區，因此**沒有實際搬移該 run，也無法再
替原始權重追加 2s 匯出**。原始指標來自消失前的讀取；搬移／刪除機制用獨立測試資料
驗證。工作區目前的新增測試 run 已直接使用精簡格式。

## 已執行驗證

- `python -m unittest discover -s tests -v`：26 項通過。
- 真正的 1s／2s 訓練 batch 與樣本數檢查；驗證 shape／原檔聚合；共享梯度與 subnet 一致性。
- monitor accuracy 與 loss 相反走勢時，確認保存 accuracy 最佳權重；檢查 patience 與 LR 下降。
- 舊 DW stem 相容性；lossless JSON 整併與權重保留；拒絕整理 active training run。
- synthetic 混合長度訓練與兩種固定長度 INT8 invoke，確認權重 hash 一致。
- standalone conversion／evaluation CLI 通過；subnet 不需要重複的 config.yaml。
  另做 4 檔子集評估時寫入 `val_subset_4`，原 16 檔驗證結果保留。
- M1 GPU 上以真實資料 16 training／16 validation 檔案各跑一個 epoch：
  DW `width=all`，以及目前 SC 設定皆完成混合長度訓練、FP32／INT8 匯出。
  DW all run 產生 3 widths × 2 durations 的六組 static subnet 輸出。
  SC run 同樣確認第一個實際 training minibatch 位於 GPU:0。
  這些是診斷子集，不是準確率實驗；共同權重完整訓練與 held-out fold 10 評估尚未執行。

所有 Python 執行均使用既有 tinyml-tf。lock SHA-256 保持：
`91c6982b6763258442807c560fadd9a46a7fa89060d0e53f065260b4ba5b8317`。
