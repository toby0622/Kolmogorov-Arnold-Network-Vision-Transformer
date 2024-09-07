## 使用方法

1. 下載好資料集（Dataset），目前代碼預設（Default）用的是花的分類：<https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz>
2. 在 `train.py` 中把 `--data-path` 設定為解壓縮後的 `flower_photos` 文件夾的絕對路徑。
3. 下載預訓練權重（Pre-Trained Weight），在 `vit_model.py` 中，每個模型都有提供預訓練權重的下載地址，根據選擇的模型下載對應的預訓練權重。
4. 在 `train.py` 腳本中將 `--weights` 參數設定為下載好的預訓練權重路徑。
5. 設定好資料集的路徑 `--data-path` 以及預訓練權重的路徑 `--weights`，即可使用 `train.py` 開始訓練（訓練過程中會自動生成 `class_indices.json` 文件）。
6. 在 `predict.py` 中導入與訓練代碼中相同的模型，並將 `model_weight_path` 設定為訓練好的模型權重路徑（預設保存在 `weights` 文件夾下）。
7. 在 `predict.py` 中將 `img_path` 設定為你需要預測的圖片的絕對路徑。
8. 設定好權重路徑 `model_weight_path` 和預測圖片路徑 `img_path`，即可使用 `predict.py` 進行預測。
9. 如果要使用其他資料集，就比照預設採用的「花的分類」之文件結構進行擺放（即一個類別對應一個文件夾），並且將訓練及預測代碼中的 `num_classes` 設定為你自己的數據的類別數。
