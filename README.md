# shinra_tutorial
森羅プロジェクトの日本語構造化タスクのチュートリアルです。

## 学習
`python train.py --input_path /path/to/training_dataset --model_path /path/to/saving_model_path`

## 検証
train.pyの引数に `--valid` を加えることで、モデル出力の精度、再現率、F1スコアを算出できます。  
学習データを8：2の割合で分割を行い、8割部分で学習、2割部分で推論を行い、スコアを算出します。  
引数`--text`を追加することで、plain textを元にエラーファイルを書き出すことができます。  
`python train.py --input_path /path/to/training_dataset --model_path /path/to/saving_model_path  --valid --text /path/to/plain_text_dataset/`  
スコア算出プログラムは以下を参照  
https://github.com/k141303/shinra_jp_scorer  

## 推論
`python evaluate.py --input_path /path/to/test_datset --model_path /path/to/model_loaded_in_script --output_path /path/to/results_saving`


