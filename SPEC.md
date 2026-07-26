# 競艇予測モデル 構築仕様

## 1. ゴール
ボートレース（競艇）の公開データから、レース結果の確率分布を推定するモデルを作る。
**最終的な成功指標は的中率ではなく回収率。**
- 舟券の払戻率は 75%（控除率 25%）
- 「1号艇が1着」と常に答えるだけで的中率 55% が出るが、回収率は 50% 台にしかならない
- 解くべき問題は「較正された確率を推定し、オッズが示す市場確率より高い目だけを買う」

最終成果物: 毎朝実行すると当日の全レースについて期待値がプラスの買い目だけをレポート出力するパイプライン。

## 2. 絶対に守る原則
### 2.1 特徴量リークの禁止
過去成績系の特徴量は必ずそのレースの発走時刻より前のデータのみから計算する。
as-of join / カットオフ付き累積集計で実装。リーク検証テストを必ず書く。
### 2.2 分割は必ず時系列
- train: 2015–2023 / valid: 2024 / test: 2025–2026（最後まで見ない）
- walk-forward 検証（毎年再学習して翌年評価）も実装。
### 2.3 レース内で正規化（conditional logit / Plackett-Luce）
P(i が1着) = exp(s_i) / Σ_j exp(s_j)。6艇スコアから3連単120通りを導出。120クラス分類にしない。
### 2.4 評価は精度ではない
キャリブレーション（信頼性ダイアグラム・Brier）、回収率。ROC-AUCは参考。accuracyは報告しない。
### 2.5 バックテストは実データの払戻金（Kファイル）で行う
期待値が閾値超えの目だけ買う。閾値と購入率を必ず併記。全レース購入シミュレーションはしない。

## 3. データソース
### 3.1 日次ファイル（LZH, CP932）
- 番組表: https://www1.mbrace.or.jp/od2/B/{YYYYMM}/b{YYMMDD}.lzh
- 競走成績: https://www1.mbrace.or.jp/od2/K/{YYYYMM}/k{YYMMDD}.lzh
- 範囲 2015-01-01〜当日。非開催日は404（正常）。リクエスト間 1秒sleep。冪等取得。
### 3.2 固定長レイアウト仕様（必読）
https://www.boatrace.jp/owpc/pc/extra/data/layout.html
パーサ実装前に必読。選手名の全角パディング、年代によるレイアウト変更に注意。年別成功率を出す。
### 3.3 レーサー期別成績（半年ごと）
https://www.boatrace.jp/static_extra/pc_static/download/data/kibetsu/fan{YYMM}.lzh
N年前期=fan{N-1}10, N年後期=fan{N}04。2015〜2026 の 24ファイル。
### 3.4 解凍
システム lhasa を subprocess で叩くのが安定（macOS: brew install lhasa / apt: lhasa）。
本環境はネットワーク/apt制約のため pure-Python の lhafile を優先採用。

## 4. 技術スタック
Python 3.12 (uv), polars, LightGBM(lambdarank または binary+レース内softmax),
PostgreSQL + psycopg3, pytest。**PyTorch/NNは使わない**（テーブルデータはGBDT優位）。

## 5. ディレクトリ構成
kyotei-ml/ に SPEC.md, pyproject.toml, sql/schema.sql, src/kyotei/{download,parse_b,parse_k,load,features,model,backtest,predict}.py, tests/, reports/。
Notebookをメイン成果物にしない。すべて再実行可能なスクリプト。

## 6. フェーズと完了条件
### Phase 1 — データ取得と格納
取得成功率、**年別パース成功率**、DBのレース数・エントリ数（1レース=6行検証）、
course(進入)とlane(枠番)が別カラム、年間レース数が常識的水準か。
### Phase 2 — 特徴量
A: 進入コース、場ID×コース交互作用、級別。B: 平均ST・展示ST・展示タイム・当地勝率・
モーター2連率(shrinkage)。C: 気象×コース交互作用、決まり手傾向、節間成績、チルト、体重。
完了: リーク無しテスト通過、特徴量行数=エントリ数。
### Phase 3 — ベースライン（1号艇1着の二値分類）
時系列split ROC-AUC/Brier、信頼性ダイアグラム、EV分布。
### Phase 4 — 6艇 conditional softmax、3連単120通り。合計1.0テスト、キャリブレーション曲線。
### Phase 5 — バックテスト
EV閾値ごとの購入率/的中率/回収率/最大DD表。均等額と1/4ケリー。年別推移。testは最後の1回のみ。
### Phase 6 — 日次推論CLI。直前情報タイミング対応。reports/YYYY-MM-DD.md 出力。

## 7. やってはいけないこと
ランダムsplit / accuracy報告 / 未実行の数値記載 / try-except:pass / 未検証の行数変化結合 /
全レース購入バックテスト / testの反復チューニング / 巨大単一スクリプトやNotebookのみ。

## 8. 報告フォーマット
各Phase完了時: 実行コマンドと生出力 / 完了条件の数値 / 怪しい点を必ず1つ / 次Phase計画。
**回収率が100%超なら、まずリークを疑う**（公開競艇AIは概ね80〜95%）。
