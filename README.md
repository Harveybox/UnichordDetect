# UnichordDetect 录取概率项目代码

## 快速开始

```bash
pip install pandas scikit-learn joblib numpy
python src/train_logit.py --input data/feature/train.csv --province Zhejiang
python src/train_lgbm.py --input data/feature/train.csv --province Zhejiang
```

## 项目结构

- `src/features.py`：特征构建
- `src/train_logit.py`：Logistic 回归训练脚本
- `src/train_lgbm.py`：LightGBM 类树模型训练脚本（使用 sklearn HGB 替代）
- `src/calibrate.py`：概率校准（Isotonic）
- `src/bayes_adjust.py`：贝叶斯修正
- `src/mc_simulation.py`：蒙特卡洛区间估计
- `src/backtest.py`：分省回测报告
- `src/explain_top_features.py`：Top 特征解释
- `docs/feature_dictionary.md`：特征字典
- `docs/probability_rules.md`：冲稳保垫规则

## 输入数据要求

训练输入 CSV 至少包含：
- province, year, batch, school, major
- candidate_rank, major_min_rank, major_avg_rank
- plan_count, plan_yoy_change, heat_index
- admit

> 注意：必须分省建模，不可直接跨省混用模型。
