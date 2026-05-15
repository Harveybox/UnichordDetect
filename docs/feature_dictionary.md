# 特征字典

| 字段 | 类型 | 含义 |
|---|---|---|
| province | str | 省份编码（必须分省建模） |
| year | int | 年份 |
| batch | str | 批次 |
| school | str | 院校 |
| major | str | 专业 |
| candidate_rank | int | 考生位次，越小越好 |
| major_min_rank | int | 专业当年最低录取位次 |
| major_avg_rank | int | 专业当年平均录取位次 |
| plan_count | int | 招生计划数 |
| plan_yoy_change | float | 计划同比变化 |
| heat_index | float | 热度指标 |
| admit | 0/1 | 标签 |
