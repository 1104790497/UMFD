# -*- coding: utf-8 -*-
"""
主入口脚本:对 AITA 帖子的 selftext 列(正文)做 LDA 主题预测,
复刻 ICWSM 2022 论文《Mapping Topics in 100,000 Real-life Moral Dilemmas》。

用法(须在 scripts/ 目录下运行,模型文件为相对路径):
    python run_topic_prediction.py <输入.xlsx> [输出.csv]

输入:包含 post_id 与 selftext 列的 Excel 文件。
输出:CSV(utf-8-sig),列为 post_id + 48 个主题后验概率 p(k|d)
     + top1_topic/top2_topic/top1_prob/top2_prob。
"""
import os
import sys

import numpy as np
import pandas as pd

from utils.topic_modeling_utils import LDAMerged


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    in_path = os.path.abspath(sys.argv[1])
    if len(sys.argv) > 2:
        out_path = os.path.abspath(sys.argv[2])
    else:
        out_dir = os.path.join(os.path.dirname(in_path), "output")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(out_dir, base + "_topics.csv")

    df = pd.read_excel(in_path)
    if "selftext" not in df.columns:
        sys.exit("错误:输入文件缺少 selftext 列")
    print(f"[1/3] 读取 {in_path}: {len(df)} 行")

    # 空值统一转为空字符串(与论文数据管线一致:正文缺失则无词可建模)
    texts = df["selftext"].fillna("").astype(str).tolist()
    n_empty = sum(1 for t in texts if not t.strip())
    if n_empty:
        print(f"      警告: {n_empty} 条 selftext 为空,将预测为均匀分布")

    print("[2/3] 加载 LDA 主题模型并预测(48 主题)...")
    model = LDAMerged()
    posteriors = model.predict(texts)  # DataFrame: (n, 48), 行和 = 1

    # 组装输出:post_id + 48 主题概率 + top-1/top-2
    out = df[["post_id"]].copy() if "post_id" in df.columns else pd.DataFrame(index=df.index)
    out = pd.concat([out.reset_index(drop=True), posteriors.reset_index(drop=True)], axis=1)
    topics = posteriors.columns
    order = np.argsort(-posteriors.values, axis=1)
    rows = np.arange(len(order))
    out["top1_topic"] = [topics[i] for i in order[:, 0]]
    out["top2_topic"] = [topics[i] for i in order[:, 1]]
    out["top1_prob"] = posteriors.values[rows, order[:, 0]]
    out["top2_prob"] = posteriors.values[rows, order[:, 1]]

    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[3/3] 输出 {len(out)} 行 -> {out_path}")
    print("      top-1 主题分布:")
    print(out["top1_topic"].value_counts().head(10).to_string())


if __name__ == "__main__":
    main()
