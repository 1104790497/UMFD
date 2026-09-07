import os
import html
import pandas as pd
import numpy as np
import re
import warnings

warnings.filterwarnings("ignore")

# ===================== 文件路径配置 =====================
USER_POST_FILE = "F:\\RoBERTa\\RoBERTa-base\\data\\user_post.xlsx"
USER_COMMENT_FILE = "F:\\RoBERTa\\RoBERTa-base\\data\\user_comment.xlsx"
COMMENTED_POST_FILE = "F:\\RoBERTa\\RoBERTa-base\\data\\post_commented_by_user.xlsx"

# ===================== 字段配置（更新MFT列名） =====================
FIELD_CONFIG = {
    "comment": {
        "comment_id": "comment_id",
        "user_id": "user_id",
        "comment_text": "combined_text",
        "comment_mft": [
            "care", "fairness", "loyalty", "authority", "purity"
        ],
        "post_id": "post_id",
        "stance": "label"
    },
    "commented_post": {
        "post_id": "post_id",
        "post_title": "title",
        "post_text": "selftext",
        "post_mft": [
            "care", "fairness", "loyalty", "authority", "purity"
        ]
    }
}

# 输出路径
OUTPUT_FILE = "F:\\RoBERTa\\RoBERTa-base\\data\\整合后的立场检测样本.xlsx"
TRAIN_FILE = "F:\\RoBERTa\\RoBERTa-base\\data\\训练集.xlsx"
VAL_FILE = "F:\\RoBERTa\\RoBERTa-base\\data\\验证集.xlsx"
TEST_FILE = "F:\\RoBERTa\\RoBERTa-base\\data\\测试集.xlsx"

# 全局统计计数器
_stats = {
    "urls_removed": 0,
    "comments_with_urls": 0,
    "posts_with_urls": 0,
    "posts_with_tldr": 0,
    "posts_with_edits": 0,
    "edit_paragraphs_removed": 0,
    "mft_invalid": 0,
    "mft_out_of_range": 0,
}


# ===================== 基础文本清洗 =====================

def _clean_html(text: str) -> str:
    return html.unescape(text)


_URL_RE = re.compile(r'https?://\S+')

def _remove_urls(text: str) -> str:
    result, n = _URL_RE.subn(' ', text)
    if n > 0:
        _stats["urls_removed"] += n
    return result


_REDDIT_BOLD_RE = re.compile(r'\*\*(.+?)\*\*')
_REDDIT_STRIKE_RE = re.compile(r'~~(.+?)~~')
_REDDIT_LINK_RE = re.compile(r'\[([^\]]+?)\]\([^\)]+\)')
_REDDIT_QUOTE_RE = re.compile(r'^>\s?', re.MULTILINE)

def _clean_reddit_formatting(text: str) -> str:
    text = _REDDIT_BOLD_RE.sub(r'\1', text)
    text = _REDDIT_STRIKE_RE.sub(r'\1', text)
    text = _REDDIT_LINK_RE.sub(r'\1', text)
    text = _REDDIT_QUOTE_RE.sub('', text)
    return text


_UNICODE_NORM_MAP = {
    '\u2018': "'",     # left single quote  '
    '\u2019': "'",     # right single quote '
    '\u201a': "'",     # single low-9 quote
    '\u201b': "'",     # single reversed-9 quote
    '\u201c': '"',     # left double quote  "
    '\u201d': '"',     # right double quote "
    '\u201e': '"',     # double low-9 quote
    '\u201f': '"',     # double reversed-9 quote
    '\u2013': '-',     # en dash  -
    '\u2014': '-',     # em dash  -
    '\u2012': '-',     # figure dash
    '\u2015': '-',     # horizontal bar
    '\u2026': '...',   # ellipsis
    '\u2022': '*',     # bullet
    '\u00a0': ' ',     # non-breaking space
}

_UNICODE_NORM_RE = re.compile('|'.join(re.escape(k) for k in _UNICODE_NORM_MAP))

def _normalize_unicode(text: str) -> str:
    return _UNICODE_NORM_RE.sub(lambda m: _UNICODE_NORM_MAP[m.group()], text)


_CONTROL_CHAR_RE = re.compile(r'[^\x20-\x7E\x0A\x0D]')

def _remove_control_chars(text: str) -> str:
    return _CONTROL_CHAR_RE.sub(' ', text)


def _normalize_whitespace(text: str) -> str:
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def clean_text(text: str) -> str:
    """基础文本清洗：HTML解码 → URL移除 → 格式清理 → 控制字符过滤 → 空白规范化"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = _clean_html(text)
    text = _remove_urls(text)
    text = _clean_reddit_formatting(text)
    text = _normalize_unicode(text)
    text = _remove_control_chars(text)
    text = _normalize_whitespace(text)
    return text


# ===================== 帖子结构清洗（Edit/Update去除 + TL;DR提取） =====================

_EDIT_MARKER_RE = re.compile(
    r'^(?:edit(?:ed)?|update|upd(?:ated)?|eta)'
    r'(?:\s*(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten))?'
    r'[\s:\-\[\(\#\.\/\*]',
    re.IGNORECASE
)

_TLDR_MARKER_RE = re.compile(
    r'(?:tl[;:\s\-]*dr|tldr)\s*[:;\-]*',
    re.IGNORECASE
)


def _extract_tldr(text: str):
    """返回 (移除TL;DR后的正文, TL;DR内容或None)
    只匹配出现在段落开头（\\n\\n 之后或文本开头）的 TL;DR 标记。
    """
    m = _TLDR_MARKER_RE.search(text)
    while m:
        if m.start() == 0 or (m.start() >= 2 and text[m.start() - 2:m.start()] == '\n\n'):
            break
        m = _TLDR_MARKER_RE.search(text, m.start() + 1)

    if not m:
        return text, None

    tldr_start = m.end()
    tldr_end = text.find('\n\n', tldr_start)
    if tldr_end == -1:
        tldr_end = len(text)

    tldr_content = text[tldr_start:tldr_end].strip()
    body = text[:m.start()] + text[tldr_end:]

    if tldr_content:
        _stats["posts_with_tldr"] += 1

    return body, tldr_content


def _remove_edits_updates(text: str) -> str:
    """位置感知的 Edit/Update 段落去除。

    策略：
    1. 按双换行切段落
    2. 找出所有以 Edit/Update/UPD/ETA 开头的段落
    3. 尾部编辑（最后30%段落）→ 从第一个尾部编辑处一刀截断
    4. 前中段编辑 → 仅删除该段落本身
    """
    paragraphs = [p.strip() for p in text.split('\n\n')]
    paragraphs = [p for p in paragraphs if p]

    if not paragraphs:
        return text

    total = len(paragraphs)
    edit_indices = [i for i, p in enumerate(paragraphs) if _EDIT_MARKER_RE.match(p)]

    if not edit_indices:
        return text

    _stats["posts_with_edits"] += 1
    _stats["edit_paragraphs_removed"] += len(edit_indices)

    tail_threshold = int(total * 0.7)
    tail_edits = [i for i in edit_indices if i >= tail_threshold]
    front_edits = [i for i in edit_indices if i < tail_threshold]

    if tail_edits:
        cut_at = min(tail_edits)
        _stats["edit_paragraphs_removed"] += (len(paragraphs) - cut_at) - len(tail_edits)
        paragraphs = paragraphs[:cut_at]

    for i in sorted(front_edits, reverse=True):
        if i < len(paragraphs):
            del paragraphs[i]

    return '\n\n'.join(paragraphs)


def _clean_post_body(text: str) -> str:
    """帖子的完整结构清洗管道"""
    text = _clean_html(text)
    text = _remove_urls(text)
    text, tldr = _extract_tldr(text)
    text = _remove_edits_updates(text)
    text = _clean_reddit_formatting(text)
    text = _normalize_unicode(text)
    text = _remove_control_chars(text)
    text = _normalize_whitespace(text)
    if tldr:
        tldr_clean = clean_text(tldr)
        if tldr_clean:
            text = text + " [TLDR] " + tldr_clean
    return text


def merge_title_text(title, text):
    """合并帖子标题和正文，清洗后返回"""
    clean_t = clean_text(title) if pd.notna(title) else ""
    clean_c = _clean_post_body(str(text)) if pd.notna(text) else ""
    if clean_t and clean_c:
        return f"{clean_t} | {clean_c}"
    return clean_t or clean_c


# ===================== MFT 与立场校验 =====================

def validate_mft_label(label):
    """校验MFT标签，确保为-1/0/1。无效值静默填0并计入统计。"""
    try:
        v = int(label)
        if v in (-1, 0, 1):
            return v
        _stats["mft_out_of_range"] += 1
        return 0
    except (ValueError, TypeError):
        _stats["mft_invalid"] += 1
        return 0


def validate_stance(stance):
    """标准化立场标签，非FAVOR/AGAINST均归为NEUTRAL"""
    if pd.isna(stance):
        return "NEUTRAL"
    s = str(stance).strip().upper()
    if s == "FAVOR":
        return "FAVOR"
    elif s == "AGAINST":
        return "AGAINST"
    else:
        return "NEUTRAL"


# ===================== 数据加载与清洗 =====================

def load_data():
    """加载原始评论和帖子数据"""
    data = {}
    data["comment"] = pd.read_excel(USER_COMMENT_FILE, engine="openpyxl")
    data["commented_post"] = pd.read_excel(COMMENTED_POST_FILE, engine="openpyxl")
    print(f"原始评论数据量：{len(data['comment'])} 条")
    print(f"原始帖子数据量：{len(data['commented_post'])} 条")
    return data


def clean_data(data):
    """清洗评论和帖子数据（不进行截断）"""
    _stats["comments_with_urls"] = 0
    _stats["posts_with_urls"] = 0
    _stats["posts_with_tldr"] = 0
    _stats["posts_with_edits"] = 0
    _stats["edit_paragraphs_removed"] = 0
    _stats["mft_invalid"] = 0
    _stats["mft_out_of_range"] = 0

    # --- 清洗评论数据 ---
    cmt = data["comment"].copy()
    n_before = len(cmt)
    print(f"\n--- 清洗评论 ---")
    print(f"清洗前：{n_before} 条")

    cmt["combined_text"] = cmt["combined_text"].apply(clean_text)

    for mft in FIELD_CONFIG["comment"]["comment_mft"]:
        cmt[mft] = cmt[mft].apply(validate_mft_label)

    cmt["label"] = cmt["label"].apply(validate_stance)

    cmt = cmt[cmt["combined_text"].str.len() > 0]
    cmt = cmt.dropna(subset=["post_id", "user_id", "comment_id"])
    print(f"清洗后：{len(cmt)} 条（移除空文本/缺失关键字段 {n_before - len(cmt)} 条）")

    # --- 清洗帖子数据 ---
    post = data["commented_post"].copy()
    n_before = len(post)
    print(f"\n--- 清洗帖子 ---")
    print(f"清洗前：{n_before} 条")

    post["post_text"] = post.apply(
        lambda r: merge_title_text(r["title"], r["selftext"]), axis=1
    )

    for mft in FIELD_CONFIG["commented_post"]["post_mft"]:
        post[mft] = post[mft].apply(validate_mft_label)

    post = post[post["post_text"].str.len() > 0]
    print(f"清洗后：{len(post)} 条（移除空文本 {n_before - len(post)} 条）")

    if _stats["posts_with_edits"] > 0:
        print(f"  含 Edit/Update 的帖子数：{_stats['posts_with_edits']}")
        print(f"  移除的 Edit/Update 段落数：{_stats['edit_paragraphs_removed']}")
    if _stats["posts_with_tldr"] > 0:
        print(f"  提取到 TL;DR 的帖子数：{_stats['posts_with_tldr']}")

    if _stats["mft_invalid"] > 0 or _stats["mft_out_of_range"] > 0:
        print(f"\n  MFT 标签校验问题：无效值 {_stats['mft_invalid']}，越界值 {_stats['mft_out_of_range']}")

    return {"comment": cmt, "commented_post": post}


def merge_data(cleaned):
    """合并评论和帖子数据（通过post_id关联）"""
    cmt = cleaned["comment"]
    post = cleaned["commented_post"]

    n_before = len(cmt)

    # 重命名评论MFT列
    mft_comment = FIELD_CONFIG["comment"]["comment_mft"]
    rename_dict_comment = {
        mft_comment[0]: "comment_mft1",
        mft_comment[1]: "comment_mft2",
        mft_comment[2]: "comment_mft3",
        mft_comment[3]: "comment_mft4",
        mft_comment[4]: "comment_mft5",
        "label": "stance"
    }
    cmt = cmt.rename(columns=rename_dict_comment)

    # 重命名帖子MFT列
    mft_post = FIELD_CONFIG["commented_post"]["post_mft"]
    rename_dict_post = {
        mft_post[0]: "post_mft1",
        mft_post[1]: "post_mft2",
        mft_post[2]: "post_mft3",
        mft_post[3]: "post_mft4",
        mft_post[4]: "post_mft5",
    }
    post = post.rename(columns=rename_dict_post)

    # 合并
    merged = pd.merge(
        cmt,
        post[["post_id", "post_text", "post_mft1", "post_mft2", "post_mft3", "post_mft4", "post_mft5"]],
        on="post_id",
        how="inner"
    )

    n_lost = n_before - len(merged)
    if n_lost > 0:
        print(f"\n  合并丢失：{n_lost} 条评论无匹配帖子（{(n_lost / n_before * 100):.1f}%）")

    # 保留核心列并去重
    final_cols = [
        "comment_id", "user_id", "post_id",
        "combined_text", "post_text",
        "comment_mft1", "comment_mft2", "comment_mft3", "comment_mft4", "comment_mft5",
        "post_mft1", "post_mft2", "post_mft3", "post_mft4", "post_mft5",
        "stance"
    ]
    merged = merged[final_cols].drop_duplicates("comment_id").reset_index(drop=True)
    return merged


# ===================== 按用户分层拆分 =====================

_MFT_POLARITY_COLS = [
    "comment_mft1", "comment_mft2", "comment_mft3", "comment_mft4", "comment_mft5",
    "post_mft1", "post_mft2", "post_mft3", "post_mft4", "post_mft5",
]


def split_by_user_stratified_binary(df, val_ratio=0.1, test_ratio=0.1):
    """
    按用户分层拆分。先处理稀有标签用户（轮询预分配），再对剩余用户立场分层。
    """
    np.random.seed(42)

    # ---- 1. 找出持有稀有 MFT 标签的用户，提前轮询分配 ----
    rare_users_train, rare_users_val, rare_users_test = set(), set(), set()
    rare_user_set = set()

    for col in _MFT_POLARITY_COLS:
        if col not in df.columns:
            continue
        for v in [-1, 1]:
            if (df[col] == v).sum() < 50:
                # 找持有该标签的用户，按样本量升序
                users_with = df[df[col] == v]['user_id'].unique()
                for uid in users_with:
                    if uid not in rare_user_set:
                        rare_user_set.add(uid)
                        n = len(rare_users_train) + len(rare_users_val) + len(rare_users_test)
                        if n % 3 == 0:
                            rare_users_train.add(uid)
                        elif n % 3 == 1:
                            rare_users_val.add(uid)
                        else:
                            rare_users_test.add(uid)

    # ---- 2. 剩余用户按立场分层 ----
    remaining_df = df[~df['user_id'].isin(rare_user_set)]

    user_stats = remaining_df.groupby('user_id').agg(
        samples=('stance', 'count'),
        stances=('stance', lambda x: set(x))
    ).reset_index()
    user_stats['has_favor'] = user_stats['stances'].apply(lambda s: 'FAVOR' in s)
    user_stats['has_against'] = user_stats['stances'].apply(lambda s: 'AGAINST' in s)

    layer1 = user_stats[user_stats['has_favor'] & ~user_stats['has_against']]
    layer2 = user_stats[user_stats['has_against']]

    def _split_layer(layer):
        if len(layer) == 0:
            return [], []
        layer = layer.sample(frac=1, random_state=42).reset_index(drop=True)
        total_samples = layer['samples'].sum()
        target_val = total_samples * val_ratio
        target_test = total_samples * test_ratio

        val_u, val_s = [], 0
        for _, row in layer.iterrows():
            if val_s + row['samples'] <= target_val * 1.1:
                val_u.append(row['user_id']); val_s += row['samples']
            else:
                break
        remaining = layer.iloc[len(val_u):]
        test_u, test_s = [], 0
        for _, row in remaining.iterrows():
            if test_s + row['samples'] <= target_test * 1.1:
                test_u.append(row['user_id']); test_s += row['samples']
            else:
                break
        train_u = layer.iloc[len(val_u) + len(test_u):]['user_id'].tolist()
        return train_u, val_u, test_u

    train_r, val_r, test_r = _split_layer(layer1)
    train_a, val_a, test_a = _split_layer(layer2)

    # ---- 3. 合并：稀有用户 + 剩余用户 ----
    train_u = list(rare_users_train) + train_r + train_a
    val_u = list(rare_users_val) + val_r + val_a
    test_u = list(rare_users_test) + test_r + test_a

    train_u = list(set(train_u))
    val_u = list(set(val_u) - set(train_u))
    test_u = list(set(test_u) - set(train_u) - set(val_u))

    if len(val_u) == 0 and len(train_u) > 0:
        s = df[df['user_id'].isin(train_u)].groupby('user_id').size()
        train_u.remove(s.idxmin()); val_u = [s.idxmin()]
    if len(test_u) == 0 and len(train_u) > 0:
        s = df[df['user_id'].isin(train_u)].groupby('user_id').size()
        train_u.remove(s.idxmin()); test_u = [s.idxmin()]

    train = df[df['user_id'].isin(train_u)].reset_index(drop=True)
    val = df[df['user_id'].isin(val_u)].reset_index(drop=True)
    test = df[df['user_id'].isin(test_u)].reset_index(drop=True)

    total = len(df)
    print("\n二分类划分后各数据集标签分布：")
    for name, data in [("训练集", train), ("验证集", val), ("测试集", test)]:
        if len(data) == 0:
            print(f"  {name}: 无样本")
            continue
        dist = data["stance"].value_counts(normalize=True) * 100
        print(f"  {name}（{len(data)}条, {len(data)/total*100:.1f}%）: "
              f"FAVOR={dist.get('FAVOR', 0):.1f}%, AGAINST={dist.get('AGAINST', 0):.1f}%")

    # 稀有标签分布验证
    mft_cols = [c for c in _MFT_POLARITY_COLS if c in df.columns]
    print("\n稀有 MFT 标签分布检查：")
    for col in mft_cols:
        for v in [-1, 1]:
            g = (df[col] == v).sum()
            if g < 50:
                tc = (train[col] == v).sum()
                vc = (val[col] == v).sum()
                sc = (test[col] == v).sum()
                print(f"  {col}[{v:+d}] 全局={g} → 训练={tc} 验证={vc} 测试={sc}")

    return train, val, test


# ===================== 主程序入口 =====================

if __name__ == "__main__":
    if os.path.exists(OUTPUT_FILE):
        print(f"发现已存在的整合文件：{OUTPUT_FILE}，将直接读取...")
        merged_data = pd.read_excel(OUTPUT_FILE, engine="openpyxl")
    else:
        raw_data = load_data()
        cleaned_data = clean_data(raw_data)
        merged_data = merge_data(cleaned_data)
        merged_data.to_excel(OUTPUT_FILE, index=False)
        print(f"\n数据整合完成，共生成 {len(merged_data)} 条样本（含中立）")
        print(f"整合文件保存路径：{OUTPUT_FILE}")

    # 过滤中立样本
    original_len = len(merged_data)
    merged_data = merged_data[merged_data["stance"] != "NEUTRAL"].reset_index(drop=True)
    print(f"过滤中立样本后，剩余样本数：{len(merged_data)}（移除 {original_len - len(merged_data)} 条）")

    # 按用户分层拆分
    train_set, val_set, test_set = split_by_user_stratified_binary(merged_data)

    # 保存拆分后的数据集
    train_set.to_excel(TRAIN_FILE, index=False)
    val_set.to_excel(VAL_FILE, index=False)
    test_set.to_excel(TEST_FILE, index=False)

    # 最终统计
    print(f"\n最终数据集拆分结果（仅FAVOR/AGAINST）：")
    print(f"  训练集：{len(train_set)} 条（{len(train_set) / len(merged_data) * 100:.1f}%）")
    print(f"  验证集：{len(val_set)} 条（{len(val_set) / len(merged_data) * 100:.1f}%）")
    print(f"  测试集：{len(test_set)} 条（{len(test_set) / len(merged_data) * 100:.1f}%）")

    print(f"\n全局立场分布（仅FAVOR/AGAINST）：")
    global_dist = merged_data["stance"].value_counts(normalize=True) * 100
    print(f"  FAVOR: {global_dist.get('FAVOR', 0):.1f}%")
    print(f"  AGAINST: {global_dist.get('AGAINST', 0):.1f}%")

    print(f"\n拆分后文件路径：")
    print(f"  训练集：{TRAIN_FILE}")
    print(f"  验证集：{VAL_FILE}")
    print(f"  测试集：{TEST_FILE}")
