#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import random
from typing import List, Tuple

from ase.io import read
import lmdb
import pickle as pkl
import numpy as np
from tqdm import tqdm


# ============================== 基础工具 ==============================

def _safe_float(x):
    """尽量把各种数值类型（含 numpy 标量）转为 float。失败返回 None。"""
    try:
        x = x.item() if hasattr(x, "item") else x
        x = float(x)
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def get_kappa_log(traj):
    """从 ASE Atoms 的 info['kappa_log'] 读取为 float;若缺失或非有限则返回 None。"""
    if not hasattr(traj, "info") or not isinstance(traj.info, dict):
        return None
    # kappa_log
    v = traj.info.get("kappa_log", None)
    return _safe_float(v)


# ============================== 分层 + 全局严格 8:1:1 ==============================

def stratified_split_by_kappa_log_exact(
    trajs: List,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_bins: int = 20,
    seed: int = 2000,
    drop_missing: bool = True,
) -> Tuple[List, List, List]:
    """
    按 kappa_log 分布做分层，并用 Hamilton 配额法保证**全局严格** 8:1:1。
    - ratios: (train, valid, test)，需相加为 1
    - num_bins: 按等频（分位数）切的箱数（建议 10~50）
    - drop_missing: True 则丢弃没有 kappa_log 或非有限值的样本；False 会把它们作为一个额外箱参与
    返回三个列表：train_trajs, valid_trajs, test_trajs
    """
    assert abs(sum(ratios) - 1.0) < 1e-8, "ratios 必须相加为 1.0"
    r_train, r_valid, r_test = ratios

    # 1) 组装 (idx, kappa) 列表
    idx_kappa, missing_idx = [], []
    for i, t in enumerate(trajs):
        k = get_kappa_log(t)
        if k is None:
            missing_idx.append(i)
        else:
            idx_kappa.append((i, k))

    if drop_missing:
        base_indices = [i for i, _ in idx_kappa]
    else:
        base_indices = [i for i, _ in idx_kappa] + list(missing_idx)

    n_total = len(base_indices)
    if n_total == 0:
        # 全缺失或空，退化为随机划分
        rng = random.Random(seed)
        indices = list(range(len(trajs)))
        rng.shuffle(indices)
        N_train = int(round(len(indices) * r_train))
        N_valid = int(round(len(indices) * r_valid))
        N_test = len(indices) - N_train - N_valid
        return (
            [trajs[i] for i in indices[:N_train]],
            [trajs[i] for i in indices[N_train:N_train + N_valid]],
            [trajs[i] for i in indices[N_train + N_valid:]],
        )

    # 2) 等频分箱（按 kappa 排序后均匀切块）
    idx_kappa_sorted = sorted(idx_kappa, key=lambda x: x[1])  # 稳定排序
    bins = []
    start = 0
    for b in range(num_bins):
        remaining = len(idx_kappa_sorted) - start
        size = int(round(remaining / (num_bins - b))) if (num_bins - b) > 0 else 0
        size = max(size, 0)
        end = start + size
        bins.append([i for (i, _) in idx_kappa_sorted[start:end]])
        start = end
    # 四舍五入误差导致的尾部残留合入最后一箱
    if start < len(idx_kappa_sorted):
        bins[-1].extend([i for (i, _) in idx_kappa_sorted[start:]])

    # 不丢弃缺失时，把缺失样本作为一个额外箱参与
    if not drop_missing and len(missing_idx) > 0:
        bins.append(list(missing_idx))

    # 3) 全局目标配额
    N_train = int(round(n_total * r_train))
    N_valid = int(round(n_total * r_valid))
    N_test = n_total - N_train - N_valid  # 保证和为 n_total

    rng = random.Random(seed)

    # 4) 每箱计算 floor 基数 + 小数部分（Hamilton 配额法的基础）
    # per_bin_alloc: [t_b, v_b, s_b, [frac_t, frac_v, frac_s], m, indices]
    per_bin_alloc = []
    base_train = base_valid = base_test = 0
    for b_indices in bins:
        m = len(b_indices)
        if m == 0:
            per_bin_alloc.append([0, 0, 0, [0.0, 0.0, 0.0], 0, b_indices])
            continue
        ideal_t = m * r_train
        ideal_v = m * r_valid
        ideal_s = m * r_test

        ft = math.floor(ideal_t)
        fv = math.floor(ideal_v)
        fs = math.floor(ideal_s)

        frac_t = ideal_t - ft
        frac_v = ideal_v - fv
        frac_s = ideal_s - fs

        per_bin_alloc.append([ft, fv, fs, [frac_t, frac_v, frac_s], m, b_indices])
        base_train += ft
        base_valid += fv
        base_test += fs

    # 5) 计算全局缺口
    gap_train = N_train - base_train
    gap_valid = N_valid - base_valid
    gap_test = N_test - base_test

    # 6) Hamilton 配额修正：按小数部分从大到小补足缺口
    bin_order = list(range(len(per_bin_alloc)))
    rng.shuffle(bin_order)  # 打乱箱顺序，避免固定偏置

    changed = True
    while changed:
        changed = False
        for b in bin_order:
            t_b, v_b, s_b, fracs, m, _ = per_bin_alloc[b]
            remainder = m - (t_b + v_b + s_b)
            if remainder <= 0:
                continue

            # 本箱内按小数部分（大->小）给名额，先随机打乱再排序打破完全相等时的稳定偏置
            order = [(0, fracs[0]), (1, fracs[1]), (2, fracs[2])]
            rng.shuffle(order)
            order.sort(key=lambda x: x[1], reverse=True)

            for _ in range(remainder):
                assigned = False
                for split_id, _frac in order:
                    if split_id == 0 and gap_train > 0:
                        per_bin_alloc[b][0] += 1
                        gap_train -= 1
                        assigned = True
                        break
                    if split_id == 1 and gap_valid > 0:
                        per_bin_alloc[b][1] += 1
                        gap_valid -= 1
                        assigned = True
                        break
                    if split_id == 2 and gap_test > 0:
                        per_bin_alloc[b][2] += 1
                        gap_test -= 1
                        assigned = True
                        break
                if assigned:
                    changed = True
                else:
                    break  # 没有全局缺口可补了

        if gap_train <= 0 and gap_valid <= 0 and gap_test <= 0:
            break
        if not changed:
            break  # 没法继续分配（极少见）

    # 7) 按 per_bin_alloc 在箱内采样并拼接
    train_idx, valid_idx, test_idx = [], [], []
    for (t_b, v_b, s_b, _fracs, _m, b_indices) in per_bin_alloc:
        if not b_indices:
            continue
        pool = list(b_indices)
        rng.shuffle(pool)
        t_take = min(t_b, len(pool))
        train_idx.extend(pool[:t_take])
        pool = pool[t_take:]

        v_take = min(v_b, len(pool))
        valid_idx.extend(pool[:v_take])
        pool = pool[v_take:]

        s_take = min(s_b, len(pool))
        test_idx.extend(pool[:s_take])
        pool = pool[s_take:]

        # 若还有剩余（通常不发生），把剩余塞入当前最缺的 split
        if pool:
            need = [(0, N_train - len(train_idx)),
                    (1, N_valid - len(valid_idx)),
                    (2, N_test - len(test_idx))]
            need.sort(key=lambda x: x[1], reverse=True)
            top = need[0]
            if top[1] > 0:
                if top[0] == 0:
                    train_idx.extend(pool)
                elif top[0] == 1:
                    valid_idx.extend(pool)
                else:
                    test_idx.extend(pool)

    # 8) 兜底裁剪，确保**全局精确**命中目标数量
    rng.shuffle(train_idx)
    rng.shuffle(valid_idx)
    rng.shuffle(test_idx)
    train_idx = train_idx[:N_train]
    valid_idx = valid_idx[:N_valid]
    test_idx = test_idx[:N_test]

    # 去重（理论上不会重复，这里稳妥起见）
    train_idx = list(dict.fromkeys(train_idx))
    valid_idx = list(dict.fromkeys(valid_idx))
    test_idx = list(dict.fromkeys(test_idx))

    # 最后若因为极端边界导致略少（极少见），从剩余未用样本中补齐到精确目标
    used = set(train_idx) | set(valid_idx) | set(test_idx)
    remaining = [i for i in base_indices if i not in used]
    rng.shuffle(remaining)

    def _fill_to(lst, target):
        nonlocal remaining
        if len(lst) < target:
            need = target - len(lst)
            lst.extend(remaining[:need])
            remaining = remaining[need:]

    _fill_to(train_idx, N_train)
    _fill_to(valid_idx, N_valid)
    _fill_to(test_idx, N_test)

    # 转对象
    train_trajs = [trajs[i] for i in train_idx]
    valid_trajs = [trajs[i] for i in valid_idx]
    test_trajs = [trajs[i] for i in test_idx]

    print(f"[Stratified-Exact] total={n_total} (drop_missing={drop_missing}) "
          f"-> train={len(train_trajs)}, valid={len(valid_trajs)}, test={len(test_trajs)}")
    return train_trajs, valid_trajs, test_trajs


# ============================== 写 LMDB ==============================

def convert_to_lmdb(
    data_name: str,
    split: str,
    trajs: List,
    save_path: str = "/home/work/data/test",
    max_atoms_num: int = 10000,
):
    """
    把一份 traj 列表写入某个 split 的 LMDB。
    目录：{save_path}/{data_name}/{split}
    Key: {data_name}-{i}  (i 为本 split 内的序号)
    """
    data_save_path = os.path.join(save_path, data_name, split)
    os.makedirs(data_save_path, exist_ok=True)

    env = lmdb.open(data_save_path, map_size=1024 ** 4)
    txn = env.begin(write=True)

    for i, traj in tqdm(
        enumerate(trajs),
        total=len(trajs),
        desc=f"writing {data_name}:{split}",
        miniters=1,
    ):
        if len(traj) > max_atoms_num:
            continue
        graph = traj.todict()
        graph["data_name"] = data_name
        if "info" not in graph:
            graph["info"] = {}
        graph["positions"] = traj.get_positions()

        key = f"{data_name}-{i}".encode()
        txn.put(key, pkl.dumps(graph))

    txn.commit()
    env.close()


# ============================== 主程序 ==============================

if __name__ == "__main__":
    # —— 路径与参数（按需修改）——
    data_path_dir = "/share/home/u15502/mfwu/kappaformer/data/kappaformer/kappadata/"
    save_path_dir = "/share/home/u15502/mfwu/kappaformer/data/kappaformer/kappadata_lmdb/"
    num_bins = 20
    # seed = 2000
    # seed = 110
    seed = 112
    ratios = (0.8, 0.1, 0.1)
    drop_missing = True  # 若想保留无 kappa_log 的样本，改为 False

    data_path_list = os.listdir(data_path_dir)
    print("files:", data_path_list)

    for fname in data_path_list:
        data_path = os.path.join(data_path_dir, fname)
        data_name = os.path.splitext(os.path.basename(data_path))[0]
        print(f"\nprocessing: {data_path}")

        ext = os.path.splitext(fname)[1].lower().lstrip(".")
        if ext in ("xyz", "extxyz"):
            trajs = read(data_path, index=":", format="extxyz")
        else:
            with open(data_path, "rb") as f:
                trajs = pkl.load(f)

        # —— 关键：分层 + 全局严格 8:1:1 —— #
        train_trajs, valid_trajs, test_trajs = stratified_split_by_kappa_log_exact(
            trajs,
            ratios=ratios,
            num_bins=num_bins,
            seed=seed,
            drop_missing=drop_missing,
        )

        # —— 写 LMDB —— #
        convert_to_lmdb(data_name, "train", train_trajs, save_path=save_path_dir, max_atoms_num=1000)
        convert_to_lmdb(data_name, "valid", valid_trajs, save_path=save_path_dir, max_atoms_num=1000)
        convert_to_lmdb(data_name, "test",  test_trajs,  save_path=save_path_dir, max_atoms_num=1000)

        # 统计一下分布概况，便于 sanity check（可按需注释）
        def _summ(klist):
            if not klist:
                return "[]"
            arr = np.array(klist, dtype=float)
            return (f"n={arr.size}, min={arr.min():.3f}, "
                    f"p50={np.median(arr):.3f}, max={arr.max():.3f}")

        k_train = [get_kappa_log(t) for t in train_trajs if get_kappa_log(t) is not None]
        k_valid = [get_kappa_log(t) for t in valid_trajs if get_kappa_log(t) is not None]
        k_test  = [get_kappa_log(t) for t in test_trajs  if get_kappa_log(t) is not None]
        print("[kappa_log] train:", _summ(k_train))
        print("[kappa_log] valid:", _summ(k_valid))
        print("[kappa_log] test :", _summ(k_test))
