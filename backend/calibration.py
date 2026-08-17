"""
人工校准存储
============
运营人员 (或食堂经理) 可在特定时段手动调整某食堂的「当前人流系数」(+1 或 -1)，
作为紧急纠偏。校准偏移量会叠加到 base_hot_level 上，再乘以时间系数。

存储方式: JSON 文件 (calibrations.json)，简单可靠，无需数据库。
"""

import json
import os
from typing import Optional

# 校准文件路径 (与本文件同目录)
_CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations.json")

# 校准偏移量范围
MIN_DELTA = -1
MAX_DELTA = 1


def _load() -> dict[str, int]:
    """从文件加载校准数据。"""
    if not os.path.exists(_CALIBRATION_FILE):
        return {}
    try:
        with open(_CALIBRATION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save(data: dict[str, int]) -> None:
    """保存校准数据到文件。"""
    with open(_CALIBRATION_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_calibration_delta(canteen_name: str) -> int:
    """
    获取某食堂的校准偏移量。
    返回 0 表示无校准。
    """
    data = _load()
    return data.get(canteen_name, 0)


def set_calibration(canteen_name: str, delta: int) -> dict:
    """
    设置某食堂的校准偏移量。

    参数:
        canteen_name: 食堂中文名
        delta: +1 (人流偏多) 或 -1 (人流偏少)

    返回:
        操作结果 dict
    """
    delta = max(MIN_DELTA, min(MAX_DELTA, delta))

    data = _load()
    if delta == 0:
        # delta=0 等于清除校准
        data.pop(canteen_name, None)
    else:
        data[canteen_name] = delta
    _save(data)

    return {
        "canteen_name": canteen_name,
        "delta": delta,
        "message": f"已设置 {canteen_name} 的人流校准: {'+' if delta > 0 else ''}{delta}"
                   if delta != 0
                   else f"已清除 {canteen_name} 的人流校准",
    }


def clear_calibration(canteen_name: str) -> dict:
    """清除某食堂的校准。"""
    data = _load()
    existed = canteen_name in data
    data.pop(canteen_name, None)
    _save(data)
    return {
        "canteen_name": canteen_name,
        "cleared": existed,
        "message": f"已清除 {canteen_name} 的校准" if existed else f"{canteen_name} 本来就没有校准",
    }


def get_all_calibrations() -> dict[str, int]:
    """获取所有校准记录。"""
    return _load()
