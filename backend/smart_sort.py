"""
智能综合排序算法
================
输入用户经纬度，综合「距离」和「预估排队」两个维度打分，
返回从高到低排序的食堂列表。

评分公式:
  综合分 = 距离得分 * 0.6 + 排队得分 * 0.4

  - 距离得分: Haversine 直线距离越近 → 分越高 (满分 100)
  - 排队得分: 结合 base_hot_level + 当前时间段系数 → 越不挤分越高 (满分 100)
"""

import math
from datetime import datetime
from dataclasses import dataclass, asdict

from data import CANTEENS, Canteen
from calibration import get_calibration_delta


# ─── 权重配置 ───
WEIGHT_DISTANCE = 0.6
WEIGHT_QUEUE = 0.4

# 距离归一化上限 (km) — 超过此距离的食堂距离得分为 0
MAX_DISTANCE_KM = 2.0

# 排队分钟数映射: queue_score 100 → 5 min, queue_score 0 → 20 min
QUEUE_MIN_MIN = 5
QUEUE_MIN_MAX = 20

# 最大可能有效热度 = max(base_hot_level) * max(time_multiplier) = 5 * 1.5
MAX_EFFECTIVE_HOT = 7.5


# ─── 时间段系数 ───
def get_time_multiplier(now: datetime | None = None) -> tuple[float, str]:
    """
    根据当前系统时间返回 (时间系数, 时段描述)。

    时段划分 (遵循需求):
      08:00-11:30  上午:      0.8
      11:30-12:30  午餐高峰:  1.5
      12:30-13:30  午餐尾声:  1.2
      13:30-17:30  下午非高峰: 0.6
      17:30-19:00  晚餐高峰:  1.3
      19:00-20:00  晚餐尾声:  0.8
      其他          非营业:    0.3
    """
    if now is None:
        now = datetime.now()

    minutes = now.hour * 60 + now.minute

    if minutes < 8 * 60 or minutes >= 20 * 60:
        return 0.3, "非营业时段"
    elif 11 * 60 + 30 <= minutes < 12 * 60 + 30:
        return 1.5, "午餐高峰"
    elif 12 * 60 + 30 <= minutes < 13 * 60 + 30:
        return 1.2, "午餐尾声"
    elif 13 * 60 + 30 <= minutes < 17 * 60 + 30:
        return 0.6, "下午非高峰"
    elif 17 * 60 + 30 <= minutes < 19 * 60:
        return 1.3, "晚餐高峰"
    elif 19 * 60 <= minutes < 20 * 60:
        return 0.8, "晚餐尾声"
    else:
        # 08:00-11:30
        return 0.8, "上午时段"


# ─── Haversine 直线距离 ───
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """计算两个经纬度之间的直线距离 (km)，使用 Haversine 公式。"""
    R = 6371.0  # 地球半径 (km)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    return R * c


# ─── 评分函数 ───
def calc_distance_score(distance_km: float) -> float:
    """
    距离 → 得分 (0-100)。
    距离 0 km → 100 分; 距离 >= MAX_DISTANCE_KM → 0 分。
    线性衰减。
    """
    if distance_km >= MAX_DISTANCE_KM:
        return 0.0
    return round(100 * (1 - distance_km / MAX_DISTANCE_KM), 2)


def calc_queue_score(effective_hot: float) -> float:
    """
    有效热度 → 排队得分 (0-100)。
    越不挤 (effective_hot 越低) → 得分越高。
    effective_hot = 0 → 100 分; effective_hot = MAX_EFFECTIVE_HOT → 0 分。
    """
    score = 100 * (1 - min(effective_hot, MAX_EFFECTIVE_HOT) / MAX_EFFECTIVE_HOT)
    return round(score, 2)


def calc_queue_minutes(queue_score: float) -> int:
    """
    排队得分 → 预估排队分钟数。
    得分 100 → 5 分钟; 得分 0 → 20 分钟。线性映射。
    """
    minutes = QUEUE_MIN_MAX - (queue_score / 100) * (QUEUE_MIN_MAX - QUEUE_MIN_MIN)
    return max(1, round(minutes))


def get_queue_level(minutes: int) -> str:
    """排队分钟数 → 中文排队指数 (低/中/高)。"""
    if minutes <= 8:
        return "低"
    elif minutes <= 15:
        return "中"
    else:
        return "高"


# ─── 核心排序函数 ───
@dataclass
class SortResult:
    name: str
    name_en: str
    icon: str
    latitude: float
    longitude: float
    base_hot_level: int
    distance_km: float
    distance_meters: int
    distance_text: str           # "距离您XXX米" 或 "距离您X.X公里"
    distance_score: float        # 0-100
    effective_hot: float          # 实际热度 (含时间系数 + 人工校准)
    queue_score: float           # 0-100
    queue_minutes: int            # 预估排队分钟数
    queue_text: str              # "当前预估排队约X分钟"
    queue_level: str             # 低/中/高
    composite_score: float       # 综合分 0-100
    time_multiplier: float
    period_label: str
    calibration_delta: int        # 人工校准偏移量 (0 = 无校准)


def smart_sort(user_lat: float, user_lng: float, now: datetime | None = None) -> list[SortResult]:
    """
    智能综合排序主函数。

    参数:
        user_lat: 用户纬度
        user_lng: 用户经度
        now: 自定义时间 (测试用)，默认当前时间

    返回:
        按综合分从高到低排序的 SortResult 列表
    """
    if now is None:
        now = datetime.now()

    time_mult, period_label = get_time_multiplier(now)

    results: list[SortResult] = []

    for canteen in CANTEENS:
        # 1. 距离计算
        dist_km = haversine_km(user_lat, user_lng, canteen.latitude, canteen.longitude)
        dist_score = calc_distance_score(dist_km)

        # 2. 排队估算
        # 人工校准: 运营人员可手动 +1/-1 调整
        calib_delta = get_calibration_delta(canteen.name)
        # 有效热度 = (base_hot_level + 校准偏移) * 时间系数
        effective_hot = max(0, (canteen.base_hot_level + calib_delta)) * time_mult
        queue_score = calc_queue_score(effective_hot)

        # 3. 排队分钟数 & 排队指数
        queue_min = calc_queue_minutes(queue_score)
        queue_level = get_queue_level(queue_min)

        # 4. 综合分
        composite = round(dist_score * WEIGHT_DISTANCE + queue_score * WEIGHT_QUEUE, 2)

        # 5. 距离文案
        if dist_km < 1.0:
            dist_text = f"距离您{int(dist_km * 1000)}米"
        else:
            dist_text = f"距离您{dist_km:.1f}公里"

        results.append(SortResult(
            name=canteen.name,
            name_en=canteen.name_en,
            icon=canteen.icon,
            latitude=canteen.latitude,
            longitude=canteen.longitude,
            base_hot_level=canteen.base_hot_level,
            distance_km=round(dist_km, 3),
            distance_meters=int(dist_km * 1000),
            distance_text=dist_text,
            distance_score=dist_score,
            effective_hot=round(effective_hot, 2),
            queue_score=queue_score,
            queue_minutes=queue_min,
            queue_text=f"当前预估排队约{queue_min}分钟",
            queue_level=queue_level,
            composite_score=composite,
            time_multiplier=time_mult,
            period_label=period_label,
            calibration_delta=calib_delta,
        ))

    # 按综合分降序
    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results


def to_dict_list(results: list[SortResult]) -> list[dict]:
    """将 SortResult 列表转为可直接 JSON 序列化的 dict 列表。"""
    return [asdict(r) for r in results]
