"""
HKU 食堂数据 — 包含真实 GPS 坐标和静态热度等级
================================================
base_hot_level: 1-5，5 代表最热门（最拥挤）
坐标为港大各食堂的近似经纬度（22.28xx N, 114.13xx E）
"""

from dataclasses import dataclass


@dataclass
class Canteen:
    name: str           # 中文名
    name_en: str        # 英文名
    latitude: float     # 纬度
    longitude: float   # 经度
    base_hot_level: int # 1-5，5 = 最热门
    icon: str           # emoji 图标
    campus_zone: str    # 校园区: main / centenary / mtr / medical


# 13 个港大食堂 — 坐标基于港大校园实际位置
CANTEENS: list[Canteen] = [
    Canteen("庄月明食堂",       "Meng Wah Complex",          22.2820, 114.1368, 5, "\U0001F35C", "main"),
    Canteen("学生会食堂",       "Student Union Canteen",     22.2828, 114.1375, 5, "\U0001F371", "main"),
    Canteen("方树泉食堂",       "Fong Shu Chuen Hall",       22.2815, 114.1365, 4, "\U0001F961", "main"),
    Canteen("亚洲滋味餐厅",     "Asian Flavours",            22.2825, 114.1370, 3, "\U0001F372", "main"),
    Canteen("一念素食",         "Yi Nian Vegetarian",        22.2830, 114.1360, 3, "\U0001F957", "main"),
    Canteen("cafe330",          "Cafe 330",                  22.2868, 114.1395, 3, "\u2615",     "centenary"),
    Canteen("Coffee Academics", "Coffee Academics",          22.2870, 114.1398, 2, "\u2615",     "centenary"),
    Canteen("U Deli",           "U Deli",                    22.2835, 114.1375, 2, "\U0001F96A", "main"),
    Canteen("alfafa cafe",      "Alfafa Cafe",               22.2865, 114.1390, 2, "\U0001F96A", "centenary"),
    Canteen("Sandwich Club",    "Sandwich Club",             22.2822, 114.1362, 2, "\U0001F96A", "main"),
    Canteen("Super Sandwiches", "Super Sandwiches",          22.2839, 114.1319, 1, "\U0001F96A", "mtr"),
    Canteen("Subway",           "Subway",                    22.2840, 114.1315, 2, "\U0001F956", "mtr"),
    Canteen("星巴克",           "Starbucks",                 22.2872, 114.1400, 1, "\u2615",     "centenary"),
]

# 按名称建索引，方便快速查找
CANTEEN_MAP: dict[str, Canteen] = {c.name: c for c in CANTEENS}


def get_canteen(name: str) -> Canteen | None:
    """按中文名查找食堂"""
    return CANTEEN_MAP.get(name)
