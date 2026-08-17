"""
HKU Smart Dining — 智能综合排序 API
====================================
FastAPI 后端，提供以下接口:

用户端:
  GET  /api/canteens/smart-sort?lat=22.28&lng=114.13
       → 返回按综合分排序的食堂列表

管理后台:
  GET    /api/admin/calibrations          → 查看所有人工校准
  POST   /api/admin/calibrate             → 设置/更新某食堂校准 (+1/-1)
  DELETE /api/admin/calibrate/{name}      → 清除某食堂校准

启动:
  cd backend
  uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

from data import CANTEENS, Canteen
from smart_sort import smart_sort, to_dict_list
from calibration import (
    get_all_calibrations,
    set_calibration,
    clear_calibration,
)

app = FastAPI(
    title="HKU Smart Dining API",
    description="智能综合排序 — 距离 + 预估排队",
    version="2.0.0",
)

# CORS — 允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为前端域名
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 数据模型 ───
class CalibrateRequest(BaseModel):
    canteen_name: str
    delta: int  # +1 或 -1


# ─── 用户端接口 ───

@app.get("/api/canteens/smart-sort")
def smart_sort_api(
    lat: float = Query(..., description="用户纬度", example=22.2835),
    lng: float = Query(..., description="用户经度", example=114.1375),
):
    """
    智能综合排序接口。

    综合分 = 距离得分 * 0.6 + 排队得分 * 0.4
    返回按综合分降序排列的食堂列表，附带距离文案和排队预估。
    """
    results = smart_sort(lat, lng)
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "count": len(results),
        "data": to_dict_list(results),
    }


@app.get("/api/canteens")
def list_canteens():
    """返回所有食堂数据 (静态信息)。"""
    return {
        "status": "ok",
        "count": len(CANTEENS),
        "data": [
            {
                "name": c.name,
                "name_en": c.name_en,
                "latitude": c.latitude,
                "longitude": c.longitude,
                "base_hot_level": c.base_hot_level,
                "icon": c.icon,
                "campus_zone": c.campus_zone,
            }
            for c in CANTEENS
        ],
    }


# ─── 管理后台接口 ───

@app.get("/api/admin/calibrations")
def list_calibrations():
    """查看所有当前生效的人工校准。"""
    calibs = get_all_calibrations()
    return {
        "status": "ok",
        "count": len(calibs),
        "data": [
            {"canteen_name": name, "delta": delta}
            for name, delta in calibs.items()
        ],
    }


@app.post("/api/admin/calibrate")
def calibrate(req: CalibrateRequest):
    """
    设置某食堂的人流校准偏移量。

    delta = +1: 人流偏多 (运营人员观察到比预期挤)
    delta = -1: 人流偏少 (运营人员观察到比预期空)
    delta =  0: 清除校准

    该偏移量会叠加到 base_hot_level 上，再乘以时间系数。
    """
    if req.delta not in (-1, 0, 1):
        raise HTTPException(status_code=400, detail="delta 只能是 +1、-1 或 0")

    # 验证食堂名
    canteen_names = [c.name for c in CANTEENS]
    if req.canteen_name not in canteen_names:
        raise HTTPException(
            status_code=404,
            detail=f"食堂 '{req.canteen_name}' 不存在。可选: {canteen_names}"
        )

    result = set_calibration(req.canteen_name, req.delta)
    return {"status": "ok", "data": result}


@app.delete("/api/admin/calibrate/{canteen_name}")
def remove_calibration(canteen_name: str):
    """清除某食堂的人工校准。"""
    result = clear_calibration(canteen_name)
    return {"status": "ok", "data": result}


# ─── 健康检查 ───
@app.get("/api/health")
def health():
    return {"status": "ok", "service": "HKU Smart Dining API", "version": "2.0.0"}
