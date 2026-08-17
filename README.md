# HKU Smart Dining App

港大食堂智能推荐系统 — 智能综合排序 + 人工校准

## 项目结构

```
hku_canteen_app/
├── hku_canteen_program.py      # 原始 Streamlit 应用 (M/G/1 排队模型)
├── requirements.txt             # Streamlit 依赖
│
├── backend/                     # FastAPI 后端 (新增)
│   ├── main.py                  # API 入口 (智能排序 + 管理后台接口)
│   ├── data.py                  # 食堂数据 (含 GPS 坐标 + base_hot_level)
│   ├── smart_sort.py            # 智能综合排序算法
│   ├── calibration.py           # 人工校准存储 (JSON 文件)
│   └── requirements.txt         # 后端依赖
│
├── frontend/                    # 前端页面 (新增)
│   ├── index.html              # 用户端首页 (Geolocation + 智能排序)
│   └── admin.html              # 管理后台 (人工校准)
│
└── README.md
```

## 快速启动

### 1. 启动后端 API

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

API 文档自动生成: http://localhost:8000/docs

### 2. 打开前端首页

直接用浏览器打开 `frontend/index.html`，或部署到任意静态服务器。

页面会自动:
1. 调用浏览器 Geolocation API 获取用户位置
2. 请求 `GET /api/canteens/smart-sort?lat=...&lng=...`
3. 渲染按综合分排序的食堂卡片

### 3. 打开管理后台

直接用浏览器打开 `frontend/admin.html`

运营人员可以:
- 查看所有食堂的基础热度
- 对某个食堂点 +1 (比预期挤) 或 -1 (比预期空)
- 校准立即生效，影响用户端排序

## 智能排序算法

### 评分公式

```
综合分 = 距离得分 × 0.6 + 排队得分 × 0.4
```

### 距离得分 (满分 100)

- 使用 Haversine 公式计算用户到食堂的直线距离 (km)
- 线性归一化: 0 km → 100 分, ≥2 km → 0 分
- 距离越近，得分越高

### 排队得分 (满分 100)

不直接用 `base_hot_level`，而是结合当前系统时间:

| 时段 | 时间 | 系数 |
|------|------|------|
| 上午 | 08:00-11:30 | 0.8 |
| 午餐高峰 | 11:30-12:30 | **1.5** |
| 午餐尾声 | 12:30-13:30 | 1.2 |
| 下午非高峰 | 13:30-17:30 | **0.6** |
| 晚餐高峰 | 17:30-19:00 | 1.3 |
| 晚餐尾声 | 19:00-20:00 | 0.8 |
| 非营业 | 其他 | 0.3 |

```
有效热度 = (base_hot_level + 人工校准偏移) × 时间系数
排队得分 = 100 × (1 - 有效热度 / 最大可能热度)
```

越不挤 → 排队得分越高。

### 排队分钟数映射

| 排队得分 | 预估排队 | 排队指数 |
|----------|----------|----------|
| 100 | 5 分钟 | 低 |
| 50 | 12-13 分钟 | 中 |
| 0 | 20 分钟 | 高 |

## API 接口一览

### 用户端

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/canteens/smart-sort?lat=22.28&lng=114.13` | 智能综合排序 |
| GET | `/api/canteens` | 获取所有食堂数据 |
| GET | `/api/health` | 健康检查 |

### 管理后台

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/admin/calibrations` | 查看所有人工校准 |
| POST | `/api/admin/calibrate` | 设置校准 (+1/-1) |
| DELETE | `/api/admin/calibrate/{canteen_name}` | 清除校准 |

## 人工校准机制

运营人员 (或食堂经理) 可以在管理后台手动调整某食堂的「当前人流系数」:

- **+1**: 当前比预期更挤 (如临时活动、雨天集中)
- **-1**: 当前比预期更空 (如食堂部分关闭)
- **清除**: 恢复算法默认值

校准偏移叠加到 `base_hot_level` 上，再乘以时间系数，**立即生效**于用户端排序。

存储方式: `backend/calibrations.json` (JSON 文件，无需数据库)
