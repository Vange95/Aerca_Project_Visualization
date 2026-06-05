# Aerca_Project_Visualization

AERCA 多变量时间序列根因分析的产品化可视化平台。

原论文代码请参考：https://github.com/hanxiao0607/AERCA

---

## 🆕 v2: FastAPI + React 重构版本

旧版本基于 Streamlit（保留为 `app.py`，仍可运行）。**新版本将前后端解耦**，便于后续扩展：

- **后端**：FastAPI + Uvicorn（REST API + WebSocket 实时进度推送）
- **前端**：React 18 + Vite + TypeScript + TailwindCSS + Plotly.js + Zustand
- **特性**：异步训练任务、WebSocket 实时 train/val loss 曲线、Session 多会话管理、生产期 FastAPI 单端口托管前端

### 整体架构

```
┌────────────────┐    REST + WebSocket    ┌────────────────────┐
│  React 前端    │ ◄────────────────────► │  FastAPI 后端      │
│  (Vite/TSX)    │                        │  (uvicorn)         │
└────────────────┘                        └────────┬───────────┘
                                                   │
                                                   ▼
                                ┌─────────────────────────────────┐
                                │ main.main(argv, callback,...)   │
                                │   ├─ datasets/{linear,...}      │
                                │   ├─ models.aerca.AERCA         │
                                │   │   └─ _training(progress_cb) │
                                │   ├─ _testing_causal_discover   │
                                │   └─ _testing_root_cause        │
                                └─────────────────────────────────┘
```

### 目录结构

```
Aerca_Project_Visualization/
├── app.py                    # ⓘ 旧版 Streamlit 入口（仍可用，作为备份）
├── main.py                   # 模型训练/测试主流程（已增加 progress_callback）
├── models/                   # AERCA + SENNGC 模型
├── datasets/                 # 6 个数据集生成器
├── args/                     # 各数据集参数 argparse 定义
├── utils/utils.py            # KL 散度 / POT 阈值 / topk 等工具
│
├── backend/                  # 🆕 FastAPI 后端
│   ├── server.py             # 主入口（REST + WebSocket + 静态托管）
│   ├── session_store.py      # 会话内存存储
│   ├── dataset_registry.py   # 6 个数据集懒加载注册表
│   ├── smoke_test.py         # API 端到端冒烟测试
│   └── requirements.txt
│
└── frontend/                 # 🆕 React 前端
    ├── package.json
    ├── vite.config.ts
    ├── src/
    │   ├── App.tsx
    │   ├── api.ts            # REST + WebSocket 客户端
    │   ├── store.ts          # Zustand 全局状态
    │   ├── types.ts
    │   └── components/
    │       ├── Sidebar.tsx       # 控制面板
    │       ├── DataView.tsx      # 数据可视化（正常/异常时序、热力图、因果矩阵）
    │       ├── ModelRunner.tsx   # 模型训练（带 WebSocket 实时 loss 曲线）
    │       └── ResultsView.tsx   # 根因高亮 + 因果发现矩阵对比 + 指标面板
    └── dist/                 # `npm run build` 产物（生产期由 FastAPI 托管）
```

---

## 📦 安装依赖

### 1. Python 后端

```bash
# 项目根目录
pip install -r requirements.txt
pip install -r backend/requirements.txt
```

> ⚠️ 已知环境兼容性问题：原项目依赖 `numba`、`scipy.optimize`，对 `numpy<2` 强制依赖。
> 如使用 `numpy>=2.0` 会出现 `cannot import name 'Inf' from 'numpy'` 等错误（不是本项目代码问题）。
> 推荐：`pip install "numpy<2" "scipy>=1.11" "numba>=0.59"`

### 2. Node 前端

```bash
cd frontend
npm install
```

需要 Node ≥ 18。

---

## 🚀 运行方式

### 方式 A：开发模式（前后端分离热重载）

```bash
# 终端 1：启动 FastAPI 后端（端口 8000）
python -m uvicorn backend.server:app --host 0.0.0.0 --port 8000 --reload

# 终端 2：启动 Vite 开发服务器（端口 5173，自动代理 /api 到 8000）
cd frontend
npm run dev
```

浏览器访问 `http://localhost:5173`。

### 方式 B：生产模式（单端口）

```bash
cd frontend && npm run build && cd ..
uvicorn backend.server:app --host 0.0.0.0 --port 8000
```

浏览器访问 `http://localhost:8000`，FastAPI 同时提供 API 和静态 React 资源。

### 方式 C：旧版 Streamlit（兼容保留）

```bash
streamlit run app.py
```

---

## 🔌 后端 API 概览

| 方法 | 路径 | 功能 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/datasets` | 列出所有支持的数据集 |
| GET | `/api/datasets/{name}/defaults` | 获取数据集默认参数 |
| POST | `/api/sessions` | 创建会话（生成或加载数据） |
| GET | `/api/sessions` | 列出所有会话 |
| GET | `/api/sessions/{sid}/info` | 会话状态摘要 |
| GET | `/api/sessions/{sid}/sample/{idx}` | 获取单个样本（自动选择测试集/原始） |
| GET | `/api/sessions/{sid}/causal` | 获取真实因果矩阵 |
| POST | `/api/sessions/{sid}/run` | 启动模型训练（异步） |
| GET | `/api/sessions/{sid}/status` | 查询运行状态 |
| GET | `/api/sessions/{sid}/progress` | 获取完整进度日志 |
| GET | `/api/sessions/{sid}/results` | 获取训练完成后的结构化结果 |
| WS | `/api/sessions/{sid}/ws` | 订阅实时训练进度（每 epoch 推送） |
| DELETE | `/api/sessions/{sid}` | 删除会话 |

API 详细 schema 见 `http://localhost:8000/docs`（FastAPI 自动生成的 Swagger UI）。

---

## ✅ 测试

后端冒烟测试（验证所有 API 端点的正确响应、异常路径、错误捕获机制）：

```bash
python -m backend.smoke_test
```

预期输出：
```
✅ Smoke test passed.
```

> 该测试会创建 linear 数据集会话、获取样本、获取因果矩阵、启动模型训练，并验证错误正确通过 API 报告（即使本地缺少 torch/scipy 等依赖也能验证 API 流程）。

前端构建验证：
```bash
cd frontend
npm run build
```

最近一次完整测试结果：**所有 API 端点通过**，前端 TypeScript 编译 + Vite 打包成功。

---

## 🔬 核心算法模块（保持不变）

### AERCA 模型 (`models/aerca.py`)

```
正常序列 x → [Encoder: SENNGC] → 残差 u → [Decoder: SENNGC] → 重建 x_hat
                                  │
                                  └→ KL(u, N(0,I)) 约束
```

- **编码器/解码器**：基于 GVAR 的自解释神经网络（SENNGC）
- **损失**：重建 + 系数稀疏性 + 系数平滑性 + KL 散度
- **训练**：仅用正常数据；早停（patience=20）
- **测试**：通过 z-score + POT (Peak-over-Threshold) 自适应阈值定位异常变量与时刻
- **新增**：`_training(xs, progress_callback)` 支持每 epoch 回调

### 主流程入口 (`main.py`)

```python
from main import main
results = main(argv, progress_callback=cb, data_class=existing_data_class)
# results 包含: root_cause_results, causal_results, test_x_ab, test_label, ...
```

### 6 个数据集

| 数据集 | 类型 | 异常类型 |
|---|---|---|
| linear | 合成 (4变量线性 VAR) | spike / step / causal |
| nonlinear | 合成 (非线性) | non_causal |
| lotka_volterra | 合成 (捕食者-猎物 ODE) | non_causal |
| lorenz96 | 合成 (混沌系统) | non_causal |
| msds | 真实工业传感器 | — |
| swat | 真实水处理系统 | — |

---

## 🖼️ 旧架构示意图（仍参考）

<img src="other/mermaid-diagram.svg" alt="AERCA 可视化流程图" width="800">
