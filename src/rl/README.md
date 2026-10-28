# 第 7 章　强化学习

教材本章无原始代码清单；本章代码是配套教学补充的可运行实现，用于课件中的数值实例与算法对照。
全部实现以 3×3 网格世界（图 7.2(a)）为统一环境，起点 (1,1)、目标 (3,3) 奖励 +10、
炸弹 (1,3) 奖励 −5、普通移动 −1，折扣因子 γ = 0.9。需要神经网络的方法改用 CartPole-v1。

## 代码清单

| 文件 | 对应教材 | 说明 |
|---|---|---|
| `envs/grid_world.py` | 7.1.3 节 | 网格世界环境：`p(s',r\|s,a)`、`p(s'\|s,a)`、`r(s,a)` 与采样接口，三种转移场景 |
| `envs/cartpole.py` | 7.1.3 节 | CartPole-v1 薄包装：`--render` 开关、随机种子、`terminated` 与 `truncated` 的区分 |
| `base.py` | 7.3 节 | 表格型智能体基类（`Q` 表 / `V` 表）、`ε`-贪心行为策略、评测入口 |
| `bellman.py` | 式 (7-3)–(7-23) | 贝尔曼方程数值实例：联合分布、边缘化、奖励函数、一条幕、期望式与最优式 |
| `transition.py` | 式 (7-3)–(7-5) | 状态转移概率与奖励函数的数值实例（对应课件「式 (7-3)–(7-5) 的数值实例」） |
| `why_v_is_harder.py` | 7.3 节 | 由 `V` 与由 `Q` 比较动作的代价，以及模型误差如何让 `V` 路线选错动作 |
| `dp.py` | 7.3 节 | 策略评估、策略改进、策略迭代、值迭代 |
| `mc.py` | 7.4 节 | 首次访问 MC 估计、试探性出发 MC 控制 |
| `td.py` | 7.5.1 节 | 表格型 TD(0) 策略评估 |
| `td_lambda.py` | 7.5.4 节 | `n` 步回报、`λ`-回报、TD(`λ`) |
| `sarsa.py` | 7.5.2 节 | SARSA 与期望 SARSA |
| `q_learning.py` | 7.5.3 节 | Q 学习 |
| `dqn.py` | 7.5.5 节 | DQN（经验回放 + 目标网络，CartPole） |
| `pg.py` | 7.6.3 节 | REINFORCE、带基线、离轨策略（CartPole） |
| `actor_critic.py` | 7.7.1 节 | 演员-评论家（CartPole） |
| `gae.py` | 7.8.2 节 | 广义优势估计的纯函数实现：递推式与展开式对照 |
| `ppo.py` | 7.7.3 节 | PPO-Clip，内部实现 GAE（CartPole） |
| `run_all.py` | 7.3–7.5 节 | 交叉验证：各表格型方法对照精确解 |

## 运行

```bash
conda activate xr1
cd token-to-llm/src/rl
python envs/grid_world.py
python bellman.py
python transition.py
python why_v_is_harder.py
python dp.py
python mc.py
python td.py
python td_lambda.py      # 约 50 秒
python sarsa.py
python q_learning.py
python dqn.py            # 约 15 秒
python pg.py             # 约 35 秒
python actor_critic.py   # 约 36 秒
python gae.py
python ppo.py            # 约 10 秒
python run_all.py        # 约 3 分钟
```

所有脚本在 `src/rl/` 目录下运行；`envs/`、`base.py` 是公共依赖。
加 `--render` 可在训练时逐帧显示网格世界或 CartPole 窗口，默认关闭。

## 三种转移场景

`GridWorld(scenario=...)` 只改变 `p(s'|s,a)`，奖励始终是确定函数。

| 场景 | 转移 | 用途 |
|---|---|---|
| `deterministic` | 动作必成功，撞墙留在原地 | 动态规划与值迭代的基线 |
| `slip` | 主方向 0.8，两个垂直方向各 0.1 | 让 `p(s',r\|s,a)` 非退化，MC / TD / SARSA 才有可采样的随机性 |
| `wind` | 主方向 0.8，0.2 概率额外下滑一格 | 模型误差如何让 `V` 路线选错动作 |

注意 `slip` 的一个副作用：以回报为目标时，(1,2) 的最优动作是「下」，而「下」的打滑分支会走进炸弹，
所以最优策略本身就有约 10% 的失败率。成功率与回报会把策略排出不同的顺序。

旧实现中「10% 奖励抖动」是第二个独立噪声源，已去掉：奖励抖动会让 `r(s,a)` 不再是确定函数，
`bellman.py` 的数值实例就写不成 $r(s,a)$ 的形式。现在只保留转移打滑。

## 主要数值结论

| 量 | 值 | 出处 |
|---|---|---|
| $p((2,3)\mid(2,2),\text{右})$ | 0.80 | `transition.py` |
| $r((2,2),\text{右})$ | −1.00 | `transition.py` |
| $r((1,2),\text{右})$ | −4.20 | `transition.py`（0.8 概率撞炸弹） |
| $v_\pi$（均匀随机策略，起点） | −6.091 | `bellman.py` |
| $v_*$（起点，`slip`） | 3.159 | `dp.py`、`bellman.py` |
| $v_*$（起点，`deterministic`） | 4.580 | `dp.py` |
| 策略迭代 / 值迭代轮数（`slip`） | 3 / 17 | `dp.py` |
| 模型误设为确定性后选错动作 | 3/7 个状态，平均 $v$ 由 7.934 降至 7.666 | `why_v_is_harder.py` |
| TD(`λ`) 的 `λ`=0 / `λ`=1 | 与 TD(0) / MC 回报在单条幕上逐项相等 | `td_lambda.py` |
| DQN / PPO / AC / REINFORCE 收敛 | 平均回报 494 / 500 / 500 / 500（上限 500） | `dqn.py`、`ppo.py`、`actor_critic.py`、`pg.py` |

## 许可

同仓库根目录 LICENSE：仅限教学与学习使用，不得用于商业用途。
