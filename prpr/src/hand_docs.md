# Hand.rs 模块技术文档

## 1. 模块架构详解

### 1.1 整体架构图
```
┌─────────────────────────────────────────────────────────────────┐
│                        Hand Assignment System                   │
├───────────────┬───────────────┬───────────────┬───────────────┤
│  AI Worker    │ Request       │ Response      │ Line State    │
│  Thread       │ Queue         │ Queue         │ Management    │
├───────────────┼───────────────┼───────────────┼───────────────┤
│ Async Processing │ Thread Pool │ Feature       │ Note Matching │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心组件交互流程
1. **请求发起**：主线程调用`assign_hands`函数
2. **状态检查**：检查谱面行(line)的更新状态和待处理请求数
3. **请求提交**：创建AI处理请求并发送到工作线程
4. **异步处理**：AI工作线程处理请求并返回结果
5. **结果合并**：主线程接收响应并合并到原始音符数据

## 2. 核心结构深度解析

### 2.1 主要数据结构

#### AiRequest 结构体
```rust
struct AiRequest {
    id: u64,                    // 请求唯一ID
    line_id: usize,             // 谱面行ID
    version: u64,               // 请求版本号
    timestamp: Instant,         // 请求时间戳
    notes: Vec<Note>,           // 音符数据快照
    rotation: f32,              // 谱面旋转角度
    config: Arc<Config>,        // 游戏配置
    bpm_list: Arc<BpmList>,     // BPM列表
}
```

#### AiResponse 结构体
```rust
struct AiResponse {
    id: u64,                    // 对应请求ID
    line_id: usize,             // 谱面行ID
    version: u64,               // 响应版本号
    timestamp: Instant,         // 响应时间戳
    notes: Vec<Note>,           // 处理后的音符数据
    checksum: u64,              // 数据校验和
}
```

#### LineState 结构体
```rust
struct LineState {
    current_version: u64,                       // 当前版本
    last_full_update: Instant,                  // 上次完整更新时间
    last_light_update: Instant,                 // 上次轻量更新时间
    pending_requests: HashMap<u64, (Instant, u64)>, // 待处理请求
}
```

### 2.2 PhiTKAdvancedAI 结构体

#### 核心字段
| 字段 | 类型 | 作用 |
|------|------|------|
| `main_network` | DeepNeuralNetwork | 主神经网络 |
| `target_network` | DeepNeuralNetwork | 目标网络（用于Q-learning） |
| `feature_extractor` | AdvancedFeatureExtractor | 特征提取器 |
| `experience_replay` | ExperienceReplay | 经验回放缓冲区 |
| `rotation` | f32 | 谱面旋转角度 |
| `game_mode` | GameMode | 游戏模式（2指/4指） |
| `recent_assignments` | VecDeque<(Hand, f32, f32)> | 最近分配记录 |
| `hand_switch_count` | u32 | 手部切换计数 |

#### 神经网络架构
- **输入层**: 40维特征向量
- **LSTM层**: 3层双向LSTM (40→128→128→128 hidden units)
- **Attention层**: 2层注意力机制 (256→128→64)
- **Dense层**: 6层全连接层 (64→512→256→256→128→128→64)
- **输出层**: 5维输出 (左右手置信度、稳定性指标等)

#### GPU加速支持
- 使用WebGPU进行矩阵运算加速
- 自动检测软件渲染器并回退到CPU
- 支持批量处理（batch_size=256）
- 异步缓冲区映射和数据传输

## 3. 核心算法实现

### 3.1 手部分配主流程

```rust
pub fn assign_hands(notes: &mut [Note], config: &Config, line_id: usize, rotation: f32, bpm_list: &BpmList)
```

**执行流程**:
1. 检查音符数组是否为空
2. 启动AI工作线程（首次调用时）
3. 获取谱面行状态
4. 处理已完成的响应（批量处理最多5个）
5. 判断是否需要发起新的完整更新请求
6. 限制待处理请求数量（最多3个）

### 3.2 音符匹配与合并算法

```rust
fn match_and_merge_notes(original: &mut [Note], updated: &[Note]) -> bool
```

**匹配策略**:
- **时间窗口**: 50ms (0.05秒)
- **位置阈值**: 20像素距离 (400平方)
- **类型匹配**: 确保音符类型相同
- **最优匹配**: 使用时间+距离的综合评分

### 3.3 特征提取实现

#### ProcessedNote 结构
```rust
struct ProcessedNote {
    index: usize,           // 音符索引
    position: Vector2,      // 位置坐标
    time: f32,              // 时间戳
    kind: NoteKind,         // 音符类型
    assigned_hand: Option<Hand>, // 分配的手部
    confidence: f32,        // 置信度
    features: Vec<f32>,     // 特征向量
    judge: JudgeStatus,     // 判定状态
    difficulty: f32,        // 难度评分
    duration: f32,          // 持续时间
}
```

#### Vector2 工具方法
- `distance_to()`: 计算两点间距离
- `rotate()`: 旋转向量
- `magnitude()`: 计算向量长度
- `normalize()`: 向量归一化
- `dot()`: 点积计算
- `clean()`: 清理非有限值

### 3.4 神经网络前向传播

#### CPU实现 (`light_forward`)
- 顺序处理各层
- 支持Dense、LSTM、Attention、Residual层类型
- 使用并行计算优化矩阵运算

#### GPU实现 (`gpu_forward`)
- 创建输入/输出缓冲区
- 矩阵乘法计算
- 激活函数应用
- 异步结果读取
- 失败时自动回退到CPU

## 4. 异步处理机制

### 4.1 线程模型
- **主线程**: 负责游戏逻辑和UI渲染
- **AI工作线程**: 专门处理手部分配请求
- **响应分发线程**: 处理AI响应的分发

### 4.2 请求管理
- **请求频率**: 每20ms最多发起一次完整更新
- **超时机制**: 请求5秒后自动过期
- **资源限制**: 每行最多3个待处理请求
- **批量响应**: 每帧最多处理5个响应

### 4.3 错误处理
- **Panic防护**: 使用`catch_unwind`防止AI崩溃影响主线程
- **资源清理**: 定期清理过期请求（每30秒）
- **校验机制**: 使用checksum验证响应数据完整性

## 5. 性能优化策略

### 5.1 内存优化
- 使用`Arc`共享不可变数据（Config、BpmList）
- 音符数据按需克隆，避免不必要的内存分配
- 使用`VecDeque`实现高效的队列操作

### 5.2 计算优化
- 并行处理神经网络的矩阵运算
- 批量处理多个响应减少锁竞争
- 预排序音符索引优化匹配算法

### 5.3 GPU优化
- 异步缓冲区管理
- 工作组大小优化（64线程/组）
- 资源重用减少创建开销

## 6. 配置与调优

### 6.1 关键参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `FULL_UPDATE_INTERVAL_MS` | 20 | 完整更新间隔(ms) |
| `REQUEST_TIMEOUT_MS` | 5000 | 请求超时时间(ms) |
| `MAX_PENDING_REQUESTS` | 3 | 最大待处理请求数 |
| `MAX_RESPONSES_PER_FRAME` | 5 | 每帧最大响应数 |

### 6.2 神经网络参数
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `learning_rate` | 0.001 | 学习率 |
| `momentum` | 0.9 | 动量参数 |
| `dropout_rate` | 0.1 | Dropout率 |
| `batch_size` | 256 | 批处理大小 |
| `max_grad_norm` | 10.0 | 梯度裁剪阈值 |

## 7. 使用示例

### 7.1 基本用法
```rust
// 在游戏场景中调用手部分配
assign_hands(
    &mut judge_line.notes,
    &config,
    judge_line.id,
    judge_line.rotation,
    &bpm_list
);
```

### 7.2 模型保存与加载
```rust
// 创建或加载AI模型
let mut ai = PhiTKAdvancedAI::load_or_create("phitk_ai_model.bin", rotation, &config);

// 初始化GPU（可选）
ai.main_network.init_gpu_sync();
ai.target_network.init_gpu_sync();
```

## 8. 调试与监控

### 8.1 性能指标
- `TOTAL_TOKENS_USED`: 总处理请求数
- `correct_predictions`: 正确预测数
- `average_reward`: 平均奖励值
- `training_episodes`: 训练轮次

### 8.2 日志输出
- GPU初始化状态
- 请求/响应处理状态
- 错误和警告信息

---
*本文档基于实际代码实现，最后更新时间：2025-10-06*