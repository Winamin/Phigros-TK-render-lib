# Hand.rs 模块文档

## 概述

`hand.rs` 是 Phi-TK Render Library 中核心的手部分配和人工智能决策模块。该模块实现了一套先进的神经网络系统，用于在节奏游戏中自动分配音符到左右手，特别优化了2指模式和4指模式的游戏体验。

## 主要功能

### 1. 智能手部分配
- 基于位置、时间、音符类型的智能手部分配
- 支持多种游戏模式（2指/4指）
- 动态模式切换和适应
- 双押音符的优化处理

### 2. 神经网络系统
- 深度神经网络（DNN）用于决策优化
- 强化学习算法持续改进分配策略
- GPU 加速计算支持
- 经验回放机制

### 3. 高级特征提取
- 空间特征分析
- 时间模式识别
- 难度评估
- 节奏复杂度计算

## 核心结构

### PhiTKAdvancedAI

主要的人工智能结构，负责整体的手部分配决策。

```rust
pub struct PhiTKAdvancedAI {
    main_network: DeepNeuralNetwork,
    target_network: DeepNeuralNetwork,
    feature_extractor: AdvancedFeatureExtractor,
    experience_replay: ExperienceReplay,
    left_hand_state: HandState,
    right_hand_state: HandState,
    finger_states: Vec<FingerState>,
    game_mode: GameMode,
    // ... 更多字段
}
```

#### 主要功能

- **神经网络决策**：使用深度神经网络进行手部分配决策
- **模式识别**：识别交替、连打、和弦等模式
- **自适应学习**：根据游戏表现动态调整策略
- **多线程处理**：支持并行计算提高性能

### DeepNeuralNetwork

深度神经网络实现，支持多种层类型和激活函数。

```rust
pub struct DeepNeuralNetwork {
    layers: Vec<NetworkLayer>,
    learning_rate: f32,
    momentum: f32,
    batch_size: usize,
    // GPU 相关字段
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    // ...
}
```

#### 特性

- **多种层类型**：Dense、LSTM、Attention、Residual
- **GPU 加速**：支持 WebGPU/WGPU 加速计算
- **批量处理**：支持批量输入处理
- **动态学习率**：自适应学习率调整

### FingerState

手指状态管理，跟踪每个手指的实时状态。

```rust
pub struct FingerState {
    finger: Finger,
    position: Vector2,
    velocity: Vector2,
    fatigue: f32,
    confidence: f32,
    is_busy: bool,
    busy_until: f32,
    // ...
}
```

### AdvancedFeatureExtractor

高级特征提取器，用于分析音符序列的各种特征。

```rust
pub struct AdvancedFeatureExtractor {
    pattern_library: HashMap<String, PatternSignature>,
    temporal_patterns: VecDeque<TemporalFeature>,
    spatial_patterns: Vec<SpatialFeature>,
    difficulty_estimator: DifficultyEstimator,
    // ...
}
```

## 游戏模式

### TwoFinger (2指模式)

专为2指操作优化的模式，重点改进了双押处理。

#### 特性

- **智能双押处理**：在同一个x轴区域内积极使用左右手
- **交替模式优化**：减少连续使用同一手的情况
- **位置容忍度**：增加位置分配的灵活性
- **平衡性奖励**：鼓励双手平衡使用

#### 双押处理逻辑

```rust
// 在2指双押模式下，优先将音符分配给离手部较近的位置
for (pos, (note_idx, x)) in sorted_group.iter().enumerate() {
    let hand = if pos % 2 == 0 {
        // 偶数位置音符，根据x坐标相对于中心的位置决定
        if *x <= center_x {
            Hand::Left
        } else {
            Hand::Right
        }
    } else {
        // 奇数位置音符，优先分配给另一只手，确保双手交替
        if assigned_left <= assigned_right && assigned_left < max_per_hand {
            Hand::Left
        } else if assigned_right < max_per_hand {
            Hand::Right
        } else {
            Hand::Left
        }
    };
}
```

### FourFinger (4指模式)

支持4指操作的高级模式，提供更精确的手指分配。

#### 特性

- **四指支持**：左右手各两指（食指和中指）
- **精确分配**：基于位置的精确手部分配
- **复杂模式处理**：处理复杂的音符模式
- **性能优化**：针对高密度音符的优化

## 核心算法

### 1. 手部分配算法

#### 主要步骤

1. **音符预处理**：将原始音符转换为处理后的格式
2. **同时音符检测**：识别时间上相近的音符组
3. **模式分析**：分析音符序列的模式特征
4. **AI决策**：使用神经网络进行手部分配
5. **后处理优化**：优化分配结果，确保合理性

#### 关键函数

```rust
pub fn analyze_and_assign(&mut self, notes: &mut [Note], config: &Config, bpm_list: &BpmList, line_id: usize) {
    // 预处理音符
    let mut processed_notes = self.preprocess_notes(notes);
    
    // 检测同时音符组
    let simultaneous_groups = self.detect_simultaneous_groups(&processed_notes);
    
    // 分配同时音符组
    self.assign_simultaneous_groups(&mut processed_notes, &simultaneous_groups, bpm_list, line_id);
    
    // AI分配单个音符
    self.ai_assign_single_notes(&mut processed_notes, &simultaneous_groups, bpm_list, line_id);
    
    // 后处理优化
    self.post_process_assignments(&mut processed_notes);
}
```

### 2. 神经网络训练

#### 训练流程

1. **经验收集**：记录游戏过程中的决策和结果
2. **经验回放**：从历史经验中采样训练数据
3. **网络训练**：使用批量梯度下降训练网络
4. **目标网络更新**：定期更新目标网络参数

#### 关键函数

```rust
fn train_network(&mut self) {
    let batch_size = 24;
    let experiences: Vec<_> = self.experience_replay.sample(batch_size).into_iter().cloned().collect();
    
    let mut training_data = Vec::with_capacity(batch_size);
    for exp in &experiences {
        let target_value = (exp.reward + self.discount_factor * self.estimate_future_value(&exp.next_state)).clamp(0.0, 1.0);
        let mut target_output = vec![0.5, 0.5, 1.0, normalized_reward];
        if exp.action < target_output.len() {
            target_output[exp.action] = target_value;
        }
        training_data.push((exp.state.clone(), target_output));
    }
    
    self.main_network.train(&training_data);
}
```

### 3. 特征提取算法

#### 提取的特征类型

1. **位置特征**：音符的坐标、距离、角度等
2. **时间特征**：时间间隔、节奏复杂度、BPM等
3. **模式特征**：交替、连打、和弦、跳键等模式
4. **难度特征**：基础难度、速度难度、协调难度等
5. **空间特征**：分布范围、中心位置、方向等

#### 关键函数

```rust
pub fn extract_features(&mut self, notes: &[ProcessedNote], window_size: usize, bpm_list: &BpmList) -> Vec<f32> {
    let mut features = Vec::new();
    features.extend(self.extract_position_features(notes));
    features.extend(self.extract_temporal_features(notes, window_size, bpm_list));
    features.extend(self.extract_pattern_features(notes));
    features.extend(self.extract_difficulty_features(notes));
    features.extend(self.extract_velocity_features(notes));
    features.extend(self.extract_spatial_features(notes));
    
    // 特征归一化
    self.normalize_features(&mut features);
    
    features
}
```

## 使用方法

### 基本使用

```rust
use prpr::hand::{assign_hands, Config, BpmList};

// 准备数据
let mut notes = Vec::new(); // 你的音符数据
let config = Config::default();
let bpm_list = BpmList::new();
let line_id = 0;
let rotation = 0.0;

// 分配手部
assign_hands(&mut notes, &config, line_id, rotation, &bpm_list);
```

### 高级配置

```rust
use prpr::hand::{PhiTKAdvancedAI, GameMode};

// 创建AI实例
let mut ai = PhiTKAdvancedAI::new(0.0); // 0.0度旋转

// 设置游戏模式
ai.game_mode = GameMode::TwoFinger;

// 自定义参数
ai.exploration_rate = 0.1;
ai.confidence_threshold = 0.7;
ai.stability_factor = 0.85;

// 处理音符
ai.analyze_and_assign(&mut notes, &config, &bpm_list, line_id);
```

## 性能优化

### 1. GPU 加速

模块支持使用 WebGPU 进行神经网络计算加速：

```rust
// 启用GPU加速
ai.main_network.init_gpu_sync();
ai.target_network.init_gpu_sync();
```

### 2. 多线程处理

使用 Rayon 库进行并行计算：

```rust
// 创建线程池
let thread_pool = rayon::ThreadPoolBuilder::new()
    .num_threads(32)
    .build()
    .ok().map(Arc::new);

ai.thread_pool = thread_pool;
```

### 3. 批量处理

支持批量处理多个音符以提高效率：

```rust
const BATCH_SIZE: usize = 24;
let outputs = ai.main_network.gpu_forward_batch(&feature_slices, batch_indices.len());
```

## 配置参数

### 主要参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `exploration_rate` | f32 | 0.05 | 探索率，控制随机决策的概率 |
| `learning_rate` | f32 | 0.001 | 神经网络学习率 |
| `confidence_threshold` | f32 | 0.7 | 置信度阈值 |
| `stability_factor` | f32 | 0.85 | 稳定性因子 |
| `hand_switch_penalty` | f32 | 0.4 | 手部切换惩罚 |
| `consistency_bonus` | f32 | 0.3 | 一致性奖励 |

### 2指模式专用参数

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `consecutive_threshold` | usize | 1 | 连续使用同一手的阈值 |
| `position_tolerance` | f32 | 0.15 | 位置容忍度 |
| `center_region_bonus` | f32 | 0.2 | 中心区域奖励 |
| `balance_threshold` | f32 | 0.6 | 平衡性阈值 |

## 模式识别

### 支持的模式

1. **交替模式 (Alternating)**
   - 识别左右手交替的音符序列
   - 适用于经典的交叉音符模式

2. **连打模式 (Stream)**
   - 识别连续的单向音符流
   - 优化高速连打的分配策略

3. **和弦模式 (Chord)**
   - 识别同时出现的多个音符
   - 优化多音符同时处理

4. **跳键模式 (Jack)**
   - 识别相同位置的连续音符
   - 优化重复位置的快速处理

5. **交叉模式 (Crossing)**
   - 识别跨越中心线的音符
   - 优化交叉移动的处理

### 模式优化

每种模式都有专门的优化算法：

```rust
fn optimize_alternating_pattern(&self, notes: &mut [ProcessedNote]) {
    for i in 1..notes.len() {
        if let Some(prev_hand) = notes[i - 1].assigned_hand {
            let should_alternate = notes[i].position.x * notes[i - 1].position.x < 0.0;
            if should_alternate {
                let opposite_hand = if prev_hand == Hand::Left { Hand::Right } else { Hand::Left };
                notes[i].assigned_hand = Some(opposite_hand);
                notes[i].confidence = 0.85;
            }
        }
    }
}
```

## 调试和监控

### 日志输出

模块提供详细的日志输出用于调试：

```rust
// 启用详细日志
println!("[AI决策] 线路{} 时间{:.2}s 位置({:.2},{:.2}) 网络输出: L:{:.3} R:{:.3}",
         line_id, note.time, note.position.x, note.position.y,
         left_ai_confidence, right_ai_confidence);
```

### 性能指标

跟踪关键性能指标：

```rust
pub struct PerformanceMetrics {
    timestamp: f32,
    accuracy: f32,
    speed: f32,
    consistency: f32,
    difficulty_handled: f32,
    patterns_recognized: Vec<String>,
}
```

### 模型保存和加载

支持模型的保存和加载：

```rust
// 保存模型
ai.save_model("phitk_ai_model.bin");

// 加载模型
let mut ai = PhiTKAdvancedAI::load_or_create("phitk_ai_model.bin", rotation);
```

## 故障排除

### 常见问题

1. **GPU 初始化失败**
   - 确保 WebGPU 支持已启用
   - 检查显卡驱动是否最新
   - 系统会自动回退到 CPU 计算

2. **内存使用过高**
   - 调整经验回放缓冲区大小
   - 减少批量处理大小
   - 优化线程池大小

3. **决策质量不佳**
   - 增加训练数据
   - 调整学习率和探索率
   - 检查特征提取的质量

### 调试技巧

1. **启用详细日志**
   ```rust
   println!("[Token Usage] AI Decision - Features: {}, Time: {:.2}", features.len(), note.time);
   ```

2. **监控网络输出**
   ```rust
   println!("Raw network output: {:?}", raw_output);
   ```

3. **检查手部状态**
   ```rust
   println!("Hand states - Left: {:?}, Right: {:?}", ai.left_hand_state, ai.right_hand_state);
   ```

## 示例代码

### 完整示例

```rust
use prpr::hand::{PhiTKAdvancedAI, Config, BpmList, GameMode};

fn main() {
    // 初始化AI
    let mut ai = PhiTKAdvancedAI::new(0.0);
    ai.game_mode = GameMode::TwoFinger;
    
    // 配置参数
    ai.exploration_rate = 0.05;
    ai.confidence_threshold = 0.7;
    
    // 准备数据
    let config = Config::default();
    let bpm_list = BpmList::new();
    let mut notes = Vec::new(); // 填充音符数据
    
    // 处理音符
    ai.analyze_and_assign(&mut notes, &config, &bpm_list, 0);
    
    // 输出结果
    for (i, note) in notes.iter().enumerate() {
        println!("Note {}: Hand = {:?}, Confidence = {:.3}", i, note.hand, note.confidence);
    }
    
    // 保存模型
    ai.save_model("my_ai_model.bin");
}
```

### 自定义特征提取

```rust
use prpr::hand::{AdvancedFeatureExtractor, ProcessedNote};

struct CustomFeatureExtractor {
    base_extractor: AdvancedFeatureExtractor,
}

impl CustomFeatureExtractor {
    fn extract_custom_features(&mut self, notes: &[ProcessedNote]) -> Vec<f32> {
        let mut features = Vec::new();
        
        // 提取基础特征
        features.extend(self.base_extractor.extract_features(notes, 8, &BpmList::new()));
        
        // 添加自定义特征
        let custom_feature = self.calculate_custom_pattern(notes);
        features.push(custom_feature);
        
        features
    }
    
    fn calculate_custom_pattern(&self, notes: &[ProcessedNote]) -> f32 {
        // 实现自定义模式计算
        0.0
    }
}
```

## 贡献指南

### 代码风格

- 遵循 Rust 官方代码风格
- 使用有意义的变量名和函数名
- 添加适当的注释和文档

### 测试

- 为新功能编写单元测试
- 确保所有测试通过
- 进行性能基准测试

### 文档

- 更新相关文档
- 添加示例代码
- 记录API变更

## 许可证

本模块遵循与主项目相同的许可证。

## 更新日志

### v0.4.0
- 优化了2指模式的双押处理
- 改进了神经网络训练算法
- 增加了GPU加速支持
- 添加了更多模式识别功能

### v0.3.0
- 引入了强化学习算法
- 支持多线程处理
- 改进了特征提取算法

### v0.2.0
- 基础手部分配功能
- 简单的神经网络实现
- 支持2指和4指模式

### v0.1.0
- 初始版本
- 基本的手部分配逻辑

---

*本文档最后更新时间：2025-09-29*