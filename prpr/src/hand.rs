use crate::config::Config;
use std::collections::{HashMap, VecDeque, BTreeMap};
use crate::core::Note;
use crate::core::note::Hand;
use crate::core::NoteKind;
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;
use bincode;

use once_cell::sync::OnceCell;
use std::sync::Mutex;
use std::time::{Instant, Duration};
use crate::core::BpmList;
use crate::judge::JudgeStatus;

pub struct HandConfig {
    pub config: Config,
}

static AI_SYSTEM: OnceCell<Mutex<PhiTKAdvancedAI>> = OnceCell::new();
static LAST_FULL_UPDATE: OnceCell<Mutex<Instant>> = OnceCell::new();
static LAST_LIGHT_UPDATE: OnceCell<Mutex<Instant>> = OnceCell::new();

pub fn assign_hands(notes: &mut [Note], config: &Config, rotation: f32, bpm_list: &BpmList) {
    if notes.is_empty() {
        return;
    }

    let now = Instant::now();

    // 初始化全局计时器
    let last_full_update_mutex = LAST_FULL_UPDATE.get_or_init(|| Mutex::new(now));
    let last_light_update_mutex = LAST_LIGHT_UPDATE.get_or_init(|| Mutex::new(now));

    let mut last_full_update = last_full_update_mutex.lock().unwrap();
    let mut last_light_update = last_light_update_mutex.lock().unwrap();

    // 全局单例，首次调用时加载模型，后续复用
    let ai_mutex = AI_SYSTEM.get_or_init(|| {
        let ai = PhiTKAdvancedAI::load_or_create("phitk_ai_model.bin", rotation);
        Mutex::new(ai)
    });

    let mut ai = ai_mutex.lock().unwrap();

    if now.duration_since(*last_light_update) >= Duration::from_millis(1) {
        *last_light_update = now;
        ai.light_update_hand_states(notes);
    }

    if now.duration_since(*last_full_update) >= Duration::from_millis(5) {
        *last_full_update = now;

        ai.rotation = rotation;
        ai.reset_for_new_chart();
        // 将BPM列表传递给AI分析
        ai.analyze_and_assign(notes, config, bpm_list);

        // 每 10000 回合保存一次
        const SAVE_EVERY_EPISODES: u64 = 10000;
        if ai.training_episodes % SAVE_EVERY_EPISODES == 0 {
            ai.save_model("phitk_ai_model.bin");
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Vector2 {
    x: f32,
    y: f32,
}

impl Vector2 {
    fn new(x: f32, y: f32) -> Self { Self { x, y } }

    fn distance_to(&self, other: &Vector2) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    fn rotate(&self, angle_rad: f32) -> Vector2 {
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();
        Vector2 {
            x: self.x * cos_a - self.y * sin_a,
            y: self.x * sin_a + self.y * cos_a,
        }
    }

    fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalize(&self) -> Vector2 {
        let mag = self.magnitude();
        if mag > 0.001 {
            Vector2 { x: self.x / mag, y: self.y / mag }
        } else {
            *self
        }
    }

    fn dot(&self, other: &Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn clean(&mut self) {
        if !self.x.is_finite() {
            self.x = 0.0;
        }
        if !self.y.is_finite() {
            self.y = 0.0;
        }
    }
}

impl std::ops::Add for Vector2 {
    type Output = Vector2;
    fn add(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x + other.x, self.y + other.y)
    }
}

impl std::ops::Sub for Vector2 {
    type Output = Vector2;
    fn sub(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x - other.x, self.y - other.y)
    }
}

impl std::ops::Mul<f32> for Vector2 {
    type Output = Vector2;
    fn mul(self, scalar: f32) -> Vector2 {
        Vector2::new(self.x * scalar, self.y * scalar)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GameMode {
    TwoFinger,   // 二指模式：左右手各一个食指
    FourFinger,  // 四指模式：左右手各食指+中指
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Finger {
    LeftIndex,      // 左食指
    LeftMiddle,     // 左中指
    RightIndex,     // 右食指
    RightMiddle,    // 右中指
}

impl Finger {
    pub fn to_hand(&self) -> Hand {
        match self {
            Finger::LeftIndex | Finger::LeftMiddle => Hand::Left,
            Finger::RightIndex | Finger::RightMiddle => Hand::Right,
        }
    }

    pub fn is_index(&self) -> bool {
        matches!(self, Finger::LeftIndex | Finger::RightIndex)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FingerState {
    finger: Finger,
    position: Vector2,
    velocity: Vector2,
    #[serde(default)]
    last_time: f32,
    #[serde(default)]
    fatigue: f32,
    #[serde(default)]
    confidence: f32,
    success_streak: u32,
    total_actions: u32,
    #[serde(default)]
    performance_score: f32,
    #[serde(default)]
    is_busy: bool,
    #[serde(default)]
    busy_until: f32,
}

impl FingerState {
    fn new(finger: Finger, initial_pos: Vector2) -> Self {
        Self {
            finger,
            position: initial_pos,
            velocity: Vector2::new(0.0, 0.0),
            last_time: -1.0,
            fatigue: 0.0,
            confidence: 1.0,
            success_streak: 0,
            total_actions: 0,
            performance_score: 1.0,
            is_busy: false,
            busy_until: -1.0,
        }
    }

    pub fn clean(&mut self) {
        self.position.clean();
        self.velocity.clean();
        if !self.last_time.is_finite() {
            self.last_time = -1.0;
        }
        if !self.fatigue.is_finite() {
            self.fatigue = 0.0;
        }
        if !self.confidence.is_finite() {
            self.confidence = 1.0;
        }
        if !self.performance_score.is_finite() {
            self.performance_score = 1.0;
        }
        if !self.busy_until.is_finite() {
            self.busy_until = -1.0;
        }
    }
/*
    fn validate(&self) -> bool {
        self.position.x.is_finite() &&
            self.position.y.is_finite() &&
            self.velocity.x.is_finite() &&
            self.velocity.y.is_finite() &&
            self.last_time.is_finite() &&
            self.fatigue.is_finite() &&
            self.confidence.is_finite() &&
            self.performance_score.is_finite() &&
            self.busy_until.is_finite()
    }


 */
    fn update_state(&mut self, new_pos: Vector2, time: f32, success: bool, note_kind: &NoteKind) {
        let time_diff = time - self.last_time;

        if time_diff > 0.001 {
            let distance = new_pos.distance_to(&self.position);
            let new_velocity = (new_pos - self.position) * (1.0 / time_diff);
            self.velocity = self.velocity * 0.6 + new_velocity * 0.4;

            let movement_cost = distance * 0.08 + self.velocity.magnitude() * 0.03;
            let speed_penalty = if self.velocity.magnitude() > 5.0 {
                (self.velocity.magnitude() - 5.0) * 0.02
            } else { 0.0 };

            self.fatigue = (self.fatigue + movement_cost + speed_penalty).min(1.0);
            let recovery = (time_diff * 0.4).min(0.2);
            self.fatigue = (self.fatigue - recovery).max(0.0);
        }

        let busy_duration = match note_kind {
            NoteKind::Hold { end_time, .. } => end_time - time + 0.1, // Hold到结束时间
            NoteKind::Drag => 0.25,
            NoteKind::Flick => 0.2,
            NoteKind::Click => 0.15,
        };

        self.is_busy = true;
        self.busy_until = time + busy_duration;

        self.total_actions += 1;
        if success {
            self.success_streak += 1;
            self.confidence = (self.confidence + 0.005).min(1.0);
        } else {
            self.success_streak = 0;
            self.confidence = (self.confidence - 0.01).max(0.2);
        }

        let recent_window = 20.0_f32.min(self.total_actions as f32);
        let recent_success_rate = self.success_streak as f32 / recent_window;
        self.performance_score = recent_success_rate * 0.4 +
            self.confidence * 0.3 +
            (1.0 - self.fatigue) * 0.3;

        self.position = new_pos;
        self.last_time = time;
    }

    fn update_busy_status(&mut self, current_time: f32) {
        if current_time > self.busy_until {
            self.is_busy = false;
        }
    }

    fn calculate_assignment_score(&self, target_pos: Vector2, time: f32, note_difficulty: f32, current_time: f32, note_kind: &NoteKind, _note_duration: f32) -> f32 {
        let is_available = current_time > self.busy_until;
        let note_duration = match note_kind {
            NoteKind::Hold { end_time, .. } => end_time - time,
            NoteKind::Drag => 0.2,
            NoteKind::Flick => 0.15,
            NoteKind::Click => 0.1,
        };

        let distance = target_pos.distance_to(&self.position);
        let time_diff = time - self.last_time;
        let mut score = 1.0;

        if !is_available {
            let busy_penalty = (self.busy_until - current_time) * 10.0;
            score -= busy_penalty;
        }

        let position_weight = match self.finger.to_hand() {
            Hand::Left => {
                // 左手更适合处理左侧音符
                if target_pos.x < -0.1 {
                    0.4  // 强加分
                } else if target_pos.x > 0.1 {
                    -0.3 // 减分
                } else {
                    0.0
                }
            }
            Hand::Right => {
                // 右手更适合处理右侧音符
                if target_pos.x > 0.1 {
                    0.4  // 强加分
                } else if target_pos.x < -0.1 {
                    -0.3 // 减分
                } else {
                    0.0
                }
            }
        };
        score += position_weight;

        let distance_score = if distance < 0.2 {
            1.0 - distance * 2.0
        } else if distance < 0.5 {
            0.6 - (distance - 0.2) * 1.5
        } else {
            0.15 - (distance - 0.5).min(0.5) * 0.3
        };
        score *= distance_score;

        score *= 1.0 - self.fatigue * 0.3;

        if time_diff > 0.0 {
            let time_score = if time_diff < 0.08 {
                (time_diff / 0.08) * 0.5
            } else if time_diff < 0.15 {
                1.0
            } else if time_diff < 0.3 {
                0.9
            } else {
                1.1
            };
            score *= time_score;
        }

        if matches!(note_kind, NoteKind::Hold { .. }) {
            let hold_end_time = time + note_duration;
            if current_time < hold_end_time {
                // Hold音符会占用更长时间，需要更仔细的规划
                score *= 0.8; // 轻微降低Hold的优先级以避免长时间占用
            }
        }

        score *= 0.7 + self.performance_score * 0.3;
        let difficulty_factor = 1.0 - (note_difficulty - 1.0) * (1.0 - self.confidence) * 0.2;
        score *= difficulty_factor;

        score.max(0.0)
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepNeuralNetwork {
    layers: Vec<NetworkLayer>,
    #[serde(default)]
    learning_rate: f32,
    #[serde(default)]
    momentum: f32,
    #[serde(default)]
    dropout_rate: f32,
    batch_size: usize,
    epoch_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NetworkLayer {
    #[serde(default)]
    weights: Vec<Vec<f32>>,
    #[serde(default)]
    biases: Vec<f32>,
    #[serde(default)]
    activations: Vec<f32>,
    #[serde(default)]
    gradients: Vec<f32>,
    #[serde(default)]
    momentum_weights: Vec<Vec<f32>>,
    #[serde(default)]
    momentum_biases: Vec<f32>,
    layer_type: LayerType,
    activation_func: ActivationFunction, //wtfbro
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LayerType {
    Dense,
    LSTM,
    Attention,
    Residual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
}



impl DeepNeuralNetwork {
    pub fn clean(&mut self) {
        for layer in &mut self.layers {
            // 清理权重
            for weights in &mut layer.weights {
                for w in weights {
                    if !w.is_finite() {
                        *w = 0.0;
                    }
                }
            }

            // 清理偏置
            for bias in &mut layer.biases {
                if !bias.is_finite() {
                    *bias = 0.0;
                }
            }

            // 清理激活值
            for activation in &mut layer.activations {
                if !activation.is_finite() {
                    *activation = 0.0;
                }
            }

            // 清理梯度
            for gradient in &mut layer.gradients {
                if !gradient.is_finite() {
                    *gradient = 0.0;
                }
            }

            // 清理动量权重
            for momentum_row in &mut layer.momentum_weights {
                for momentum in momentum_row {
                    if !momentum.is_finite() {
                        *momentum = 0.0;
                    }
                }
            }

            // 清理动量偏置
            for momentum_bias in &mut layer.momentum_biases {
                if !momentum_bias.is_finite() {
                    *momentum_bias = 0.0;
                }
            }
        }

        // 清理学习率和其他参数
        if !self.learning_rate.is_finite() {
            self.learning_rate = 0.001;
        }
        if !self.momentum.is_finite() {
            self.momentum = 0.9;
        }
        if !self.dropout_rate.is_finite() {
            self.dropout_rate = 0.1;
        }
    }
    pub fn validate(&self) -> bool {
        // 检查所有层是否有效
        for layer in &self.layers {
            // 检查权重
            for weights in &layer.weights {
                for w in weights {
                    if !w.is_finite() {
                        return false;
                    }
                }
            }

            // 检查偏置
            for bias in &layer.biases {
                if !bias.is_finite() {
                    return false;
                }
            }

            // 检查其他浮点字段
            for activation in &layer.activations {
                if !activation.is_finite() {
                    return false;
                }
            }

            for gradient in &layer.gradients {
                if !gradient.is_finite() {
                    return false;
                }
            }

            for momentum_row in &layer.momentum_weights {
                for momentum in momentum_row {
                    if !momentum.is_finite() {
                        return false;
                    }
                }
            }

            for momentum_bias in &layer.momentum_biases {
                if !momentum_bias.is_finite() {
                    return false;
                }
            }
        }

        // 检查网络参数
        self.learning_rate.is_finite()
            && self.momentum.is_finite()
            && self.dropout_rate.is_finite()
    }
    pub fn activate(x: f32, func: &ActivationFunction) -> f32 {
        match func {
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::Swish => x * (1.0 / (1.0 + (-x).exp())),
            ActivationFunction::GELU => 0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh()),
        }
    }

    pub fn light_forward(&mut self, input: &[f32]) -> Vec<f32> {
        // 仅使用前两层进行轻量级推理
        let mut output = input.to_vec();

        // 第一层
        if self.layers.len() > 0 {
            output = Self::dense_forward(&mut self.layers[0], &output);
        }

        // 第二层
        if self.layers.len() > 1 {
            output = Self::dense_forward(&mut self.layers[1], &output);
        }

        output
    }

    fn new() -> Self {
        let mut network = Self {
            layers: Vec::new(),
            learning_rate: 0.001,
            momentum: 0.9,
            dropout_rate: 0.1,
            batch_size: 32,
            epoch_count: 0,
        };

        // 构建深度网络架构
        network.build_architecture();
        network
    }

    pub fn build_architecture(&mut self) {
        // 输入层: [position_x, position_y, time, velocity, fatigue, pattern_features...]

        // 第一层：特征提取层 (16 -> 128)
        self.add_dense_layer(16, 128, ActivationFunction::ReLU);

        // 第二层：自注意力层 (128 -> 128)
        self.add_attention_layer(128, 128);
        // 残差连接
        self.add_residual_layer(128, 128);

        // 第三层：LSTM时序层 (128 -> 256)
        self.add_lstm_layer(128, 256);

        // 第四层：残差连接层 (256 -> 256)
        self.add_residual_layer(256, 256);

        // 第五层：特征融合层 (256 -> 512)
        self.add_dense_layer(256, 512, ActivationFunction::Swish);

        // 第六层：决策层 (512 -> 256)
        self.add_dense_layer(512, 256, ActivationFunction::GELU);

        // 输出层：手部概率 (256 -> 4)
        self.add_dense_layer(256, 4, ActivationFunction::Sigmoid);
    }


    fn add_dense_layer(&mut self, input_size: usize, output_size: usize, activation: ActivationFunction) {
        let mut weights = Vec::new();
        let mut momentum_weights = Vec::new();

        // Xavier/Glorot 初始化
        let fan_avg = (input_size + output_size) as f32 / 2.0;
        let limit = (6.0 / fan_avg).sqrt();

        for _ in 0..output_size {
            let mut row = Vec::new();
            let mut momentum_row = Vec::new();
            for _ in 0..input_size {
                row.push((fastrand::f32() * 2.0 - 1.0) * limit);
                momentum_row.push(0.0);
            }
            weights.push(row);
            momentum_weights.push(momentum_row);
        }

        let layer = NetworkLayer {
            weights,
            biases: vec![0.0; output_size],
            activations: vec![0.0; output_size],
            gradients: vec![0.0; output_size],
            momentum_weights,
            momentum_biases: vec![0.0; output_size],
            layer_type: LayerType::Dense,
            activation_func: activation,
        };

        self.layers.push(layer);
    }

    fn add_lstm_layer(&mut self, input_size: usize, output_size: usize) {
        // LSTM实现 -> 别问我为什么这么少，简化
        let layer = NetworkLayer {
            weights: vec![vec![0.0; input_size]; output_size * 4], // 这里4个门
            biases: vec![0.0; output_size * 4],
            activations: vec![0.0; output_size],
            gradients: vec![0.0; output_size],
            momentum_weights: vec![vec![0.0; input_size]; output_size * 4],
            momentum_biases: vec![0.0; output_size * 4],
            layer_type: LayerType::LSTM,
            activation_func: ActivationFunction::Tanh, // LSTM通常使用Tanh激活
        };

        self.layers.push(layer);
    }

    fn add_attention_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = NetworkLayer {
            weights: vec![vec![0.0; input_size]; output_size * 3], // Q, K, V
            biases: vec![0.0; output_size],
            activations: vec![0.0; output_size],
            gradients: vec![0.0; output_size],
            momentum_weights: vec![vec![0.0; input_size]; output_size * 3],
            momentum_biases: vec![0.0; output_size],
            layer_type: LayerType::Attention,
            activation_func: ActivationFunction::ReLU, // 注意力层通常使用ReLU或Softmax
        };

        self.layers.push(layer);
    }

    fn add_residual_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = NetworkLayer {
            weights: vec![vec![0.0; input_size]; output_size],
            biases: vec![0.0; output_size],
            activations: vec![0.0; output_size],
            gradients: vec![0.0; output_size],
            momentum_weights: vec![vec![0.0; input_size]; output_size],
            momentum_biases: vec![0.0; output_size],
            layer_type: LayerType::Residual,
            activation_func: ActivationFunction::ReLU, // 残差连接通常使用ReLU
        };

        self.layers.push(layer);
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut current_input = input.to_vec();
        let mut layer_outputs = Vec::new();

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            match layer.layer_type {
                LayerType::Dense => {
                    current_input = Self::dense_forward(layer, &current_input);
                },
                LayerType::LSTM => {
                    current_input = Self::lstm_forward(layer, &current_input);
                },
                LayerType::Attention => {
                    current_input = Self::attention_forward(layer, &current_input);
                },
                LayerType::Residual => {
                    // 获取残差连接输入
                    let residual_input = layer_outputs
                        .get(layer_idx.saturating_sub(2)) // 安全索引
                        .unwrap_or(&current_input) // 回退到当前输入
                        .clone();

                    current_input = Self::residual_forward(
                        layer,
                        &current_input,
                        &residual_input
                    );
                },
            }
            layer_outputs.push(current_input.clone());
        }

        current_input
    }

    fn dense_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; layer.weights.len()];
        for (i, (weights, bias)) in layer.weights.iter().zip(layer.biases.iter()).enumerate() {
            let mut sum = *bias;
            for (w, x) in weights.iter().zip(input.iter()) {
                sum += w * x;
            }
            // 使用正确的激活函数
            output[i] = DeepNeuralNetwork::activate(sum, &layer.activation_func);
        }
        layer.activations = output.clone();
        output
    }

    // 3. 修正 LSTM 层前向传播
    fn lstm_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let output_size = layer.activations.len();
        let mut output = vec![0.0; output_size];
        for i in 0..output_size {
            let mut sum = 0.0;
            for (j, x) in input.iter().enumerate() {
                if j < layer.weights[i].len() {
                    sum += layer.weights[i][j] * x;
                }
            }
            sum += layer.biases[i];
            // 使用正确的激活函数
            output[i] = DeepNeuralNetwork::activate(sum, &layer.activation_func);
        }
        layer.activations = output.clone();
        output
    }

    fn attention_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        // 简化的自注意力机制
        let output_size = layer.activations.len();
        let mut output = vec![0.0; output_size];

        // 计算注意力权重
        let mut attention_weights = vec![0.0; input.len()];
        let mut attention_sum = 0.0;

        for i in 0..input.len() {
            let weight = (input[i] * input[i]).exp();
            attention_weights[i] = weight;
            attention_sum += weight;
        }

        // 归一化注意力权重
        for weight in &mut attention_weights {
            *weight /= attention_sum;
        }

        // 应用注意力
        for i in 0..output_size.min(input.len()) {
            output[i] = input[i] * attention_weights[i];
        }

        layer.activations = output.clone();
        output
    }

    fn residual_forward(
        layer: &mut NetworkLayer,
        input: &[f32],
        residual: &[f32]
    ) -> Vec<f32> {
        // 使用静态版本的全连接计算
        let dense_output = Self::dense_forward_static(
            &layer.weights,
            &layer.biases,
            input,
            &layer.activation_func // 传递激活函数
        );
        let mut output = vec![0.0; dense_output.len()];
        for i in 0..output.len() {
            output[i] = dense_output[i] + residual.get(i).unwrap_or(&0.0);
        }
        output
    }

    fn dense_forward_static(
        weights: &[Vec<f32>],
        biases: &[f32],
        input: &[f32],
        activation: &ActivationFunction // 添加激活函数参数
    ) -> Vec<f32> {
        let mut output = vec![0.0; weights.len()];
        for (i, (weight_row, bias)) in weights.iter().zip(biases.iter()).enumerate() {
            let mut sum = *bias;
            for (w, x) in weight_row.iter().zip(input.iter()) {
                sum += w * x;
            }
            // 使用传入的激活函数
            output[i] = DeepNeuralNetwork::activate(sum, activation);
        }
        output
    }

    fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) {
        let batch_count = (training_data.len() + self.batch_size - 1) / self.batch_size;

        for batch_idx in 0..batch_count {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = (start_idx + self.batch_size).min(training_data.len());
            let batch = &training_data[start_idx..end_idx];

            self.train_batch(batch);
        }

        self.epoch_count += 1;

        // 学习率衰减
        if self.epoch_count % 100 == 0 {
            self.learning_rate *= 0.95;
        }
    }

    fn train_batch(&mut self, batch: &[(Vec<f32>, Vec<f32>)]) {
        let mut total_gradients: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 0]; 0]; self.layers.len()];
        let mut total_bias_gradients: Vec<Vec<f32>> = vec![vec![0.0; 0]; self.layers.len()];

        // 初始化梯度累积器
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            total_gradients[layer_idx] = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            total_bias_gradients[layer_idx] = vec![0.0; layer.biases.len()];
        }

        // 批量前向和反向传播
        for (input, target) in batch {
            let output = self.forward(input);
            self.backward(&output, target, &mut total_gradients, &mut total_bias_gradients);
        }

        // 应用梯度
        self.apply_gradients(&total_gradients, &total_bias_gradients, batch.len());
    }

    fn backward(&mut self, output: &[f32], target: &[f32],
                total_gradients: &mut [Vec<Vec<f32>>],
                total_bias_gradients: &mut [Vec<f32>]) {
        // 简化的反向传播实现
        let mut layer_errors = vec![vec![0.0; 0]; self.layers.len()];

        // 计算输出层误差
        if let Some(last_layer_idx) = self.layers.len().checked_sub(1) {
            let output_errors: Vec<f32> = output.iter()
                .zip(target.iter())
                .map(|(o, t)| 2.0 * (o - t))
                .collect();
            layer_errors[last_layer_idx] = output_errors;
        }

        // 反向传播误差
        for layer_idx in (0..self.layers.len()).rev() {
            if layer_idx < self.layers.len() - 1 {
                // 计算当前层误差
                let next_layer = &self.layers[layer_idx + 1];
                let mut current_errors = vec![0.0; self.layers[layer_idx].activations.len()];

                for i in 0..current_errors.len() {
                    for (j, error) in layer_errors[layer_idx + 1].iter().enumerate() {
                        if j < next_layer.weights.len() && i < next_layer.weights[j].len() {
                            current_errors[i] += error * next_layer.weights[j][i];
                        }
                    }
                }
                layer_errors[layer_idx] = current_errors;
            }

            // 计算梯度
            self.calculate_layer_gradients(layer_idx, &layer_errors[layer_idx],
                                           &mut total_gradients[layer_idx],
                                           &mut total_bias_gradients[layer_idx]);
        }
    }

    fn calculate_layer_gradients(&self, layer_idx: usize, errors: &[f32],
                                 gradients: &mut [Vec<f32>], bias_gradients: &mut [f32]) {
        let layer = &self.layers[layer_idx];
        let prev_activations = if layer_idx > 0 {
            &self.layers[layer_idx - 1].activations
        } else {
            return; // 需要输入数据
        };

        for (i, error) in errors.iter().enumerate() {
            if i < gradients.len() {
                for (j, activation) in prev_activations.iter().enumerate() {
                    if j < gradients[i].len() {
                        gradients[i][j] += error * activation;
                    }
                }
                if i < bias_gradients.len() {
                    bias_gradients[i] += *error;
                }
            }
        }
    }

    fn apply_gradients(&mut self, gradients: &[Vec<Vec<f32>>], bias_gradients: &[Vec<f32>], batch_size: usize) {
        let batch_size_f = batch_size as f32;
        const MAX_GRAD: f32 = 5.0; // 梯度裁剪阈值

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            for (i, weight_row) in layer.weights.iter_mut().enumerate() {
                for (j, weight) in weight_row.iter_mut().enumerate() {
                    if i < gradients[layer_idx].len() && j < gradients[layer_idx][i].len() {
                        // 获取原始梯度值
                        let mut grad = gradients[layer_idx][i][j] / batch_size_f;

                        // 梯度裁剪 - 确保梯度不会爆炸
                        if grad > MAX_GRAD {
                            grad = MAX_GRAD;
                        } else if grad < -MAX_GRAD {
                            grad = -MAX_GRAD;
                        }

                        // 动量更新
                        layer.momentum_weights[i][j] =
                            self.momentum * layer.momentum_weights[i][j] - self.learning_rate * grad;
                        *weight += layer.momentum_weights[i][j];
                    }
                }
            }

            for (i, bias) in layer.biases.iter_mut().enumerate() {
                if i < bias_gradients[layer_idx].len() {
                    // 获取原始偏置梯度
                    let mut grad = bias_gradients[layer_idx][i] / batch_size_f;

                    // 梯度裁剪
                    if grad > MAX_GRAD {
                        grad = MAX_GRAD;
                    } else if grad < -MAX_GRAD {
                        grad = -MAX_GRAD;
                    }

                    layer.momentum_biases[i] =
                        self.momentum * layer.momentum_biases[i] - self.learning_rate * grad;
                    *bias += layer.momentum_biases[i];
                }
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdvancedFeatureExtractor {
    pattern_library: HashMap<String, PatternSignature>,
    temporal_patterns: VecDeque<TemporalFeature>,
    spatial_patterns: Vec<SpatialFeature>,
    difficulty_estimator: DifficultyEstimator,
    #[serde(skip)]  // 不需要序列化
    last_logged_bpm: Option<f32>,
    #[serde(skip)]
    last_logged_time: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternSignature {
    name: String,
    #[serde(default)]
    features: Vec<f32>,
    #[serde(default)]
    difficulty_multiplier: f32,
    optimal_strategy: HandStrategy,
    #[serde(default)]
    success_rate: f32,
    adaptation_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TemporalFeature {
    #[serde(default)]
    time_window: f32,
    #[serde(default)]
    note_density: f32,
    #[serde(default)]
    velocity_profile: Vec<f32>,
    #[serde(default)]
    rhythm_complexity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SpatialFeature {
    position_cluster: Vector2,
    #[serde(default)]
    spread_radius: f32,
    note_count: usize,
    dominant_direction: Vector2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum HandStrategy {
    Alternating,
    SameHand,
    Optimal,
    Stream,
    Chord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DifficultyEstimator {
    #[serde(default)]
    base_difficulty: f32,
    #[serde(default)]
    pattern_difficulty: HashMap<String, f32>,
    #[serde(default)]
    speed_difficulty: f32,
    #[serde(default)]
    coordination_difficulty: f32,
}

impl AdvancedFeatureExtractor {
    fn new() -> Self {
        Self {
            pattern_library: HashMap::new(),
            temporal_patterns: VecDeque::with_capacity(100),
            spatial_patterns: Vec::new(),
            difficulty_estimator: DifficultyEstimator {
                base_difficulty: 1.0,
                pattern_difficulty: HashMap::new(),
                speed_difficulty: 1.0,
                coordination_difficulty: 1.0,
            },
            // 初始化日志输出
            last_logged_bpm: None,
            last_logged_time: -1.0,
        }
    }

    fn extract_features(&mut self, notes: &[ProcessedNote], window_size: usize, bpm_list: &mut BpmList) -> Vec<f32> {
        let mut features = Vec::new();

        // 基础位置特征
        features.extend(self.extract_position_features(notes));

        // 时序特征
        features.extend(self.extract_temporal_features(notes, window_size, bpm_list));

        // 模式特征
        features.extend(self.extract_pattern_features(notes));

        // 难度特征
        features.extend(self.extract_difficulty_features(notes));

        // 速度和加速度特征
        features.extend(self.extract_velocity_features(notes));

        // 空间分布特征
        features.extend(self.extract_spatial_features(notes));

        features
    }

    fn extract_position_features(&self, notes: &[ProcessedNote]) -> Vec<f32> {
        if notes.is_empty() {
            return vec![0.0; 4];
        }

        let current = &notes[0];
        vec![
            current.position.x,
            current.position.y,
            current.position.magnitude(),
            current.position.x.atan2(current.position.y),
        ]
    }

    fn extract_temporal_features(&mut self, notes: &[ProcessedNote], window_size: usize, bpm_list: &mut BpmList) -> Vec<f32> {
        if notes.is_empty() {
            return vec![0.0; 6];
        }

        let window_end = window_size.min(notes.len());
        let window = &notes[0..window_end];

        // 计算时间间隔统计
        let mut intervals = Vec::new();
        for i in 1..window.len() {
            intervals.push(window[i].time - window[i-1].time);
        }

        let avg_interval = if !intervals.is_empty() {
            intervals.iter().sum::<f32>() / intervals.len() as f32
        } else {
            0.0
        };

        let interval_variance = if intervals.len() > 1 {
            let mean = avg_interval;
            intervals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / intervals.len() as f32
        } else {
            0.0
        };

        // 节奏复杂度
        let rhythm_complexity = self.calculate_rhythm_complexity(&intervals);

        // 直接使用BPM列表中的BPM值
        let current_time = notes[0].time;
        let current_bpm = bpm_list.now_bpm(current_time);

        // === BPM 日志输出 ===
        // 每秒输出一次或当 BPM 变化时输出
        let should_log = self.last_logged_bpm.is_none() ||
            current_bpm != self.last_logged_bpm.unwrap() ||
            current_time - self.last_logged_time > 1.0;

        if should_log {
            println!("[BPM LOG] Time: {:.2}s, BPM: {:.1}", current_time, current_bpm);
            self.last_logged_bpm = Some(current_bpm);
            self.last_logged_time = current_time;
        }
        // === 日志输出结束 ===

        vec![
            avg_interval,
            interval_variance,
            rhythm_complexity,
            current_bpm,  // 直接使用BPM列表中的值
            intervals.len() as f32,
            window.len() as f32 / (window.last().unwrap().time - window[0].time + 0.001),
        ]
    }

    fn calculate_rhythm_complexity(&self, intervals: &[f32]) -> f32 {
        if intervals.len() < 3 {
            return 0.0;
        }

        let mut complexity = 0.0;
        for i in 1..intervals.len() {
            let ratio = intervals[i] / intervals[i-1].max(0.001);
            complexity += (ratio.ln().abs()).min(2.0);
        }

        complexity / intervals.len() as f32
    }

    fn extract_pattern_features(&mut self, notes: &[ProcessedNote]) -> Vec<f32> {
        // 识别常见模式
        let alternating_score = self.detect_alternating_pattern(notes);
        let stream_score = self.detect_stream_pattern(notes);
        let chord_score = self.detect_chord_pattern(notes);
        let jack_score = self.detect_jack_pattern(notes);

        vec![alternating_score, stream_score, chord_score, jack_score]
    }

    fn detect_alternating_pattern(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 4 {
            return 0.0;
        }

        let mut alternating_count = 0;
        for i in 2..notes.len() {
            let dir1 = (notes[i-1].position.x - notes[i-2].position.x).signum();
            let dir2 = (notes[i].position.x - notes[i-1].position.x).signum();

            if dir1 * dir2 < 0.0 {
                alternating_count += 1;
            }
        }

        alternating_count as f32 / (notes.len() - 2) as f32
    }

    fn detect_stream_pattern(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 5 {
            return 0.0;
        }

        let mut stream_score = 0.0;
        let mut consistent_direction = 0;

        for i in 1..notes.len() {
            let time_diff = notes[i].time - notes[i-1].time;
            let distance = notes[i].position.distance_to(&notes[i-1].position);

            // 快速、密集的音符序列
            if time_diff < 0.2 && distance < 0.4 {
                stream_score += 1.0;
            }

            // 检查方向一致性
            if i >= 2 {
                let dir1 = notes[i-1].position - notes[i-2].position;
                let dir2 = notes[i].position - notes[i-1].position;

                if dir1.normalize().dot(&dir2.normalize()) > 0.5 {
                    consistent_direction += 1;
                }
            }
        }

        let direction_consistency = if notes.len() > 2 {
            consistent_direction as f32 / (notes.len() - 2) as f32
        } else {
            0.0
        };

        (stream_score / notes.len() as f32) * 0.7 + direction_consistency * 0.3
    }

    fn detect_chord_pattern(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 2 {
            return 0.0;
        }

        let mut chord_score = 0.0;
        let mut i = 0;

        while i < notes.len() {
            let mut simultaneous_count = 1;
            let base_time = notes[i].time;

            let mut j = i + 1;
            while j < notes.len() && (notes[j].time - base_time).abs() < 0.05 {
                simultaneous_count += 1;
                j += 1;
            }

            if simultaneous_count >= 2 {
                chord_score += simultaneous_count as f32;
            }

            i = j;
        }

        chord_score / notes.len() as f32
    }

    fn detect_jack_pattern(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 3 {
            return 0.0;
        }

        let mut jack_score = 0.0;

        for i in 2..notes.len() {
            let pos1 = notes[i-2].position;
            let pos2 = notes[i-1].position;
            let pos3 = notes[i].position;

            // 检查位置是否在小范围内重复
            if pos1.distance_to(&pos2) < 0.15 && pos2.distance_to(&pos3) < 0.15 {
                let time_consistency = (notes[i-1].time - notes[i-2].time - (notes[i].time - notes[i-1].time)).abs() < 0.05;
                if time_consistency {
                    jack_score += 1.0;
                }
            }
        }

        jack_score / (notes.len() - 2) as f32
    }

    fn extract_difficulty_features(&mut self, notes: &[ProcessedNote]) -> Vec<f32> {
        let base_difficulty = self.calculate_base_difficulty(notes);
        let speed_difficulty = self.calculate_speed_difficulty(notes);
        let coordination_difficulty = self.calculate_coordination_difficulty(notes);
        let pattern_difficulty = self.calculate_pattern_difficulty(notes);

        vec![base_difficulty, speed_difficulty, coordination_difficulty, pattern_difficulty]
    }

    fn calculate_base_difficulty(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.is_empty() {
            return 0.0;
        }

        let mut difficulty = 0.0;

        for note in notes {
            difficulty += match note.kind {
                NoteKind::Click => 1.0,
                NoteKind::Drag => 1.3,
                NoteKind::Flick => 1.5,
                NoteKind::Hold { .. } => 1.8,
            };
        }

        difficulty / notes.len() as f32
    }

    fn calculate_speed_difficulty(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 2 {
            return 0.0;
        }

        let mut max_speed: f32 = 0.0;
        let mut avg_speed = 0.0;

        for i in 1..notes.len() {
            let distance = notes[i].position.distance_to(&notes[i-1].position);
            let time_diff = notes[i].time - notes[i-1].time;

            if time_diff > 0.001 {
                let speed = distance / time_diff;
                max_speed = max_speed.max(speed);
                avg_speed += speed;
            }
        }

        avg_speed /= (notes.len() - 1) as f32;

        // 组合最大速度和平均速度
        max_speed * 0.6 + avg_speed * 0.4
    }

    fn calculate_coordination_difficulty(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 3 {
            return 0.0;
        }

        let mut coordination_score = 0.0;

        // 检查需要双手协调的复杂模式
        for i in 2..notes.len() {
            let pos_changes = vec![
                notes[i-2].position - notes[i-1].position,
                notes[i-1].position - notes[i].position,
            ];

            // 计算方向变化的复杂度
            if pos_changes[0].magnitude() > 0.01 && pos_changes[1].magnitude() > 0.01 {
                let angle_change = pos_changes[0].normalize().dot(&pos_changes[1].normalize());
                let direction_complexity = (1.0 - angle_change.abs()).max(0.0);

                coordination_score += direction_complexity;
            }
        }

        coordination_score / (notes.len() - 2) as f32
    }

    fn calculate_pattern_difficulty(&self, notes: &[ProcessedNote]) -> f32 {
        let alternating = self.detect_alternating_pattern(notes);
        let stream = self.detect_stream_pattern(notes);
        let chord = self.detect_chord_pattern(notes);
        let jack = self.detect_jack_pattern(notes);

        // 不同模式有不同的难度权重
        alternating * 1.2 + stream * 2.5 + chord * 1.8 + jack * 3.0
    }

    fn extract_velocity_features(&self, notes: &[ProcessedNote]) -> Vec<f32> {
        if notes.len() < 2 {
            return vec![0.0; 4];
        }

        let mut velocities = Vec::new();
        let mut accelerations = Vec::new();

        for i in 1..notes.len() {
            let distance = notes[i].position.distance_to(&notes[i-1].position);
            let time_diff = notes[i].time - notes[i-1].time;

            if time_diff > 0.001 {
                let velocity = distance / time_diff;
                velocities.push(velocity);

                if i > 1 && velocities.len() >= 2 {
                    let prev_velocity = velocities[velocities.len() - 2];
                    let acceleration = (velocity - prev_velocity) / time_diff;
                    accelerations.push(acceleration);
                }
            }
        }

        let avg_velocity = if !velocities.is_empty() {
            velocities.iter().sum::<f32>() / velocities.len() as f32
        } else { 0.0 };

        let max_velocity = velocities.iter().fold(0.0f32, |a, &b| a.max(b));

        let avg_acceleration = if !accelerations.is_empty() {
            accelerations.iter().sum::<f32>() / accelerations.len() as f32
        } else { 0.0 };

        let max_acceleration = accelerations.iter().fold(0.0f32, |a, &b| a.max(b.abs()));

        vec![avg_velocity, max_velocity, avg_acceleration, max_acceleration]
    }

    fn extract_spatial_features(&mut self, notes: &[ProcessedNote]) -> Vec<f32> {
        if notes.is_empty() {
            return vec![0.0; 6];
        }

        // 计算空间分布
        let mut x_positions: Vec<f32> = notes.iter().map(|n| n.position.x).collect();
        let mut y_positions: Vec<f32> = notes.iter().map(|n| n.position.y).collect();

        x_positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_positions.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let x_range = x_positions.last().unwrap() - x_positions.first().unwrap();
        let y_range = y_positions.last().unwrap() - y_positions.first().unwrap();

        let center_x = x_positions.iter().sum::<f32>() / x_positions.len() as f32;
        let center_y = y_positions.iter().sum::<f32>() / y_positions.len() as f32;

        // 计算分散度
        let dispersion = notes.iter().map(|n| {
            ((n.position.x - center_x).powi(2) + (n.position.y - center_y).powi(2)).sqrt()
        }).sum::<f32>() / notes.len() as f32;

        // 计算主要运动方向
        let mut direction_vector = Vector2::new(0.0, 0.0);
        for i in 1..notes.len() {
            direction_vector = direction_vector + (notes[i].position - notes[i-1].position);
        }

        vec![x_range, y_range, center_x, center_y, dispersion, direction_vector.magnitude()]
    }
}

// === 处理后的音符结构 ===

#[derive(Debug, Clone)]
struct ProcessedNote {
    index: usize,
    position: Vector2,
    //#[serde(default)]
    time: f32,
    kind: NoteKind,
    assigned_hand: Option<Hand>,
    //#[serde(default)]
    confidence: f32,
    //#[serde(default)]
    features: Vec<f32>,
    judge: JudgeStatus,  // Add this field
}

// === 经验回放系统 ===

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperienceReplay {
    experiences: VecDeque<Experience>,
    capacity: usize,
    current_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Experience {
    state: Vec<f32>,
    action: usize, // 0: left hand, 1: right hand
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
    timestamp: f32,
}

impl ExperienceReplay {
    fn new(capacity: usize) -> Self {
        Self {
            experiences: VecDeque::with_capacity(capacity),
            capacity,
            current_size: 0,
        }
    }

    fn push(&mut self, experience: Experience) {
        if self.experiences.len() >= self.capacity {
            self.experiences.pop_front();
        } else {
            self.current_size += 1;
        }
        self.experiences.push_back(experience);
    }

    fn sample(&self, batch_size: usize) -> Vec<Experience> {
        let mut samples = Vec::new();
        let sample_count = batch_size.min(self.current_size);

        for _ in 0..sample_count {
            let idx = fastrand::usize(0..self.current_size);
            if let Some(exp) = self.experiences.get(idx) {
                samples.push(exp.clone());
            }
        }

        samples
    }

    fn len(&self) -> usize {
        self.current_size
    }
}

#[derive(Serialize, Deserialize)]
struct PhiTKAdvancedAI {
    // 神经网络
    main_network: DeepNeuralNetwork, //主的
    target_network: DeepNeuralNetwork,// 目标网络

    feature_extractor: AdvancedFeatureExtractor,// 特征提取用的

    // 经验回放 or 学习
    experience_replay: ExperienceReplay,

    // 手部状态跟踪
    left_hand_state: HandState,
    right_hand_state: HandState,

    // 配置参数
    #[serde(default)]
    rotation: f32,
    #[serde(default)]
    exploration_rate: f32,
    #[serde(default)]
    discount_factor: f32,
    target_update_frequency: u64,

    // 性能统计
    total_notes_processed: u64,
    correct_predictions: u64,
    training_episodes: u64,
    average_reward: f32,

    // 自适应参数
    #[serde(default)]
    difficulty_adaptation: f32,
    #[serde(default)]
    learning_momentum: f32,
    #[serde(default)]
    confidence_threshold: f32,
    version: u32,
    pub last_save_episodes: usize,

    // 长期记忆
    pattern_memory: BTreeMap<String, PatternMemory>,
    performance_history: VecDeque<PerformanceMetrics>,
    #[serde(default)]
    last_update_time: f32,

    game_mode: GameMode,
    finger_states: Vec<FingerState>,

    // 稳定性相关
    #[serde(default)]
    stability_factor: f32,
    #[serde(default)]
    hand_switch_penalty: f32,
    #[serde(default)]
    consistency_bonus: f32,

    // 学习能力增强
    #[serde(default)]
    adaptive_learning_rate: f32,
    #[serde(default)]
    pattern_recognition_strength: f32,
    #[serde(default)]
    memory_consolidation_rate: f32,

    // 性能跟踪
    recent_assignments: VecDeque<(Hand, f32, f32)>,
    hand_switch_count: u32,
    last_assigned_hand: Option<Hand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PatternMemory {
    pattern_id: String,
    success_count: u32,
    failure_count: u32,
    #[serde(default)]
    average_difficulty: f32,
    optimal_strategy: HandStrategy,
    #[serde(default)]
    last_seen: f32,
    #[serde(default)]
    adaptation_history: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceMetrics {
    #[serde(default)]
    timestamp: f32,
    #[serde(default)]
    accuracy: f32,
    #[serde(default)]
    speed: f32,
    #[serde(default)]
    consistency: f32,
    #[serde(default)]
    difficulty_handled: f32,
    patterns_recognized: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HandState {
    hand: Hand,
    position: Vector2,
    velocity: Vector2,
    #[serde(default)]
    last_time: f32,
    #[serde(default)]
    fatigue: f32,
    #[serde(default)]
    confidence: f32,
    success_streak: u32,
    total_actions: u32,
    #[serde(default)]
    performance_score: f32,
}

impl HandState {
    fn new(hand: Hand, initial_pos: Vector2) -> Self {
        Self {
            hand,
            position: initial_pos,
            velocity: Vector2::new(0.0, 0.0),
            last_time: -1.0,
            fatigue: 0.0,
            confidence: 1.0,
            success_streak: 0,
            total_actions: 0,
            performance_score: 1.0,
        }
    }

    pub fn clean(&mut self) {
        self.position.clean();
        self.velocity.clean();
        if !self.last_time.is_finite() {
            self.last_time = -1.0;
        }
        if !self.fatigue.is_finite() {
            self.fatigue = 0.0;
        }
        if !self.confidence.is_finite() {
            self.confidence = 1.0;
        }
        if !self.performance_score.is_finite() {
            self.performance_score = 1.0;
        }
    }

    fn validate(&self) -> bool {
        self.position.x.is_finite() &&
            self.position.y.is_finite() &&
            self.velocity.x.is_finite() &&
            self.velocity.y.is_finite() &&
            self.last_time.is_finite() &&
            self.fatigue.is_finite() &&
            self.confidence.is_finite() &&
            self.performance_score.is_finite()
    }

    fn update_state(&mut self, new_pos: Vector2, time: f32, success: bool, note_kind: &NoteKind) {
        let time_diff = time - self.last_time;

        if time_diff > 0.001 {
            let distance = new_pos.distance_to(&self.position);

            // 更新速度（平滑化）
            let new_velocity = (new_pos - self.position) * (1.0 / time_diff);
            self.velocity = self.velocity * 0.7 + new_velocity * 0.3;

            // 更新疲劳度，根据音符类型调整
            let note_complexity_factor = match note_kind {
                NoteKind::Click => 1.0,
                NoteKind::Drag => 1.2,
                NoteKind::Flick => 1.3,
                NoteKind::Hold { .. } => 1.5,
            };

            let movement_cost = distance * 0.1 * note_complexity_factor + self.velocity.magnitude() * 0.05;
            self.fatigue = (self.fatigue + movement_cost).min(1.0);

            // 自然恢复
            let recovery = (time_diff * 0.3).min(0.15);
            self.fatigue = (self.fatigue - recovery).max(0.0);
        }

        // 更新成功统计
        self.total_actions += 1;
        if success {
            self.success_streak += 1;
            self.confidence = (self.confidence + 0.01).min(1.0);
        } else {
            self.success_streak = 0;
            self.confidence = (self.confidence - 0.02).max(0.1);
        }

        // 更新性能评分
        let recent_success_rate = if self.total_actions > 10 {
            self.success_streak as f32 / 10.0_f32.min(self.total_actions as f32)
        } else {
            self.success_streak as f32 / self.total_actions.max(1) as f32
        };

        self.performance_score = recent_success_rate * 0.5 + self.confidence * 0.3 + (1.0 - self.fatigue) * 0.2;

        self.position = new_pos;
        self.last_time = time;
    }
/*
    fn calculate_assignment_score(&self, target_pos: Vector2, time: f32, note_difficulty: f32) -> f32 {
        let distance = target_pos.distance_to(&self.position);
        let time_diff = time - self.last_time;

        let mut score = 1.0;

        let position_weight = match self.hand {
            Hand::Left => {
                // 左手更适合处理左侧音符
                if target_pos.x < -0.1 {
                    0.4
                } else if target_pos.x > 0.1 {
                    -0.3
                } else {
                    0.0
                }
            }
            Hand::Right => {
                // 右手更适合处理右侧音符
                if target_pos.x > 0.1 {
                    0.4
                } else if target_pos.x < -0.1 {
                    -0.3
                } else {
                    0.0
                }
            }
        };
        score += position_weight;

        // 距离惩罚（非线性）
        score -= (distance * distance) * 2.0;

        // 疲劳惩罚
        score -= self.fatigue * 1.5;

        // 时间间隔奖励/惩罚
        if time_diff > 0.0 {
            if time_diff < 0.1 {
                score -= (0.1 - time_diff) * 5.0; // 太快的连击惩罚
            } else if time_diff > 0.8 {
                score += 0.2; // 有足够休息时间的奖励
            }
        }

        // 性能加成
        score += self.performance_score * 0.5;

        // 难度适应
        score -= note_difficulty * (1.0 - self.confidence) * 0.3;

        // 速度考虑
        if time_diff > 0.001 {
            let required_speed = distance / time_diff;
            if required_speed > 6.0 {
                score -= (required_speed - 6.0) * 0.5;
            }
        }

        score
    }

 */
}

impl PhiTKAdvancedAI {
    pub fn reset_for_new_chart(&mut self) {
        let rad = self.rotation.to_radians();

        // 重置手部状态
        self.finger_states = Self::init_finger_states(self.game_mode, self.rotation.to_radians());

        // 清空性能跟踪
        self.recent_assignments.clear();
        self.hand_switch_count = 0;
        self.last_assigned_hand = None;
        self.left_hand_state = HandState::new(Hand::Left, Vector2::new(-0.3, 0.0).rotate(rad));
        self.right_hand_state = HandState::new(Hand::Right, Vector2::new(0.3, 0.0).rotate(rad));

        // 清空特征提取器的临时数据
        self.feature_extractor.temporal_patterns.clear();
        self.feature_extractor.spatial_patterns.clear();

        // 清空经验回放
        self.experience_replay = ExperienceReplay::new(50000);

        // 重置性能历史
        self.performance_history.clear();

    }
    const CURRENT_VERSION: u32 = 1;
    fn clean_model_data(&mut self) {
        // 清理神经网络
        self.main_network.clean();
        self.target_network.clean();

        // 清理手部状态
        self.left_hand_state.clean();
        self.right_hand_state.clean();

        // 清理其他浮点字段
        let default = Self::new(self.rotation);
        if !self.exploration_rate.is_finite() {
            self.exploration_rate = default.exploration_rate;
        }
        if !self.discount_factor.is_finite() {
            self.discount_factor = default.discount_factor;
        }
        if !self.average_reward.is_finite() {
            self.average_reward = default.average_reward;
        }
        if !self.difficulty_adaptation.is_finite() {
            self.difficulty_adaptation = default.difficulty_adaptation;
        }
        if !self.learning_momentum.is_finite() {
            self.learning_momentum = default.learning_momentum;
        }
        if !self.confidence_threshold.is_finite() {
            self.confidence_threshold = default.confidence_threshold;
        }
        // 清理手指状态
        for finger_state in &mut self.finger_states {
            finger_state.clean();
        }

        // 清理新增字段
        if !self.stability_factor.is_finite() {
            self.stability_factor = 0.85;
        }
        if !self.hand_switch_penalty.is_finite() {
            self.hand_switch_penalty = 0.4;
        }
        if !self.consistency_bonus.is_finite() {
            self.consistency_bonus = 0.3;
        }
        if !self.adaptive_learning_rate.is_finite() {
            self.adaptive_learning_rate = 0.001;
        }
        if !self.pattern_recognition_strength.is_finite() {
            self.pattern_recognition_strength = 1.0;
        }
        if !self.memory_consolidation_rate.is_finite() {
            self.memory_consolidation_rate = 0.1;
        }

        // 清理特征提取器中的浮点字段
        self.feature_extractor.difficulty_estimator.base_difficulty =
            self.feature_extractor.difficulty_estimator.base_difficulty.max(0.0).min(10.0);
    }
    fn validate_for_serialization(&self) -> bool {
        // 检查所有浮点字段
        let float_fields_valid = [
            self.exploration_rate,
            self.discount_factor,
            self.average_reward,
            self.difficulty_adaptation,
            self.learning_momentum,
            self.confidence_threshold
        ].iter().all(|f| f.is_finite());

        // 检查神经网络
        let networks_valid = self.main_network.validate() && self.target_network.validate();

        // 检查手部状态
        let hands_valid = self.left_hand_state.validate() && self.right_hand_state.validate();

        float_fields_valid && networks_valid && hands_valid
    }
    fn new(rotation: f32) -> Self {
        let rad = rotation.to_radians();
        let game_mode = GameMode::TwoFinger;
        let finger_states = Self::init_finger_states(game_mode, rad);


        let mut ai = Self {
            main_network: DeepNeuralNetwork::new(),
            target_network: DeepNeuralNetwork::new(),
            feature_extractor: AdvancedFeatureExtractor::new(),
            experience_replay: ExperienceReplay::new(10000),
            left_hand_state: HandState::new(Hand::Left, Vector2::new(-0.3, 0.0).rotate(rad)),
            right_hand_state: HandState::new(Hand::Right, Vector2::new(0.3, 0.0).rotate(rad)),
            rotation,
            exploration_rate: 0.05,
            discount_factor: 0.95,
            target_update_frequency: 1000, // 每1000个训练回合更新一次目标网络
            total_notes_processed: 0,
            correct_predictions: 0,
            training_episodes: 0,
            average_reward: 0.0,
            difficulty_adaptation: 1.0,
            learning_momentum: 0.9,
            confidence_threshold: 0.7,
            pattern_memory: BTreeMap::new(),
            performance_history: VecDeque::with_capacity(50000),
            version: Self::CURRENT_VERSION,
            last_save_episodes: 0,
            //self.last_update_time = -1.0;
            last_update_time: -1.0,

            game_mode,
            finger_states,
            stability_factor: 0.85,
            hand_switch_penalty: 0.4,
            consistency_bonus: 0.3,
            adaptive_learning_rate: 0.001,
            pattern_recognition_strength: 1.0,
            memory_consolidation_rate: 0.1,
            recent_assignments: VecDeque::with_capacity(50),
            hand_switch_count: 0,
            last_assigned_hand: None,
        };

        // 初始化目标网络权重
        ai.target_network = ai.main_network.clone();

        ai
    }


    fn load_or_create(filepath: &str, rotation: f32) -> Self {
        let path = Path::new(filepath);
        println!("尝试从 {} 加载模型...", path.display());

        // 如果文件存在且大小>0，就读它 111111
        if let Ok(meta) = fs::metadata(path) {
            if meta.len() > 0 {
                if let Ok(bytes) = fs::read(path) {
                    if let Ok(mut ai) = bincode::deserialize::<Self>(&bytes) {
                        if ai.validate_for_serialization() {
                            println!("成功加载已有模型，训练回合: {}", ai.training_episodes);
                            ai.rotation = rotation;
                            ai.update_hand_positions();
                            return ai;
                        }
                    }
                }
                eprintln!("模型无效，创建新模型");
            }
        }

        // 文件不存在，或为空，或读取失败 → 新建并立即保存一个初始模型
        let mut ai = Self::new(rotation);
        // 这里调用一次保存，保证后续 load 都能读到一个合法文件
        ai.save_model(filepath);
        ai
    }


    fn save_model(&mut self, filepath: &str) {
        self.clean_model_data();
        if !self.validate_for_serialization() {
            eprintln!("警告: 模型包含无效数据，无法保存");
            return;
        }

        let path = Path::new(filepath);
        match bincode::serialize(self) {
            Ok(data) => {
                if let Err(e) = fs::write(path, data) {
                    eprintln!("写入文件失败: {}", e);
                } else {
                    println!("模型成功保存，训练回合: {}", self.training_episodes);
                }
            }
            Err(e) => {
                eprintln!("模型序列化失败: {}", e);
                self.diagnose_serialization_issue();
            }
        }
    }

    fn diagnose_serialization_issue(&self) {
        eprintln!("开始诊断序列化问题...");

        // 检查网络层
        for (i, layer) in self.main_network.layers.iter().enumerate() {
            // 检查权重
            for (j, weights) in layer.weights.iter().enumerate() {
                for (k, w) in weights.iter().enumerate() {
                    if !w.is_finite() {
                        eprintln!("无效权重: 层 {} 权重[{}][{}] = {}", i, j, k, w);
                    }
                }
            }

            // 检查偏置
            for (j, bias) in layer.biases.iter().enumerate() {
                if !bias.is_finite() {
                    eprintln!("无效偏置: 层 {} 偏置[{}] = {}", i, j, bias);
                }
            }
        }

        eprintln!("诊断完成");
    }

    fn init_finger_states(mode: GameMode, rotation_rad: f32) -> Vec<FingerState> {
        let mut states = Vec::new();

        match mode {
            GameMode::TwoFinger => {
                states.push(FingerState::new(
                    Finger::LeftIndex,
                    Vector2::new(-0.3, 0.0).rotate(rotation_rad)
                ));
                states.push(FingerState::new(
                    Finger::RightIndex,
                    Vector2::new(0.3, 0.0).rotate(rotation_rad)
                ));
            }
            GameMode::FourFinger => {
                states.push(FingerState::new(
                    Finger::LeftIndex,
                    Vector2::new(-0.4, 0.0).rotate(rotation_rad)
                ));
                states.push(FingerState::new(
                    Finger::LeftMiddle,
                    Vector2::new(-0.2, 0.0).rotate(rotation_rad)
                ));
                states.push(FingerState::new(
                    Finger::RightIndex,
                    Vector2::new(0.2, 0.0).rotate(rotation_rad)
                ));
                states.push(FingerState::new(
                    Finger::RightMiddle,
                    Vector2::new(0.4, 0.0).rotate(rotation_rad)
                ));
            }
        }

        states
    }
/*
    fn select_optimal_mode(&mut self, notes: &[ProcessedNote]) -> GameMode {
        if notes.len() < 10 {
            return self.game_mode;
        }

        let mut simultaneous_count = 0;
        let mut wide_spread_count = 0;
        let mut high_speed_count = 0;

        let mut time_groups: std::collections::HashMap<i32, Vec<usize>> = std::collections::HashMap::new();

        for (i, note) in notes.iter().enumerate() {
            let time_key = (note.time * 20.0).round() as i32;
            time_groups.entry(time_key).or_insert_with(Vec::new).push(i);
        }

        for group in time_groups.values() {
            if group.len() > 1 {
                simultaneous_count += 1;

                let positions: Vec<f32> = group.iter().map(|&i| notes[i].position.x).collect();
                let min_x = positions.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max_x = positions.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                if max_x - min_x > 0.6 {
                    wide_spread_count += 1;
                }
            }
        }

        for i in 1..notes.len().min(20) {
            let time_diff = notes[i].time - notes[i-1].time;
            if time_diff < 0.1 && time_diff > 0.001 {
                high_speed_count += 1;
            }
        }

        let complexity_score = simultaneous_count as f32 * 2.0 +
            wide_spread_count as f32 * 3.0 +
            high_speed_count as f32 * 1.5;

        if complexity_score > 15.0 {
            GameMode::FourFinger
        } else {
            GameMode::TwoFinger
        }
    }


 */
    fn update_hand_positions(&mut self) {
        let rad = self.rotation.to_radians();
        self.left_hand_state.position = Vector2::new(-0.3, 0.0).rotate(rad);
        self.right_hand_state.position = Vector2::new(0.3, 0.0).rotate(rad);
    }

    fn light_update_hand_states(&mut self, notes: &[Note]) {
        // 仅更新手部位置，不进行完整分析
        let rad = self.rotation.to_radians();

        for note in notes {
            let pos = Vector2::new(
                note.object.translation.0.now(),
                note.object.translation.1.now()
            ).rotate(rad);

            match note.hand {
                Hand::Left => self.left_hand_state.position = pos,
                Hand::Right => self.right_hand_state.position = pos,
                //_ => {} // 未分配的不处理
            }
        }
    }

    fn analyze_and_assign(&mut self, notes: &mut [Note], _config: &Config, bpm_list: &BpmList) {
        if notes.is_empty() {
            return;
        }

        /*

        let current_time = notes[0].time;

        if current_time - self.last_light_update < 0.02 {
            self.light_update_hand_states(notes);
            return;
        }

        self.last_light_update = current_time;

         */

        // 预处理音符
        let mut processed_notes = self.preprocess_notes(notes);

        let simultaneous_groups = self.detect_simultaneous_groups(&processed_notes);

        let mut bpm_list_ccb: BpmList = (*bpm_list).clone(); //wtfbro
        self.assign_simultaneous_groups(&mut processed_notes, &simultaneous_groups, &mut bpm_list_ccb);
        // 使用AI处理剩余音符, 然后将BPM列表传递给单音符分配
        self.ai_assign_single_notes(&mut processed_notes, &simultaneous_groups, &mut bpm_list_ccb);

        // 后处理优化
        self.post_process_assignments(&mut processed_notes);

        // 应用结果并收集训练数据
        self.apply_and_learn(notes, &processed_notes);

        // 定期训练网络
        if self.experience_replay.len() >= 64 && self.total_notes_processed % 100 == 0 {
            println!("触发训练: 经验={}, 音符={}",
                     self.experience_replay.len(),
                     self.total_notes_processed
            );
            self.train_network();
        }

        // 更新目标网络
        if self.training_episodes % self.target_update_frequency == 0 {
            self.target_network = self.main_network.clone();
        }

        self.training_episodes += 1;
        self.last_save_episodes += 1;

        // 50 次训练保存一次
        if self.last_save_episodes >= 10000 {
            self.save_model("phitk_ai_model.bin");
            self.last_save_episodes = 0;
        }
    }

    fn preprocess_notes(&self, notes: &[Note]) -> Vec<ProcessedNote> {
        let rad = self.rotation.to_radians();

        notes.iter().enumerate().map(|(i, note)| {
            let original_pos = Vector2::new(
                note.object.translation.0.now(),
                note.object.translation.1.now()
            );
            let rotated_pos = original_pos.rotate(rad);

            ProcessedNote {
                index: i,
                position: rotated_pos,
                time: note.time,
                kind: note.kind.clone(),
                assigned_hand: None,
                confidence: 0.0,
                features: Vec::new(),
                judge: JudgeStatus::NotJudged,
            }
        }).collect()
    }

    fn detect_simultaneous_groups(&self, notes: &[ProcessedNote]) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut used = vec![false; notes.len()];
        const SIMULTANEOUS_THRESHOLD: f32 = 0.03;

        for i in 0..notes.len() {
            if used[i] {
                continue;
            }

            let mut group = vec![i];
            let base_time = notes[i].time;

            for j in i + 1..notes.len() {
                if used[j] {
                    continue;
                }

                let time_diff = (notes[j].time - base_time).abs();

                let threshold = if matches!(notes[i].kind, NoteKind::Hold { .. }) ||
                    matches!(notes[j].kind, NoteKind::Hold { .. }) {
                    SIMULTANEOUS_THRESHOLD * 0.5 // Hold音符使用更小的阈值
                } else {
                    SIMULTANEOUS_THRESHOLD
                };

                if time_diff <= threshold {
                    group.push(j);
                    used[j] = true;
                }
            }

            if group.len() > 1 {
                groups.push(group);
            }
            used[i] = true;
        }

        groups
    }

    fn assign_simultaneous_groups(&mut self, notes: &mut [ProcessedNote], groups: &[Vec<usize>], bpm_list: &mut BpmList) {
        for group in groups {
            if group.len() < 2 {
                continue;
            }

            // 按X坐标排序
            let mut sorted_group: Vec<_> = group.iter()
                .map(|&i| (i, notes[i].position.x))
                .collect();
            sorted_group.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // 计算中间点
            let mid_point = sorted_group.len() / 2;

            // 检查是否有明显的左右分离
            let leftmost_x = sorted_group[0].1;
            let rightmost_x = sorted_group[sorted_group.len() - 1].1;
            let separation = rightmost_x - leftmost_x;

            if separation > 0.4 && group.len() >= 2 {
                // 明显分离：左右手分别处理
                let mut assigned_left = 0;
                let mut assigned_right = 0;

                for (pos, (note_idx, _)) in sorted_group.iter().enumerate() {
                    // 使用mid_point来划分左右手
                    let hand = if pos < mid_point {
                        Hand::Left
                    } else {
                        Hand::Right
                    };

                    // 最多分配 (group.len() / 2).ceil() 个音符
                    match hand {
                        Hand::Left if assigned_left < (group.len() + 1) / 2 => {
                            notes[*note_idx].assigned_hand = Some(Hand::Left);
                            assigned_left += 1;
                        }
                        Hand::Right if assigned_right < group.len() / 2 => {
                            notes[*note_idx].assigned_hand = Some(Hand::Right);
                            assigned_right += 1;
                        }
                        _ => {
                            // 超出限额就交给AI处理
                            let features = self.feature_extractor.extract_features(&notes[*note_idx..*note_idx+1], 1, bpm_list);
                            let ai_decision = self.make_ai_decision(&features, &notes[*note_idx]);
                            notes[*note_idx].assigned_hand = Some(ai_decision.0);
                        }
                    }
                    notes[*note_idx].confidence = 0.9;
                }
            } else {
                // 没有明显分离，全部交给AI处理
                for (note_idx, _) in &sorted_group {
                    let features = self.feature_extractor.extract_features(&notes[*note_idx..*note_idx+1], 1, bpm_list);
                    let ai_decision = self.make_ai_decision(&features, &notes[*note_idx]);

                    notes[*note_idx].assigned_hand = Some(ai_decision.0);
                    notes[*note_idx].confidence = ai_decision.1;
                }
            }

            // 更新手部状态
            self.update_hand_states_for_group(notes, group);
        }
    }

    fn post_process_assignments(&mut self, notes: &mut [ProcessedNote]) {
        self.smooth_hand_transitions(notes);
        self.validate_physical_feasibility(notes);
        self.optimize_recognized_patterns(notes);

        // === 新增：确保连续音符交替分配 ===
        self.ensure_alternating_pattern(notes);
    }

    fn ensure_alternating_pattern(&mut self, notes: &mut [ProcessedNote]) {
        const MAX_CONSECUTIVE: usize = 3;
        const TIME_THRESHOLD: f32 = 0.3;

        let mut consecutive_count = 0;
        let mut last_hand = None;

        // 先收集需要修改的音符索引
        let mut indices_to_switch = Vec::new();

        for (i, note) in notes.iter().enumerate() {
            if let Some(current_hand) = note.assigned_hand {
                // 重置计数
                if last_hand != Some(current_hand) {
                    consecutive_count = 0;
                    last_hand = Some(current_hand);
                    continue;
                }

                // 检查是否连续同手
                consecutive_count += 1;
                if consecutive_count > MAX_CONSECUTIVE {
                    // 查找前一个不同手的音符时间
                    let prev_alt_time = notes[..i]
                        .iter()
                        .rev()
                        .find(|n| n.assigned_hand != Some(current_hand))
                        .map(|n| n.time);

                    // 如果时间间隔允许，标记需要切换
                    if let Some(prev_time) = prev_alt_time {
                        if note.time - prev_time < TIME_THRESHOLD * 2.0 {
                            indices_to_switch.push(i);
                            consecutive_count = 0;
                        }
                    }
                }
            }
        }

        // 执行切换
        for i in indices_to_switch {
            if let Some(current_hand) = notes[i].assigned_hand {
                notes[i].assigned_hand = match current_hand {
                    Hand::Left => Some(Hand::Right),
                    Hand::Right => Some(Hand::Left),
                };
                notes[i].confidence = (notes[i].confidence * 0.8).max(0.6);
            }
        }
    }

    fn update_hand_states_for_group(&mut self, notes: &[ProcessedNote], group: &[usize]) {
        let mut left_positions = Vec::new();
        let mut right_positions = Vec::new();
        let mut group_time = 0.0;

        for &idx in group {
            if let Some(hand) = notes[idx].assigned_hand {
                match hand {
                    Hand::Left => left_positions.push(notes[idx].position),
                    Hand::Right => right_positions.push(notes[idx].position),
                }
                group_time = notes[idx].time;
            }
        }

        // 更新左手状态
        if !left_positions.is_empty() {
            let avg_pos = left_positions.iter().fold(Vector2::new(0.0, 0.0), |acc, &pos| acc + pos)
                * (1.0 / left_positions.len() as f32);
            let success = notes[group[0]].judge == JudgeStatus::Judged; // 实际判定结果
            self.left_hand_state.update_state(avg_pos, group_time, success, &notes[group[0]].kind);
        }

        // 更新右手状态
        if !right_positions.is_empty() {
            let avg_pos = right_positions.iter().fold(Vector2::new(0.0, 0.0), |acc, &pos| acc + pos)
                * (1.0 / right_positions.len() as f32);
            let success = notes[group[0]].judge == JudgeStatus::Judged; // 实际判定结果
            self.right_hand_state.update_state(avg_pos, group_time, success, &notes[group[0]].kind);
        }
    }

    fn ai_assign_single_notes(&mut self, notes: &mut [ProcessedNote], simultaneous_groups: &[Vec<usize>], bpm_list: &mut BpmList) {
        let assigned_indices: std::collections::HashSet<usize> = simultaneous_groups
            .iter()
            .flatten()
            .copied()
            .collect();

        const CONTEXT_WINDOW: usize = 8;


        for i in 0..notes.len() {
            if assigned_indices.contains(&i) {
                continue;
            }

            let start_idx = i.saturating_sub(CONTEXT_WINDOW / 2);
            let end_idx = (i + CONTEXT_WINDOW / 2 + 1).min(notes.len());
            let context = &notes[start_idx..end_idx];

            let mut features = self.feature_extractor.extract_features(context, CONTEXT_WINDOW, bpm_list);
            notes[i].features = features.clone();

            let (chosen_hand, confidence) = self.make_ai_decision(&features, &notes[i]);

            notes[i].assigned_hand = Some(chosen_hand);
            notes[i].confidence = confidence;

            let success = notes[i].judge == JudgeStatus::Judged;
            match chosen_hand {
                Hand::Left => self.left_hand_state.update_state(notes[i].position, notes[i].time, success,&notes[i].kind),
                Hand::Right => self.right_hand_state.update_state(notes[i].position, notes[i].time, success,&notes[i].kind),
            }

            self.record_experience(&features, &notes[i], chosen_hand, confidence);
        }
    }

    fn make_ai_decision(&mut self, features: &[f32], note: &ProcessedNote) -> (Hand, f32) {
        for finger_state in &mut self.finger_states {
            finger_state.update_busy_status(note.time);
        }

        let network_output = self.main_network.light_forward(features);

        let left_ai_confidence = network_output.get(0).unwrap_or(&0.5);
        let right_ai_confidence = network_output.get(1).unwrap_or(&0.5);
        let predicted_difficulty = network_output.get(2).unwrap_or(&1.0);
        let certainty = network_output.get(3).unwrap_or(&0.5);

        let mut finger_scores = Vec::new();

        let note_duration = match &note.kind {
            NoteKind::Hold { end_time, .. } => end_time - note.time,
            _ => 0.1,
        };

        for finger_state in &self.finger_states {
            let score = finger_state.calculate_assignment_score(
                note.position, note.time, *predicted_difficulty, note.time, &note.kind, note_duration
            );
            finger_scores.push((finger_state.finger, score));
        }

        finger_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best_finger = finger_scores[0].0;
        let best_score = finger_scores[0].1;
        let chosen_hand = best_finger.to_hand();

        let stability_bonus = if let Some(last_hand) = self.last_assigned_hand {
            if chosen_hand == last_hand {
                self.consistency_bonus
            } else {
                -self.hand_switch_penalty
            }
        } else {
            0.0
        };

        let ai_weight = (certainty * self.pattern_recognition_strength).clamp(0.2, 0.8);
        let heuristic_weight = 1.0 - ai_weight;

        let hand_ai_confidence = match chosen_hand {
            Hand::Left => *left_ai_confidence,
            Hand::Right => *right_ai_confidence,
        };

        let final_confidence = (hand_ai_confidence * ai_weight +
            (best_score * 0.5 + 0.5) * heuristic_weight +
            stability_bonus).clamp(0.0, 1.0);

        if let Some(last_hand) = self.last_assigned_hand {
            if chosen_hand != last_hand {
                self.hand_switch_count += 1;
            }
        }

        self.recent_assignments.push_back((chosen_hand, note.time, final_confidence));
        if self.recent_assignments.len() > 50 {
            self.recent_assignments.pop_front();
        }
        self.last_assigned_hand = Some(chosen_hand);
        let adjusted_exploration = self.exploration_rate * (1.0 - self.stability_factor);
        let final_hand = if fastrand::f32() < adjusted_exploration {
            if fastrand::bool() { Hand::Left } else { Hand::Right }
        } else {
            chosen_hand
        };

        (final_hand, final_confidence)
    }

    fn record_experience(&mut self, features: &[f32], note: &ProcessedNote, chosen_hand: Hand, confidence: f32) {
        let reward = self.calculate_reward(note, chosen_hand, confidence);

        let experience = Experience {
            state: features.to_vec(),
            action: match chosen_hand { Hand::Left => 0, Hand::Right => 1 },
            reward,
            next_state: features.to_vec(),
            done: false,
            timestamp: note.time,
        };

        self.experience_replay.push(experience);
    }

    fn calculate_reward(&self, note: &ProcessedNote, chosen_hand: Hand, confidence: f32) -> f32 {
        let mut reward = confidence * 2.0;
        let position_bonus = if note.position.x < -0.1 {
            if chosen_hand == Hand::Left { 0.8 } else { -0.4 }
        } else if note.position.x > 0.1 {
            if chosen_hand == Hand::Right { 0.8 } else { -0.4 }
        } else {
            0.3
        };
        reward += position_bonus;
        let hand_fingers: Vec<_> = self.finger_states.iter()
            .filter(|f| f.finger.to_hand() == chosen_hand)
            .collect();

        if let Some(best_finger) = hand_fingers.iter().min_by(|a, b| {
            let dist_a = a.position.distance_to(&note.position);
            let dist_b = b.position.distance_to(&note.position);
            dist_a.partial_cmp(&dist_b).unwrap()
        }) {
            let distance = best_finger.position.distance_to(&note.position);
            let time_diff = note.time - best_finger.last_time;

            let distance_reward = if distance < 0.3 {
                (0.3 - distance) * 2.0
            } else {
                -(distance - 0.3).powf(1.3) * 0.8
            };
            reward += distance_reward;

            let time_reward = if time_diff > 0.15 {
                0.5
            } else if time_diff > 0.08 {
                0.2
            } else if time_diff > 0.0 {
                -((0.08 - time_diff) * 8.0).min(1.5)
            } else {
                -1.0
            };
            reward += time_reward;

            reward -= best_finger.fatigue * 0.3;

            if best_finger.is_busy && note.time < best_finger.busy_until {
                reward -= (best_finger.busy_until - note.time) * 5.0;
            }
        }
        if let Some(last_hand) = self.last_assigned_hand {
            if chosen_hand == last_hand {
                reward += self.consistency_bonus;
            } else {
                reward -= self.hand_switch_penalty;
            }
        }
        let mode_bonus = match self.game_mode {
            GameMode::TwoFinger => {
                if (note.position.x < 0.0 && chosen_hand == Hand::Left) ||
                    (note.position.x > 0.0 && chosen_hand == Hand::Right) {
                    0.3
                } else {
                    -0.1
                }
            }
            GameMode::FourFinger => 0.2,
        };
        reward += mode_bonus;

        reward.clamp(-3.0, 3.0)
    }


    fn smooth_hand_transitions(&self, notes: &mut [ProcessedNote]) {
        if notes.len() < 3 {
            return;
        }

        for i in 1..notes.len() - 1 {
            if let (Some(prev_hand), Some(curr_hand), Some(next_hand)) =
                (notes[i-1].assigned_hand, notes[i].assigned_hand, notes[i+1].assigned_hand) {

                // 减少不必要的手部切换
                if curr_hand != prev_hand && curr_hand != next_hand && prev_hand == next_hand {
                    // 检查是否可以安全切换
                    let time_gap_prev = notes[i].time - notes[i-1].time;
                    let time_gap_next = notes[i+1].time - notes[i].time;

                    if time_gap_prev > 0.15 && time_gap_next > 0.15 {
                        // 检查位置是否合理
                        let position_reasonable = match prev_hand {
                            Hand::Left => notes[i].position.x < 0.3,
                            Hand::Right => notes[i].position.x > -0.3,
                        };

                        if position_reasonable && notes[i].confidence < 0.8 {
                            notes[i].assigned_hand = Some(prev_hand);
                            notes[i].confidence = 0.75; // 中等置信度
                        }
                    }
                }
            }
        }
    }

    fn validate_physical_feasibility(&self, notes: &mut [ProcessedNote]) {
        const MAX_SPEED: f32 = 10.0;
        const MIN_TIME_GAP: f32 = 0.05;

        for i in 1..notes.len() {
            if let (Some(prev_hand), Some(curr_hand)) =
                (notes[i-1].assigned_hand, notes[i].assigned_hand) {

                if prev_hand == curr_hand {
                    let distance = notes[i].position.distance_to(&notes[i-1].position);
                    let time_diff = notes[i].time - notes[i-1].time;

                    if time_diff > 0.001 {
                        let required_speed = distance / time_diff;

                        // 检查物理可行性
                        if required_speed > MAX_SPEED || time_diff < MIN_TIME_GAP {
                            // 尝试切换到另一只手
                            let other_hand = match curr_hand {
                                Hand::Left => Hand::Right,
                                Hand::Right => Hand::Left,
                            };

                            // 检查切换的合理性
                            let switch_reasonable = match other_hand {
                                Hand::Left => notes[i].position.x < 0.3,
                                Hand::Right => notes[i].position.x > -0.3,
                            };

                            if switch_reasonable {
                                notes[i].assigned_hand = Some(other_hand);
                                notes[i].confidence = 0.6; // 降低置信度
                            }
                        }
                    }
                }
            }
        }
    }

    fn optimize_recognized_patterns(&mut self, notes: &mut [ProcessedNote]) {
        // 识别并优化常见模式
        let mut i = 0;
        while i < notes.len() {
            let window_end = (i + 8).min(notes.len());
            let window = &notes[i..window_end];

            if window.len() < 3 {
                i += 1;
                continue;
            }

            // 检测模式类型
            let alternating_score = self.feature_extractor.detect_alternating_pattern(window);
            let stream_score = self.feature_extractor.detect_stream_pattern(window);
            let chord_score = self.feature_extractor.detect_chord_pattern(window);

            // 应用模式特定的优化
            if alternating_score > 0.7 {
                self.optimize_alternating_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else if stream_score > 0.6 {
                self.optimize_stream_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else if chord_score > 0.5 {
                self.optimize_chord_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else {
                i += 1;
            }
        }
    }

    fn optimize_alternating_pattern(&self, notes: &mut [ProcessedNote]) {
        // 确保交替模式真正交替
        for i in 1..notes.len() {
            if let Some(prev_hand) = notes[i-1].assigned_hand {
                let should_alternate = notes[i].position.x * notes[i-1].position.x < 0.0; // 不同侧

                if should_alternate {
                    let opposite_hand = match prev_hand {
                        Hand::Left => Hand::Right,
                        Hand::Right => Hand::Left,
                    };
                    notes[i].assigned_hand = Some(opposite_hand);
                    notes[i].confidence = 0.85;
                }
            }
        }
    }

    fn optimize_stream_pattern(&self, notes: &mut [ProcessedNote]) {
        // 流模式：优先保持手部一致性
        if notes.is_empty() {
            return;
        }

        // 找到最适合的手
        let avg_x: f32 = notes.iter().map(|n| n.position.x).sum::<f32>() / notes.len() as f32;
        let dominant_hand = if avg_x < 0.0 { Hand::Left } else { Hand::Right };

        // 检查是否可以用同一只手处理
        let mut can_use_same_hand = true;
        for i in 1..notes.len() {
            let distance = notes[i].position.distance_to(&notes[i-1].position);
            let time_diff = notes[i].time - notes[i-1].time;

            if time_diff > 0.001 {
                let required_speed = distance / time_diff;
                if required_speed > 8.0 || time_diff < 0.08 {
                    can_use_same_hand = false;
                    break;
                }
            }
        }

        if can_use_same_hand {
            for note in notes.iter_mut() {
                note.assigned_hand = Some(dominant_hand);
                note.confidence = 0.8;
            }
        }
    }

    fn optimize_chord_pattern(&self, notes: &mut [ProcessedNote]) {
        // 和弦模式：确保同时音符分配合理
        let mut time_groups = std::collections::HashMap::new();

        for (i, note) in notes.iter().enumerate() {
            let time_key = (note.time * 20.0).round() as i32; // 50ms精度
            time_groups.entry(time_key).or_insert_with(Vec::new).push(i);
        }

        for indices in time_groups.values() {
            if indices.len() > 1 {
                // 按X坐标排序并分配
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_by(|&a, &b| {
                    notes[a].position.x.partial_cmp(&notes[b].position.x).unwrap()
                });

                let mid = sorted_indices.len() / 2;
                for (pos, &idx) in sorted_indices.iter().enumerate() {
                    notes[idx].assigned_hand = Some(if pos < mid {
                        Hand::Left
                    } else {
                        Hand::Right
                    });
                    notes[idx].confidence = 0.9;
                }
            }
        }
    }

    fn apply_and_learn(&mut self, original_notes: &mut [Note], processed_notes: &[ProcessedNote]) {
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for (i, processed) in processed_notes.iter().enumerate() {
            if let Some(hand) = processed.assigned_hand {
                original_notes[i].hand = hand;

                // 更新对应手指状态，传递音符类型信息
                for finger_state in &mut self.finger_states {
                    if finger_state.finger.to_hand() == hand {
                        finger_state.update_state(
                            processed.position,
                            processed.time,
                            processed.confidence > self.confidence_threshold,
                            &processed.kind  // 新增：传递音符类型
                        );
                        break; // 只更新第一个匹配的手指
                    }
                }

                // 简化的学习：基于置信度判断预测质量
                total_predictions += 1;
                if processed.confidence > self.confidence_threshold {
                    correct_predictions += 1;
                }
            }
        }

        // 更新统计
        self.total_notes_processed += original_notes.len() as u64;
        self.correct_predictions += correct_predictions;

        // 计算当前性能
        let current_accuracy = if total_predictions > 0 {
            correct_predictions as f32 / total_predictions as f32
        } else {
            1.0
        };

        // 更新平均奖励
        self.average_reward = self.average_reward * 0.95 + current_accuracy * 0.05;

        // 记录性能指标
        let metrics = PerformanceMetrics {
            timestamp: processed_notes.last().map(|n| n.time).unwrap_or(0.0),
            accuracy: current_accuracy,
            speed: self.calculate_processing_speed(processed_notes),
            consistency: self.calculate_consistency(processed_notes),
            difficulty_handled: self.calculate_handled_difficulty(processed_notes),
            patterns_recognized: self.get_recognized_patterns(processed_notes),
        };

        self.performance_history.push_back(metrics);
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // 自适应调整
        self.adapt_parameters(current_accuracy);
    }

    fn calculate_processing_speed(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 2 {
            return 0.0;
        }

        let total_time = notes.last().unwrap().time - notes[0].time;
        if total_time > 0.0 {
            notes.len() as f32 / total_time
        } else {
            0.0
        }
    }

    fn calculate_consistency(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.is_empty() {
            return 1.0;
        }

        let avg_confidence: f32 = notes.iter().map(|n| n.confidence).sum::<f32>() / notes.len() as f32;
        let variance: f32 = notes.iter()
            .map(|n| (n.confidence - avg_confidence).powi(2))
            .sum::<f32>() / notes.len() as f32;

        1.0 - variance.sqrt().min(1.0)
    }

    fn calculate_handled_difficulty(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.is_empty() {
            return 0.0;
        }

        let mut total_difficulty = 0.0;
        for note in notes {
            total_difficulty += match note.kind {
                NoteKind::Click => 1.0,
                NoteKind::Drag => 1.3,
                NoteKind::Flick => 1.5,
                NoteKind::Hold { .. } => 1.8,
            };
        }

        total_difficulty / notes.len() as f32
    }

    fn get_recognized_patterns(&self, notes: &[ProcessedNote]) -> Vec<String> {
        let mut patterns = Vec::new();

        if self.feature_extractor.detect_alternating_pattern(notes) > 0.6 {
            patterns.push("alternating".to_string());
        }
        if self.feature_extractor.detect_stream_pattern(notes) > 0.6 {
            patterns.push("stream".to_string());
        }
        if self.feature_extractor.detect_chord_pattern(notes) > 0.5 {
            patterns.push("chord".to_string());
        }
        if self.feature_extractor.detect_jack_pattern(notes) > 0.5 {
            patterns.push("jack".to_string());
        }

        patterns
    }

    fn adapt_parameters(&mut self, current_accuracy: f32) {
        // 自适应探索率
        let lr = &mut self.main_network.learning_rate;

        if current_accuracy < 0.3 {
            // 当准确率极低时大幅提升学习率
            *lr = (*lr * 1.5).clamp(0.001, 0.05);
        } else if current_accuracy < self.average_reward {
            // 表现不佳时温和提升
            *lr = (*lr * 1.1).min(0.03);
        } else {
            // 表现良好时正常衰减
            *lr = (*lr * 0.98).max(0.0005);
        }
        self.exploration_rate = if current_accuracy > 0.7 {
            0.05 + (0.25 * (1.0 - current_accuracy)) // 高准确率时降低探索
        } else {
            0.3 // 低准确率保持高探索
        };

        // 自适应学习率
        if self.main_network.epoch_count > 0 {
            if current_accuracy > self.average_reward {
                self.main_network.learning_rate *= 1.02;
            } else {
                self.main_network.learning_rate *= 0.98;
            }
            self.main_network.learning_rate = self.main_network.learning_rate.clamp(0.0001, 0.05);
        }

        // 自适应置信度阈值
        if current_accuracy > 0.85 {
            self.confidence_threshold = (self.confidence_threshold + 0.01).min(0.9);
        } else if current_accuracy < 0.65 {
            self.confidence_threshold = (self.confidence_threshold - 0.01).max(0.5);
        }
    }

    fn train_network(&mut self) {
        if self.training_episodes % 10 != 0 {
            return;
        }
        if self.experience_replay.len() < 8 {
            return;
        }

        let batch_size = 32.min(self.experience_replay.len());
        let experiences = self.experience_replay.sample(batch_size);

        // 准备训练数据
        let mut training_data = Vec::new();

        for exp in &experiences {
            // 计算目标值
            let target_value = exp.reward + self.discount_factor *
                self.estimate_future_value(&exp.next_state);

            // 创建目标输出
            let mut target_output = vec![0.5, 0.5, 1.0, exp.reward.abs()];

            // 更新对应动作的值
            if exp.action < target_output.len() {
                target_output[exp.action] = target_value.clamp(0.0, 1.0);
            }

            training_data.push((exp.state.clone(), target_output));
        }

        // 训练网络
        self.main_network.train(&training_data);

        println!("网络训练完成: Epoch {}, 学习率: {:.6}, 探索率: {:.3}, 准确率: {:.3}",
                 self.main_network.epoch_count,
                 self.main_network.learning_rate,
                 self.exploration_rate,
                 self.average_reward);
        //io::stdout().flush().unwrap();
    }

    fn estimate_future_value(&mut self, state: &[f32]) -> f32 {
        let output = self.target_network.forward(state);
        output.iter().fold(0.0, |a, &b| a.max(b))
    }
}
use fastrand;

/*
impl fastrand {
    pub fn f32() -> f32 {
        fastrand::f32()
    }

    pub fn bool() -> bool {
        fastrand::bool()
    }

    pub fn usize(max: usize) -> usize {
        if max == 0 { 0 } else { fastrand::usize(0..max) }
    }
}

 */