use crate::config::Config;
use crate::core::note::Hand;
use crate::core::{BpmList, Note, NoteKind};
use crate::judge::JudgeStatus;
use bincode;
use fastrand;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::panic;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::thread;
use std::time::{Duration, Instant};
use wgpu;
use wgpu::util::DeviceExt;
use crossbeam_channel::{unbounded, Receiver as CbReceiver, Sender as CbSender};
use rayon::ThreadPool;
use std::hash::{Hash, Hasher};

type StdHashMap<K, V> = HashMap<K, V>;

#[derive(Clone)]
struct AiRequest {
    id: u64,
    line_id: usize,
    version: u64,
    timestamp: Instant,
    notes: Vec<Note>,
    rotation: f32,
    config: Arc<Config>,
    bpm_list: Arc<BpmList>,
}

#[derive(Clone)]
struct AiResponse {
    id: u64,
    line_id: usize,
    version: u64,
    timestamp: Instant,
    notes: Vec<Note>,
    checksum: u64,
}

struct LineState {
    current_version: u64,
    last_full_update: Instant,
    last_light_update: Instant,
    // 记录为 request_id -> (timestamp, version)
    pending_requests: HashMap<u64, (Instant, u64)>,
}

impl Default for LineState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            current_version: 0,
            last_full_update: now,
            last_light_update: now,
            pending_requests: HashMap::new(),
        }
    }
}

static AI_REQ_TX: OnceCell<CbSender<AiRequest>> = OnceCell::new();
static AI_RESP_RX: OnceCell<CbReceiver<AiResponse>> = OnceCell::new();
static AI_SYSTEM: OnceCell<Mutex<PhiTKAdvancedAI>> = OnceCell::new();
static LINE_STATES: OnceCell<Mutex<HashMap<usize, LineState>>> = OnceCell::new();
static LINE_RESP_QUEUES: OnceCell<Mutex<HashMap<usize, VecDeque<AiResponse>>>> = OnceCell::new();
static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);
static VERSION_COUNTER: AtomicU64 = AtomicU64::new(1);
static TOTAL_TOKENS_USED: AtomicU64 = AtomicU64::new(0);

const LIGHT_UPDATE_INTERVAL_MS: u64 = 16;
const FULL_UPDATE_INTERVAL_MS: u64 = 20;
const REQUEST_TIMEOUT_MS: u64 = 5000;

static START_ONCE: Once = Once::new();

fn calculate_checksum(notes: &[Note]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hash;
    let mut hasher = DefaultHasher::new();
    
    // 优化：批量计算哈希，减少哈希器调用次数
    let len = notes.len();
    len.hash(&mut hasher);
    
    for note in notes {
        note.time.to_bits().hash(&mut hasher);
        let x = note.object.translation.0.now();
        let y = note.object.translation.1.now();
        x.to_bits().hash(&mut hasher);
        y.to_bits().hash(&mut hasher);
        std::mem::discriminant(&note.kind).hash(&mut hasher);
        std::mem::discriminant(&note.hand).hash(&mut hasher);
    }
    
    hasher.finish()
}

fn match_and_merge_notes(original: &mut [Note], updated: &[Note]) -> bool {
    const TIME_THRESHOLD_SEC: f32 = 0.05; // 50 ms
    const POS_THRESHOLD_SQ: f32 = 400.0; // 20^2 避免开方运算

    let mut used = vec![false; updated.len()];
    
    // 预计算 updated 音符的索引，按时间排序以优化匹配
    let mut updated_indices: Vec<usize> = (0..updated.len()).collect();
    updated_indices.sort_by(|&a, &b| {
        updated[a].time.partial_cmp(&updated[b].time).unwrap_or(std::cmp::Ordering::Equal)
    });

    for orig in original.iter_mut() {
        let mut best_idx: Option<usize> = None;
        let mut best_score = std::f64::INFINITY;
        
        // 使用二分查找找到时间相近的音符
        let orig_time = orig.time;
        let search_start = updated_indices.partition_point(|&i| updated[i].time < orig_time - TIME_THRESHOLD_SEC);
        let search_end = updated_indices.partition_point(|&i| updated[i].time <= orig_time + TIME_THRESHOLD_SEC);
        
        // 只检查时间窗口内的音符
        for &idx in &updated_indices[search_start..search_end] {
            if used[idx] { continue; }
            if std::mem::discriminant(&orig.kind) != std::mem::discriminant(&updated[idx].kind) {
                continue;
            }
            
            // 优化：避免重复调用 now()
            let ox = orig.object.translation.0.now();
            let oy = orig.object.translation.1.now();
            let ux = updated[idx].object.translation.0.now();
            let uy = updated[idx].object.translation.1.now();
            
            // 优化：使用平方距离避免开方
            let dx = ox - ux;
            let dy = oy - uy;
            let dist_sq = dx * dx + dy * dy;
            if dist_sq > POS_THRESHOLD_SQ {
                continue;
            }
            
            let dt = (orig_time - updated[idx].time).abs();
            let score = (dt as f64) * 1000.0 + (dist_sq as f64).sqrt();
            if score < best_score {
                best_score = score;
                best_idx = Some(idx);
            }
        }
        
        if let Some(i) = best_idx {
            orig.hand = updated[i].hand;
            used[i] = true;
        } else {
            // 优化：减少错误日志
            return false;
        }
    }

    true
}

fn cleanup_expired_requests(line_states: &mut HashMap<usize, LineState>) {
    let now = Instant::now();
    let timeout_duration = Duration::from_millis(REQUEST_TIMEOUT_MS);
    for (_, state) in line_states.iter_mut() {
        state.pending_requests.retain(|_, (timestamp, _version)| {
            now.duration_since(*timestamp) < timeout_duration
        });
    }
}

fn start_ai_worker_if_needed() {
    START_ONCE.call_once(|| {
        let (tx_req, rx_req) = unbounded::<AiRequest>();
        let (tx_resp, rx_resp) = unbounded::<AiResponse>();

        AI_REQ_TX.set(tx_req.clone()).ok();
        AI_RESP_RX.set(rx_resp.clone()).ok();
        LINE_RESP_QUEUES.get_or_init(|| Mutex::new(HashMap::new()));

        let resp_receiver = rx_resp.clone();
        thread::spawn(move || {
            loop {
                match resp_receiver.recv() {
                    Ok(resp) => {
                        let map = LINE_RESP_QUEUES.get().unwrap();
                        let mut guard = map.lock().unwrap();
                        guard.entry(resp.line_id).or_default().push_back(resp);
                    }
                    Err(_) => {
                        // channel closed -> exit dispatcher
                        break;
                    }
                }
            }
        });


        thread::spawn(move || {
            let mut worker_ai = PhiTKAdvancedAI::load_or_create("phitk_ai_model.bin", 0.0);

            while let Ok(req) = rx_req.recv() {
                let start_time = Instant::now();

                if start_time.duration_since(req.timestamp) > Duration::from_millis(REQUEST_TIMEOUT_MS) {
                    //eprintln!("Dropping expired request id={}, age={:?}", req.id, start_time.duration_since(req.timestamp));
                    continue;
                }

                if worker_ai.rotation != req.rotation {
                    worker_ai.rotation = req.rotation;
                    worker_ai.update_hand_positions();
                }

                let mut notes_copy = req.notes.clone();
                let analysis_result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                    worker_ai.analyze_and_assign(&mut notes_copy, &req.config, &req.bpm_list, req.line_id);
                    notes_copy
                }));

                match analysis_result {
                    Ok(analyzed_notes) => {
                        let checksum = calculate_checksum(&analyzed_notes);
                        let resp = AiResponse {
                            id: req.id,
                            line_id: req.line_id,
                            version: req.version,
                            timestamp: Instant::now(),
                            notes: analyzed_notes,
                            checksum,
                        };

                        if let Err(_e) = tx_resp.send(resp) {
                            //eprintln!("Failed to send AI response for id={} : {:?}", req.id, e);
                        } else {
                            //println!("AI worker: sent response for id={}", req.id);
                        }

                        TOTAL_TOKENS_USED.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(_panic_info) => {
                        //eprintln!("AI analysis panicked for request id={}: {:?}", req.id, panic_info);
                    }
                }
            }

            //println!("AI worker: exiting receive loop");
        });

        thread::spawn(|| {
            let cleanup_interval = Duration::from_secs(30);
            loop {
                thread::sleep(cleanup_interval);
                if let Some(line_states_mutex) = LINE_STATES.get() {
                    if let Ok(mut line_states) = line_states_mutex.lock() {
                        cleanup_expired_requests(&mut line_states);
                    }
                }
            }
        });
    });
}

pub fn assign_hands(notes: &mut [Note], config: &Config, line_id: usize, rotation: f32, bpm_list: &BpmList, ) {
    if notes.is_empty() {
        return;
    }

    start_ai_worker_if_needed();

    let now = Instant::now();

    let line_states = LINE_STATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut line_states_guard = line_states.lock().unwrap_or_else(|poisoned| {
        eprintln!("Line states mutex poisoned, recovering...");
        poisoned.into_inner()
    });

    let line_state = line_states_guard.entry(line_id).or_default();

    // 优化：批量处理响应，减少锁竞争
    if let Some(map) = LINE_RESP_QUEUES.get() {
        let mut responses = Vec::with_capacity(5);
        {
            let mut mq = map.lock().unwrap();
            if let Some(queue) = mq.get_mut(&line_id) {
                const MAX_RESPONSES_PER_FRAME: usize = 5;
                for _ in 0..MAX_RESPONSES_PER_FRAME {
                    if let Some(resp) = queue.pop_front() {
                        responses.push(resp);
                    } else {
                        break;
                    }
                }
            }
        }

        // 批量处理响应
        for resp in responses {
            if resp.line_id != line_id {
                continue;
            }

            if let Some(pending_entry) = line_state.pending_requests.remove(&resp.id) {
                let (_req_ts, req_version) = pending_entry;
                if resp.version == req_version && resp.checksum == calculate_checksum(&resp.notes) {
                    if match_and_merge_notes(notes, &resp.notes) {
                        line_state.current_version = resp.version;
                        line_state.last_full_update = now;
                    }
                }
            }
        }
    }

    // 优化：减少频繁的 AI 更新
    let should_light_update = now.duration_since(line_state.last_light_update) >= Duration::from_millis(LIGHT_UPDATE_INTERVAL_MS);
    if should_light_update {
        line_state.last_light_update = now;
        if let Some(ai_system) = AI_SYSTEM.get() {
            if let Ok(mut ai) = ai_system.lock() {
                if ai.rotation != rotation {
                    ai.rotation = rotation;
                    ai.update_hand_positions();
                }
                ai.light_update_hand_states(notes);
            }
        }
    }

    // 优化：减少全量更新的频率
    let should_full_update = now.duration_since(line_state.last_full_update) >= Duration::from_millis(FULL_UPDATE_INTERVAL_MS);
    if should_full_update {
        const MAX_PENDING_REQUESTS: usize = 3;
        if line_state.pending_requests.len() < MAX_PENDING_REQUESTS {
            let version = VERSION_COUNTER.fetch_add(1, Ordering::Relaxed);
            let request_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);

            line_state.pending_requests.insert(request_id, (now, version));

            // 优化：避免不必要的克隆
            let cfg_arc = Arc::new(config.clone());
            let bpm_arc = Arc::new(bpm_list.clone());

            // 优化：只复制必要的数据
            let notes_snapshot: Vec<Note> = notes.iter().map(|note| Note {
                time: note.time,
                speed: note.speed,
                height: note.height,
                kind: note.kind.clone(),
                judge: note.judge,
                hand: note.hand,
                object: note.object.clone(),
                above: note.above,
                multiple_hint: note.multiple_hint,
                fake: note.fake,
                end_speed: note.end_speed,
                start_height: note.start_height,
                format: note.format,
            }).collect();

            let req = AiRequest {
                id: request_id,
                line_id,
                version,
                timestamp: now,
                notes: notes_snapshot,
                rotation,
                config: cfg_arc,
                bpm_list: bpm_arc,
            };

            if let Some(tx) = AI_REQ_TX.get() {
                if tx.send(req).is_err() {
                    line_state.pending_requests.remove(&request_id);
                }
            }
        }
    }
    drop(line_states_guard);
}

pub fn default_max_grad_norm() -> f32 { 5.0_f32 }

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct Vector2 {
    x: f32,
    y: f32,
}

pub struct HandConfig {
    pub config: Config,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceMetrics {
    timestamp: f32,
    accuracy: f32,
    speed: f32,
    consistency: f32,
    difficulty_handled: f32,
    patterns_recognized: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PhiTKAdvancedAI {
    main_network: DeepNeuralNetwork,
    target_network: DeepNeuralNetwork,
    #[serde(skip)]
    #[serde(default)]
    thread_pool: Option<Arc<ThreadPool>>,
    //#[serde(skip)]
    //#[serde(default)]
    //thread_count: usize,
    feature_extractor: AdvancedFeatureExtractor,
    experience_replay: ExperienceReplay,
    left_hand_state: HandState,
    right_hand_state: HandState,
    rotation: f32,
    exploration_rate: f32,
    discount_factor: f32,
    target_update_frequency: u32,
    total_notes_processed: u64,
    correct_predictions: u64,
    training_episodes: u64,
    average_reward: f32,
    difficulty_adaptation: f32,
    learning_momentum: f32,
    confidence_threshold: f32,
    pattern_memory: BTreeMap<String, f32>,
    performance_history: VecDeque<PerformanceMetrics>,
    version: u32,
    last_save_episodes: u64,
    last_update_time: f32,
    line_rotations: HashMap<usize, f32>,
    game_mode: GameMode,
    finger_states: Vec<FingerState>,
    stability_factor: f32,
    hand_switch_penalty: f32,
    consistency_bonus: f32,
    adaptive_learning_rate: f32,
    pattern_recognition_strength: f32,
    memory_consolidation_rate: f32,
    recent_assignments: VecDeque<(Hand, f32, f32)>,
    hand_switch_count: u32,
    last_assigned_hand: Option<Hand>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ProcessedNote {
    index: usize,
    position: Vector2,
    time: f32,
    kind: NoteKind,
    assigned_hand: Option<Hand>,
    confidence: f32,
    features: Vec<f32>,
    judge: JudgeStatus,
    difficulty: f32,
    duration: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Experience {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
    timestamp: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperienceReplay {
    buffer: VecDeque<Experience>,
    capacity: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum GameMode {
    TwoFinger,
    FourFinger,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Finger {
    LeftIndex,
    LeftMiddle,
    RightIndex,
    RightMiddle,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DeepNeuralNetwork {
    layers: Vec<NetworkLayer>,
    #[serde(default)]learning_rate: f32,
    #[serde(default)]momentum: f32,
    #[serde(default)]dropout_rate: f32,
    batch_size: usize,
    //#[serde(skip)]
    #[serde(default = "default_max_grad_norm")]
    max_grad_norm: f32,
    //#[serde(skip)]
    //weight_decay: f32,
    #[serde(default)]
    last_loss: f32,          // 用于学习率调整
    #[serde(default)]
    bad_epochs: usize,       // 用于学习率调整
    epoch_count: u64,
    #[serde(skip)]device: Option<wgpu::Device>,
    #[serde(skip)]queue: Option<wgpu::Queue>,
    #[serde(skip)]matmul_pipeline: Option<wgpu::ComputePipeline>,
    #[serde(skip)]matmul_bind_group_layout: Option<wgpu::BindGroupLayout>,
    #[serde(skip)]activation_pipelinotes: StdHashMap<ActivationFunction, wgpu::ComputePipeline>,
    #[serde(
        skip
    )]activation_bind_group_layouts: StdHashMap<ActivationFunction, wgpu::BindGroupLayout>,
    #[serde(skip)]gpu_initialized: bool,
    #[serde(skip)]batch_size_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]initialization_attempted: bool,
    #[serde(skip)]initialization_failed: bool,
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
    pre_activations: Vec<f32>,

    #[serde(default)]
    gradients: Vec<f32>,
    #[serde(default)]
    momentum_weights: Vec<Vec<f32>>,
    #[serde(default)]
    momentum_biases: Vec<f32>,

    #[serde(default)]
    bidirectional: bool,
    #[serde(default)]
    seq_len: usize,

    layer_type: LayerType,
    activation_func: ActivationFunction,
    #[serde(skip)]
    weights_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    biases_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    activations_buffer: Option<wgpu::Buffer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum LayerType {
    Dense,
    LSTM,
    Attention,
    Residual,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AdvancedFeatureExtractor {
    pattern_library: HashMap<String, PatternSignature>,
    temporal_patterns: VecDeque<TemporalFeature>,
    spatial_patterns: Vec<SpatialFeature>,
    difficulty_estimator: DifficultyEstimator,
    #[serde(skip)]
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

impl DeepNeuralNetwork {
    fn new() -> Self {
        let mut network = Self {
            layers: Vec::new(),
            learning_rate: 0.001, //学习率
            momentum: 0.9,
            dropout_rate: 0.1,
            batch_size: 24, //批量训练
            max_grad_norm: 5.0, //最大梯度限制
            //weight_decay: 0.0001,
            last_loss: f32::INFINITY, //损失
            bad_epochs: 0,
            epoch_count: 0,
            device: None,
            queue: None,
            matmul_pipeline: None,
            matmul_bind_group_layout: None,
            activation_pipelinotes: StdHashMap::new(),
            activation_bind_group_layouts: StdHashMap::new(),
            gpu_initialized: false,
            batch_size_buffer: None,
            initialization_attempted: false,
            initialization_failed: false,
        };

        network.build_architecture();
        network.init_gpu_sync();
        network
    }

    pub fn clean(&mut self) {
        for layer in &mut self.layers {
            for weights in &mut layer.weights {
                for w in weights {
                    if !w.is_finite() {
                        *w = 0.0;
                    }
                }
            }

            for bias in &mut layer.biases {
                if !bias.is_finite() {
                    *bias = 0.0;
                }
            }

            for activation in &mut layer.activations {
                if !activation.is_finite() {
                    *activation = 0.0;
                }
            }

            for gradient in &mut layer.gradients {
                if !gradient.is_finite() {
                    *gradient = 0.0;
                }
            }

            for momentum_row in &mut layer.momentum_weights {
                for momentum in momentum_row {
                    if !momentum.is_finite() {
                        *momentum = 0.0;
                    }
                }
            }

            for momentum_bias in &mut layer.momentum_biases {
                if !momentum_bias.is_finite() {
                    *momentum_bias = 0.0;
                }
            }

            if layer.seq_len == 0 {
                layer.seq_len = 1;
            }
        }

        if !self.learning_rate.is_finite() {self.learning_rate = 0.001; }
        if !self.momentum.is_finite() { self.momentum = 0.9; }
        if !self.dropout_rate.is_finite() { self.dropout_rate = 0.1; }
    }


    pub fn validate(&self) -> bool {
        for layer in &self.layers {
            for weights in &layer.weights {
                for w in weights {
                    if !w.is_finite() {
                        return false;
                    }
                }
            }

            for bias in &layer.biases {
                if !bias.is_finite() {
                    return false;
                }
            }

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

    pub fn activate_derivative_from_z(x: f32, func: &ActivationFunction) -> f32 {
        match func {
            ActivationFunction::ReLU => {
                if x > 0.0 { 1.0 } else { 0.0 }
            }
            ActivationFunction::Sigmoid => {
                // s = sigmoid(x)
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => {
                let t = x.tanh();
                1.0 - t * t
            }
            ActivationFunction::Swish => {
                // swish(x) = x * sigmoid(x)
                let sig = 1.0 / (1.0 + (-x).exp());
                sig + x * sig * (1.0 - sig)
            }
            ActivationFunction::GELU => {
                // derivative of GELU (approximation consistent with GELU definition used)
                let sqrt_2_over_pi = 0.7978845608_f32;
                let x3 = x * x * x;
                let a = sqrt_2_over_pi * (x + 0.044715 * x3);
                let tanh_a = a.tanh();
                let left = 0.5 * (1.0 + tanh_a);
                let sech2 = 1.0 - tanh_a * tanh_a;
                let a_prime = sqrt_2_over_pi * (1.0 + 0.134145 * x * x); // 0.044715*3 = 0.134145
                left + 0.5 * x * sech2 * a_prime
            }
        }
    }

    fn light_forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut current_input = input.to_vec();

        for layer in self.layers.iter_mut() {
            match layer.layer_type {
                LayerType::Dense => {
                    current_input = Self::dense_forward(layer, &current_input);
                }
                LayerType::LSTM => {
                    current_input = Self::lstm_forward(layer, &current_input);
                }
                LayerType::Attention => {
                    current_input = Self::attention_forward(layer, &current_input);
                }
                LayerType::Residual => {
                    let input_copy = current_input.clone();
                    current_input = Self::dense_forward(layer, &current_input);

                    // 确保维度匹配后才添加残差连接
                    if current_input.len() == input_copy.len() {
                        for i in 0..current_input.len() {
                            current_input[i] += input_copy[i];
                        }
                    }
                }
            }
        }

        current_input
    }


    pub fn init_gpu_sync(&mut self) {
        if self.gpu_initialized {
            println!("[GPU] Already initialized, skipping");
            return;
        }

        if self.initialization_attempted && self.initialization_failed {
            println!("[GPU] Previous initialization failed, skipping");
            return;
        }

        // 如果是第一次初始化，尝试使用 GPU，失败则回退到 CPU
        if !self.initialization_attempted {
            self.initialization_attempted = true;

            let rt = match tokio::runtime::Runtime::new() {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("[GPU] Failed to create Tokio runtime for GPU initialization: {:?}", e);
                    self.initialization_failed = true;
                    return;
                }
            };

            let init_result = rt.block_on(async {
                self.init_gpu().await
            });

            let resources_valid = self.device.is_some()
                && self.queue.is_some()
                && self.batch_size_buffer.is_some()
                && self.matmul_pipeline.is_some();

            if init_result && resources_valid {
                self.gpu_initialized = true;
                self.initialization_failed = false;
                println!("[GPU] GPU initialization successful on first attempt");
            } else {
                self.initialization_failed = true;
                let missing = format!("device: {}, queue: {}, batch_size: {}, pipeline: {}",
                                      self.device.is_some(),
                                      self.queue.is_some(),
                                      self.batch_size_buffer.is_some(),
                                      self.matmul_pipeline.is_some()
                );
                eprintln!("[GPU] GPU initialization partially failed - missing resources: {}", missing);
                eprintln!("[GPU] Falling back to CPU implementation");
                // 清理可能的部分初始化的资源
                self.device = None;
                self.queue = None;
                self.batch_size_buffer = None;
                self.matmul_pipeline = None;
                self.activation_pipelinotes.clear();
                self.activation_bind_group_layouts.clear();
                for layer in &mut self.layers {
                    layer.weights_buffer = None;
                    layer.biases_buffer = None;
                    layer.activations_buffer = None;
                }
            }
        }
        println!("[GPU INIT CHECK] gpu_initialized={}, device.is_some()={}",
                 self.gpu_initialized, self.device.is_some());

    }

    //初始化
    pub async fn init_gpu(&mut self) -> bool {
        if self.gpu_initialized {
            return true;
        }

        let instance_desc = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        };
        let instance = wgpu::Instance::new(&instance_desc);

        let adapter = match instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await {
            Ok(adapter) => adapter,
            Err(e) => {
                eprintln!("Failed to request GPU adapter: {:?}", e);
                return false;
            }
        };

        let adapter_info = adapter.get_info();
        println!(
            "[GPU/CPU SWITCH] Adapter chosen: name=\"{}\", backend={:?}, vendor=0x{:x}, device=0x{:x}",
            adapter_info.name, adapter_info.backend, adapter_info.vendor, adapter_info.device
        );

        let is_software_renderer = {
            let name_lower = adapter_info.name.to_lowercase();
            name_lower.contains("llvmpipe")
                || name_lower.contains("swiftshader")
                || name_lower.contains("software")
                || name_lower.contains("virtual")
                || adapter_info.vendor == 0x10005
                || (adapter_info.vendor == 0x8086 && name_lower.contains("haswell"))
        };

        if is_software_renderer {
            eprintln!(
                "[GPU/CPU SWITCH] Detected software renderer (e.g., llvmpipe). Forcing CPU fallback."
            );
            return false;
        }

        let (device, queue) = match adapter.request_device(&wgpu::DeviceDescriptor::default()).await {
            Ok((d, q)) => (d, q),
            Err(e) => {
                eprintln!("Failed to request GPU device: {:?}", e);
                return false;
            }
        };

        let batch_size_data = [self.batch_size as u32];
        let batch_size_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Batch Size Buffer"),
            contents: bytemuck::cast_slice(&batch_size_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let matmul_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: None,
        });

        let matmul_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&matmul_bind_group_layout],
            push_constant_ranges: &[],
        });

        let matmul_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("matmul.wgsl").into()),
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let mut activation_pipelinotes = StdHashMap::new();
        let mut activation_bind_group_layouts = StdHashMap::new();
        for func in [
            ActivationFunction::ReLU,
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
            ActivationFunction::Swish,
            ActivationFunction::GELU,
        ] {
            let activation_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some(&format!("{:?} Activation Layout", func)),
            });

            let activation_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&activation_bind_group_layout],
                push_constant_ranges: &[],
            });

            let shader_src = self.get_activation_shader(&func);
            let activation_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&activation_pipeline_layout),
                module: &activation_shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

            activation_pipelinotes.insert(func.clone(), pipeline);
            activation_bind_group_layouts.insert(func, activation_bind_group_layout);
        }

        for layer in &mut self.layers {
            let weights_flat = layer.weights.iter().flatten().cloned().collect::<Vec<f32>>();
            layer.weights_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&weights_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            layer.biases_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&layer.biases),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            layer.activations_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (layer.activations.len() * size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }

        self.device = Some(device);
        self.queue = Some(queue);
        self.matmul_pipeline = Some(matmul_pipeline);
        self.matmul_bind_group_layout = Some(matmul_bind_group_layout);
        self.activation_pipelinotes = activation_pipelinotes;
        self.activation_bind_group_layouts = activation_bind_group_layouts;
        self.batch_size_buffer = Some(batch_size_buffer);

        true
    }

    fn get_activation_shader(&self, func: &ActivationFunction) -> String {
        let fn_body = match func {
            ActivationFunction::ReLU => "return max(val, 0.0);",
            ActivationFunction::Sigmoid => "return 1.0 / (1.0 + exp(-val));",
            ActivationFunction::Tanh => "return tanh(val);",
            ActivationFunction::Swish => "return val * (1.0 / (1.0 + exp(-val)));",
            ActivationFunction::GELU => "return 0.5 * val * (1.0 + tanh(val * 0.7978845608 * (1.0 + 0.044715 * val * val)));",
        };

        format!(r#"
    struct BatchSize {{
        size: u32,
    }};

    @group(0) @binding(0) var<storage, read_write> data: array<f32>;
    @group(0) @binding(1) var<storage, read> batch_size: BatchSize;

    fn activate(val: f32) -> f32 {{
        {fn_body}
    }}

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
        let idx = id.x;
        if (idx >= arrayLength(&data)) {{
            return;
        }}
        data[idx] = activate(data[idx]);
    }}
"#)
    }

    fn gpu_forward(&mut self, input: &[f32]) -> Vec<f32> {
        if !self.gpu_initialized {
            self.init_gpu_sync();
            if !self.gpu_initialized {
                println!("[GPU] GPU not available, falling back to CPU");
                return self.light_forward(input);
            }
        }

        let device = match self.device.as_ref() {
            Some(d) => d,
            None => {
                println!("[GPU DEBUG] 'device' is None. FALLING BACK TO CPU (light_forward).");
                return self.light_forward(input);
            },
        };

        let queue = self.queue.as_ref().unwrap();

        // 创建输入缓冲区
        let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Input Buffer"),
            contents: bytemuck::cast_slice(input),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut current_input_buffer = input_buffer;
        let mut current_input_size = input.len();

        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let output_size = layer.weights.len();

            // 创建输出缓冲区
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Output Buffer Layer {}", i)),
                size: (output_size * size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            // 矩阵乘法
            let matmul_bind_group = self.create_matmul_bind_group(
                layer,
                &current_input_buffer,
                &output_buffer
            );

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Matmul Encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Matmul Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(self.matmul_pipeline.as_ref().unwrap());
                cpass.set_bind_group(0, &matmul_bind_group, &[]);

                // 正确计算工作组数量
                let workgroup_count = ((output_size as u32) + 63) / 64;
                cpass.dispatch_workgroups(workgroup_count, 1, 1);
            }

            queue.submit(Some(encoder.finish()));

            // 激活函数
            let activation_bind_group = self.create_activation_bind_group(
                layer,
                &output_buffer
            );

            let activation_pipeline = self.activation_pipelinotes
                .get(&layer.activation_func)
                .unwrap();

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Activation Encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Activation Pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(activation_pipeline);
                cpass.set_bind_group(0, &activation_bind_group, &[]);

                // 正确计算工作组数量
                let workgroup_count = ((output_size as u32) + 63) / 64;
                cpass.dispatch_workgroups(workgroup_count, 1, 1);
            }

            queue.submit(Some(encoder.finish()));

            // 准备下一层的输入
            if i < self.layers.len() - 1 {
                let next_input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Input Buffer Layer {}", i + 1)),
                    size: (output_size * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Copy Encoder"),
                });

                encoder.copy_buffer_to_buffer(
                    &output_buffer,
                    0,
                    &next_input_buffer,
                    0,
                    (output_size * size_of::<f32>()) as u64
                );

                queue.submit(Some(encoder.finish()));
                current_input_buffer = next_input_buffer;
                current_input_size = output_size;
            } else {
                // 最后一层，直接使用输出缓冲区
                current_input_buffer = output_buffer;
            }
        }

        // 读取结果
        let result_size = self.layers.last().unwrap().weights.len();
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (result_size * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &current_input_buffer,
            0,
            &staging_buffer,
            0,
            (result_size * size_of::<f32>()) as u64
        );

        queue.submit(Some(encoder.finish()));

        // 映射缓冲区并读取数据
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        let _ = device.poll(wgpu::PollType::Wait);

        match receiver.recv() {
            Ok(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();

                // 更新最后一层的激活值
                if let Some(last_layer) = self.layers.last_mut() {
                    last_layer.activations = result.clone();
                }

                result
            }
            _ => {
                // 失败时回退到CPU
                println!("[GPU DEBUG] Buffer mapping failed. FALLING BACK TO CPU (light_forward).");
                self.light_forward(input)
            }
        }
    }

    fn create_matmul_bind_group(
        &self,
        layer: &NetworkLayer,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let device = self.device.as_ref().unwrap();
        let bind_group_layout = self.matmul_bind_group_layout.as_ref().unwrap();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: layer.weights_buffer.as_ref().unwrap().as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: layer.biases_buffer.as_ref().unwrap().as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.batch_size_buffer.as_ref().unwrap(),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
            label: None,
        })
    }

    fn create_activation_bind_group(
        &self,
        layer: &NetworkLayer,
        buffer: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        let device = self.device.as_ref().unwrap();
        let layout = self.activation_bind_group_layouts.get(&layer.activation_func).unwrap();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding()
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.batch_size_buffer.as_ref().unwrap(),
                        offset: 0,
                        size: None,
                    }),
                },
            ],
            label: None,
        })
    }

    fn gpu_forward_batch(&mut self, inputs: &[&[f32]], actual_batch_size: usize) -> Vec<Vec<f32>> {
        //println!("[GPU BATCH DEBUG] Processing batch with {} samples.", actual_batch_size);
        if self.device.is_none() || self.queue.is_none() || self.matmul_pipeline.is_none() {
            println!("[GPU DEBUG] Critical GPU resource (device/queue/pipeline) is None. FALLING BACK TO CPU.");
            return inputs.iter().map(|input| self.light_forward(input)).collect();
        }

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let input_size = inputs[0].len();
        let total_input_size = input_size * actual_batch_size;
        let mut batch_input_data = Vec::with_capacity(total_input_size);
        for input in inputs {
            batch_input_data.extend_from_slice(input);
        }
        let batch_input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Batch Input Buffer"),
            contents: bytemuck::cast_slice(&batch_input_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let mut current_input_buffer = batch_input_buffer;
        let mut layer_output_buffers = Vec::new();
        for i in 0..self.layers.len() {
            let layer = &self.layers[i];
            let output_size = layer.weights.len() as u32;
            let total_output_size = output_size * (actual_batch_size as u32);

            let activation_buffer = match layer.activations_buffer.as_ref() {
                Some(b) => b,
                None => panic!("Activation buffer for layer {} is not initialized.", i),
            };

            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Batch Output Buffer for Layer {}", layer_output_buffers.len())),
                size: (total_output_size * size_of::<f32>() as u32) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let matmul_bind_group = self.create_matmul_bind_group(layer, &current_input_buffer, activation_buffer);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Matmul Encoder")
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Batch Matmul Pass"),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(self.matmul_pipeline.as_ref().unwrap());
                cpass.set_bind_group(0, &matmul_bind_group, &[]);
                cpass.dispatch_workgroups(
                    ((output_size * actual_batch_size as u32) + 7) / 8,
                    1,
                    1
                );
            }
            queue.submit(Some(encoder.finish()));

            let activation_bind_group = self.create_activation_bind_group(
                layer,
                &output_buffer,
            );
            let activation_pipeline = self.activation_pipelinotes
                .get(&layer.activation_func)
                .expect("Activation pipeline not initialized");
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Activation Encoder")
            });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Batch Activation Pass"),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(activation_pipeline);
                cpass.set_bind_group(0, &activation_bind_group, &[]);
                cpass.dispatch_workgroups(
                    ((output_size * actual_batch_size as u32) + 63) / 64,
                    1,
                    1
                );
            }
            queue.submit(Some(encoder.finish()));

            current_input_buffer = output_buffer.clone();
            layer_output_buffers.push(output_buffer);
        }

        // 读取最终结果
        let last_layer_output = layer_output_buffers.last().unwrap();
        let last_layer = self.layers.last().unwrap();
        let output_size = last_layer.weights.len();
        let total_output_size = output_size * actual_batch_size;

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batch Staging Buffer"),
            size: (total_output_size * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batch Readback Encoder")
        });

        encoder.copy_buffer_to_buffer(
            last_layer_output,
            0,
            &staging_buffer,
            0,
            (total_output_size * size_of::<f32>()) as u64
        );

        queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = device.poll(wgpu::PollType::wait());

        match rx.recv() {
            Ok(Ok(())) => {},
            _ => return inputs.iter().map(|input| self.light_forward(input)).collect(),
        }

        let data = buffer_slice.get_mapped_range();
        let batch_results: &[f32] = bytemuck::cast_slice(&data);

        let mut results = Vec::with_capacity(actual_batch_size);
        for i in 0..actual_batch_size {
            let start = i * output_size;
            let end = start + output_size;
            results.push(batch_results[start..end].to_vec());
        }

        drop(data);
        staging_buffer.unmap();

        results
    }

    /*
    * 构建神经网络架构
     */
    fn build_architecture(&mut self) {
        self.add_lstm_layer_bi(36, 64, false, 64);
        self.add_attention_layer(64, 32);
        self.add_dense_layer(32, 64, ActivationFunction::GELU);
        self.add_dense_layer(64, 3, ActivationFunction::GELU);
    }

    //TODO: 归一化输出
    fn normalize_output(&self, raw_output: &[f32]) -> Vec<f32> {
        let mut normalized = raw_output.to_vec();
        if normalized.len() >= 2 {
            if (normalized[0] - normalized[1]).abs() < 1e-6 {
                normalized[0] = 0.5;
                normalized[1] = 0.5;
            } else {
                let temperature = 0.1;
                let left_exp = (normalized[0] / temperature).exp();
                let right_exp = (normalized[1] / temperature).exp();
                let sum = left_exp + right_exp;
                if sum > 1e-6 {
                    normalized[0] = left_exp / sum;
                    normalized[1] = right_exp / sum;
                } else {
                    normalized[0] = 0.5;
                    normalized[1] = 0.5;
                }
            }
        }
        if normalized.len() >= 4 {
            normalized[3] = normalized[3].clamp(0.3, 0.96);
        }

        normalized
    }

    fn add_dense_layer(&mut self, input_size: usize, output_size: usize, activation: ActivationFunction) {
        let mut weights = Vec::with_capacity(output_size);
        let mut biases = Vec::with_capacity(output_size);
        let mut momentum_weights = Vec::with_capacity(output_size);

        let std_dev = match activation {
            ActivationFunction::ReLU | ActivationFunction::GELU | ActivationFunction::Swish => {
                (2.0 / input_size as f32).sqrt()
            }
            ActivationFunction::Sigmoid | ActivationFunction::Tanh => {
                (1.0 / input_size as f32).sqrt()
            }
        };

        for _ in 0..output_size {
            let mut row = Vec::with_capacity(input_size);
            let mut momentum_row = Vec::with_capacity(input_size);

            for _ in 0..input_size {
                let weight = (fastrand::f32() * 2.0 - 1.0) * std_dev;
                row.push(weight);
                momentum_row.push(0.0);
            }

            weights.push(row);
            biases.push(0.0);
            momentum_weights.push(momentum_row);
        }

        let layer = NetworkLayer {
            weights,
            biases,
            activations: vec![0.0; output_size],
            pre_activations: vec![0.0; output_size],
            gradients: vec![0.0; output_size],
            momentum_weights,
            momentum_biases: vec![0.0; output_size],
            bidirectional: false,
            seq_len: 1,
            layer_type: LayerType::Dense,
            activation_func: activation,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
        };

        self.layers.push(layer);
    }

    // 双向 LSTM 层 (支持单向)
    fn add_lstm_layer_bi(&mut self, input_size: usize, output_size: usize, bidirectional: bool, seq_len: usize) {
        let dir_mul = if bidirectional { 2 } else { 1 };
        // LSTM 需要 4 * output_size * dir_mul 个 bias（i, f, g, o）
        // 权重：W_ih (4*output_size × input_size) + W_hh (4*output_size × output_size)
        let weights_rows = 4 * output_size * dir_mul;
        
        // 初始化权重 - LSTM 使用 Tanh 激活函数，使用 Xavier/Glorot 初始化
        let std_dev = (1.0 / input_size as f32).sqrt();
        let mut weights = Vec::with_capacity(weights_rows);
        let mut momentum_weights = Vec::with_capacity(weights_rows);
        
        for _ in 0..weights_rows {
            let mut weight_row = Vec::with_capacity(input_size + output_size);
            let mut momentum_row = Vec::with_capacity(input_size + output_size);
            
            // 初始化 W_ih 部分 (input_size 个权重)
            for _ in 0..input_size {
                let weight = (fastrand::f32() * 2.0 - 1.0) * std_dev;
                weight_row.push(weight);
                momentum_row.push(0.0);
            }
            
            // 初始化 W_hh 部分 (output_size 个权重)
            for _ in 0..output_size {
                let weight = (fastrand::f32() * 2.0 - 1.0) * std_dev;
                weight_row.push(weight);
                momentum_row.push(0.0);
            }
            
            weights.push(weight_row);
            momentum_weights.push(momentum_row);
        }
        
        let layer = NetworkLayer {
            weights,
            biases: vec![0.0; 4 * output_size * dir_mul],
            activations: vec![0.0; output_size * dir_mul],
            pre_activations: vec![0.0; weights_rows],
            gradients: vec![0.0; output_size * dir_mul],
            momentum_weights,
            momentum_biases: vec![0.0; 4 * output_size * dir_mul],
            bidirectional,
            seq_len: if seq_len == 0 { 1 } else { seq_len },
            layer_type: LayerType::LSTM,
            activation_func: ActivationFunction::Tanh,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
        };
        self.layers.push(layer);
    }

    fn add_attention_layer(&mut self, input_size: usize, output_size: usize) {
        // Attention 层：Q, K, V 投影
        // 初始化权重 - Attention 使用 GELU 激活函数，使用 Xavier/Glorot 初始化
        let std_dev = (2.0 / input_size as f32).sqrt();
        let mut weights = Vec::with_capacity(3 * output_size);
        let mut momentum_weights = Vec::with_capacity(3 * output_size);
        
        for _ in 0..(3 * output_size) {
            let mut weight_row = Vec::with_capacity(input_size);
            let mut momentum_row = Vec::with_capacity(input_size);
            
            for _ in 0..input_size {
                let weight = (fastrand::f32() * 2.0 - 1.0) * std_dev;
                weight_row.push(weight);
                momentum_row.push(0.0);
            }
            
            weights.push(weight_row);
            momentum_weights.push(momentum_row);
        }
        
        let layer = NetworkLayer {
            weights,
            biases: vec![0.0; 3 * output_size],
            activations: vec![0.0; output_size],
            pre_activations: vec![0.0; 3 * output_size],
            gradients: vec![0.0; output_size],
            momentum_weights,
            momentum_biases: vec![0.0; 3 * output_size],
            bidirectional: false,
            seq_len: 1,
            layer_type: LayerType::Attention,
            activation_func: ActivationFunction::GELU,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
        };
        self.layers.push(layer);
    }

    fn add_residual_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = NetworkLayer {
            weights: vec![vec![0.0; input_size]; output_size],
            biases: vec![0.0; output_size],
            activations: vec![0.0; output_size],
            pre_activations: vec![0.0; output_size],
            gradients: vec![0.0; output_size],
            momentum_weights: vec![vec![0.0; input_size]; output_size],
            momentum_biases: vec![0.0; output_size],
            bidirectional: false,
            seq_len: 1,
            layer_type: LayerType::Residual,
            activation_func: ActivationFunction::ReLU,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
        };

        self.layers.push(layer);
    }

    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        if self.device.is_some() {
            self.gpu_forward(input)
        } else {
            let mut current_input = input.to_vec();
            let mut layer_outputs = Vec::new();

            for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
                match layer.layer_type {
                    LayerType::Dense => {
                        current_input = Self::dense_forward(layer, &current_input);
                    }
                    LayerType::LSTM => {
                        current_input = Self::lstm_forward(layer, &current_input);
                    }
                    LayerType::Attention => {
                        current_input = Self::attention_forward(layer, &current_input);
                    }
                    LayerType::Residual => {
                        let residual_input = layer_outputs
                            .get(layer_idx.saturating_sub(2))
                            .unwrap_or(&current_input)
                            .clone();

                        current_input = Self::residual_forward(layer, &current_input, &residual_input);
                    }
                }
                layer_outputs.push(current_input.clone());
            }

            current_input
        }
    }

    fn dense_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; layer.weights.len()];
        let mut zs = vec![0.0; layer.weights.len()];

        layer.weights.par_iter()
            .zip(layer.biases.par_iter())
            .zip(zs.par_iter_mut())
            .zip(output.par_iter_mut())
            .for_each(|(((weights, bias), z), out)| {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(input.iter()) {
                    sum += w * x;
                }
                *z = sum;
                *out = DeepNeuralNetwork::activate(sum, &layer.activation_func);
            });

        layer.pre_activations = zs;
        layer.activations = output.clone();
        output
    }

    fn lstm_cell(
        x_t: &[f32],
        h_prev: &[f32],
        c_prev: &[f32],
        weights_ih: &[Vec<f32>],
        weights_hh: &[Vec<f32>],
        biases: &[f32]
    ) -> (Vec<f32>, Vec<f32>) {
        let hidden_size = h_prev.len();
        let input_size = x_t.len();

        fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
        fn tanh(x: f32) -> f32 { x.tanh() }

        // 计算四个门的线性组合
        let mut ifgo = vec![0.0; 4 * hidden_size];

        // 计算输入门 (i)
        for i in 0..hidden_size {
            let mut sum = biases[i];
            for j in 0..input_size {
                sum += weights_ih[i][j] * x_t[j];
            }
            for j in 0..hidden_size {
                sum += weights_hh[i][j] * h_prev[j];
            }
            ifgo[i] = sum;
        }

        // 计算遗忘门 (f)
        for i in 0..hidden_size {
            let mut sum = biases[hidden_size + i];
            for j in 0..input_size {
                sum += weights_ih[hidden_size + i][j] * x_t[j];
            }
            for j in 0..hidden_size {
                sum += weights_hh[hidden_size + i][j] * h_prev[j];
            }
            ifgo[hidden_size + i] = sum;
        }

        // 计算候选值 (g)
        for i in 0..hidden_size {
            let mut sum = biases[2 * hidden_size + i];
            for j in 0..input_size {
                sum += weights_ih[2 * hidden_size + i][j] * x_t[j];
            }
            for j in 0..hidden_size {
                sum += weights_hh[2 * hidden_size + i][j] * h_prev[j];
            }
            ifgo[2 * hidden_size + i] = sum;
        }

        // 计算输出门 (o)
        for i in 0..hidden_size {
            let mut sum = biases[3 * hidden_size + i];
            for j in 0..input_size {
                sum += weights_ih[3 * hidden_size + i][j] * x_t[j];
            }
            for j in 0..hidden_size {
                sum += weights_hh[3 * hidden_size + i][j] * h_prev[j];
            }
            ifgo[3 * hidden_size + i] = sum;
        }

        // 应用激活函数
        let i_gate: Vec<f32> = ifgo[0..hidden_size].iter().map(|&x| sigmoid(x)).collect();
        let f_gate: Vec<f32> = ifgo[hidden_size..2*hidden_size].iter().map(|&x| sigmoid(x)).collect();
        let g_candidate: Vec<f32> = ifgo[2*hidden_size..3*hidden_size].iter().map(|&x| tanh(x)).collect();
        let o_gate: Vec<f32> = ifgo[3*hidden_size..4*hidden_size].iter().map(|&x| sigmoid(x)).collect();

        // 更新细胞状态和隐藏状态
        let mut c_t = vec![0.0; hidden_size];
        let mut h_t = vec![0.0; hidden_size];

        for i in 0..hidden_size {
            c_t[i] = f_gate[i] * c_prev[i] + i_gate[i] * g_candidate[i];
            h_t[i] = o_gate[i] * tanh(c_t[i]);
        }

        (h_t, c_t)
    }

    fn lstm_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let seq_len = layer.seq_len;
        if seq_len <= 1 {
            return Self::dense_forward(layer, input);
        }

        let total_input_len = input.len();
        let input_size = total_input_len / seq_len;
        let hidden_size = layer.activations.len();

        // 权重布局: [W_ih | W_hh] 每行长度 = input_size + hidden_size
        let weights = &layer.weights;
        let biases = &layer.biases;

        // 初始化隐藏状态和细胞状态
        let mut h_t = vec![0.0; hidden_size];
        let mut c_t = vec![0.0; hidden_size];

        // 分离权重矩阵
        let weights_ih: Vec<_> = weights.iter()
            .map(|row| row[0..input_size].to_vec())
            .collect();
        let weights_hh: Vec<_> = weights.iter()
            .map(|row| row[input_size..].to_vec())
            .collect();

        // 处理每个时间步
        for t in 0..seq_len {
            let start = t * input_size;
            if start >= total_input_len {
                break;
            }
            let end = (start + input_size).min(total_input_len);
            let x_t = &input[start..end];

            // 使用 lstm_cell 函数处理当前时间步
            let (new_h, new_c) = Self::lstm_cell(x_t, &h_t, &c_t, &weights_ih, &weights_hh, biases);
            h_t = new_h;
            c_t = new_c;
        }

        layer.activations = h_t.clone();
        h_t
    }

    // TODO: 更复杂的注意力机制
    fn attention_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let output_size = layer.activations.len();
        let input_len = input.len();

        // 假设 input 是 [seq_len * feat_dim]，我们需要 reshape
        // 但为了简化，这里假设 input 已经是 [seq_len, feat_dim] 的扁平化
        // 实际上，在 LSTM 后，input 是 [hidden_size]，所以我们做 self-attention on a single vector
        // 更合理的做法：Attention 层接收 [seq_len, hidden_size]，但当前架构限制，我们简化

        // 简化版：对单个向量做 QKV
        let qkv_weights = &layer.weights;
        let qkv_biases = &layer.biases;

        let mut q = vec![0.0; output_size];
        let mut k = vec![0.0; output_size];
        let mut v = vec![0.0; output_size];

        for i in 0..output_size {
            let mut sum_q = qkv_biases[i];
            let mut sum_k = qkv_biases[output_size + i];
            let mut sum_v = qkv_biases[2 * output_size + i];
            for j in 0..input_len {
                if j < qkv_weights[i].len() {
                    sum_q += qkv_weights[i][j] * input[j];
                    sum_k += qkv_weights[output_size + i][j] * input[j];
                    sum_v += qkv_weights[2 * output_size + i][j] * input[j];
                }
            }
            q[i] = sum_q;
            k[i] = sum_k;
            v[i] = sum_v;
        }

        // Scaled Dot-Product Attention
        let dk = output_size as f32;
        let mut score = 0.0;
        for i in 0..output_size {
            score += q[i] * k[i];
        }
        score /= dk.sqrt();

        // Softmax (单值)
        let weight = 1.0; // 简化

        let mut output = vec![0.0; output_size];
        for i in 0..output_size {
            output[i] = weight * v[i];
        }

        layer.activations = output.clone();
        output
    }

    fn residual_forward(layer: &mut NetworkLayer, input: &[f32], residual: &[f32]) -> Vec<f32> {
        let dense_output = Self::dense_forward_static(&layer.weights, &layer.biases, input, &layer.activation_func);
        let mut output = vec![0.0; dense_output.len()];
        for i in 0..output.len() {
            output[i] = dense_output[i] + residual.get(i).copied().unwrap_or(0.0);
        }
        output
    }

    fn dense_forward_static(weights: &[Vec<f32>], biases: &[f32], input: &[f32], activation: &ActivationFunction) -> Vec<f32> {
        let mut output = vec![0.0; weights.len()];
        for (i, (weight_row, bias)) in weights.iter().zip(biases.iter()).enumerate() {
            let mut sum = *bias;
            for (w, x) in weight_row.iter().zip(input.iter()) {
                sum += w * x;
            }
            output[i] = DeepNeuralNetwork::activate(sum, activation);
        }
        output
    }

    fn get_momentum_range(&self) -> (f32, f32) {
        let mut min_momentum = f32::INFINITY;
        let mut max_momentum = f32::NEG_INFINITY;
        for layer in &self.layers {
            for row in &layer.momentum_weights {
                for &m in row {
                    min_momentum = min_momentum.min(m);
                    max_momentum = max_momentum.max(m);
                }
            }
            for &m in &layer.momentum_biases {
                min_momentum = min_momentum.min(m);
                max_momentum = max_momentum.max(m);
            }
        }
        if min_momentum.is_infinite() {
            min_momentum = 0.0;
        }
        if max_momentum.is_infinite() {
            max_momentum = 0.0;
        }

        (min_momentum, max_momentum)
    }

    fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) {
        if training_data.is_empty() {
            eprintln!("警告: 训练数据为空，跳过训练");
            return;
        }
        for layer in &mut self.layers {
            if !layer.weights.is_empty() && !layer.momentum_weights.is_empty() {
                if layer.momentum_weights.len() != layer.weights.len() {
                    layer.momentum_weights = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
                } else {
                    for i in 0..layer.momentum_weights.len() {
                        if layer.momentum_weights[i].len() != layer.weights[i].len() {
                            layer.momentum_weights[i] = vec![0.0; layer.weights[i].len()];
                        }
                    }
                }
            } else if !layer.weights.is_empty() {
                layer.momentum_weights = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            }
            if layer.momentum_biases.len() != layer.biases.len() {
                layer.momentum_biases = vec![0.0; layer.biases.len()];
            }
        }

        let batch_count = (training_data.len() + self.batch_size - 1) / self.batch_size;
        let mut total_loss = 0.0;

        for batch_idx in 0..batch_count {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = start_idx + self.batch_size.min(training_data.len() - start_idx);
            let batch = &training_data[start_idx..end_idx];
            let mut batch_loss = 0.0;
            for (input, target) in batch {
                let output = self.light_forward(input);
                let loss: f32 = output.iter().zip(target.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum();
                batch_loss += loss;
            }
            total_loss += batch_loss;

            self.train_batch(batch);

            if batch_idx == batch_count - 1 {
                let avg_loss = total_loss / training_data.len() as f32;
                self.adapt_learning_rate(avg_loss);
                self.epoch_count += 1;
                if self.epoch_count % 1 == 0 {
                    let (min_weight, max_weight) = self.get_weight_range();
                    let (min_momentum, max_momentum) = self.get_momentum_range();
                    println!("[训练状态] Epoch: {}, Loss: {:.6}, LR: {:.6}, 权重范围 [{:.4}, {:.4}], 动量范围 [{:.4}, {:.4}]",
                             self.epoch_count, avg_loss, self.learning_rate, min_weight, max_weight, min_momentum, max_momentum);
                }
            }
        }
    }

    fn calculate_gradient_norm(&self, gradients: &[Vec<Vec<f32>>], bias_gradients: &[Vec<f32>]) -> f32 {
        let mut total_norm = 0.0;

        for layer_grad in gradients {
            for row in layer_grad {
                for &g in row {
                    total_norm += g * g;
                }
            }
        }

        for bias_grad in bias_gradients {
            for &g in bias_grad {
                total_norm += g * g;
            }
        }

        total_norm.sqrt()
    }

    fn clip_gradients(&self, gradients: &mut [Vec<Vec<f32>>], bias_gradients: &mut [Vec<f32>]) {
        let current_norm = self.calculate_gradient_norm(gradients, bias_gradients);

        // Defensive checks
        if !current_norm.is_finite() {
            eprintln!("[GradClip][EMERGENCY] current_norm is not finite (NaN/Inf). Zeroing gradients and skipping this batch.");
            for layer_grad in gradients.iter_mut() {
                for row in layer_grad.iter_mut() {
                    for g in row.iter_mut() { *g = 0.0; }
                }
            }
            for b in bias_gradients.iter_mut() { for g in b.iter_mut() { *g = 0.0; } }
            return;
        }

        // If within limit, nothing to do
        if current_norm <= self.max_grad_norm {
            return;
        }

        // Compute desired scale (<= 1.0)
        let desired_scale = self.max_grad_norm / current_norm;

        // 如果 desired_scale 极小（说明 gradient too huge），直接丢弃该 batch 的梯度以防爆炸
        const EMERGENCY_SCALE_FLOOR: f32 = 1e-6;
        if desired_scale < EMERGENCY_SCALE_FLOOR {
            eprintln!(
                "[GradClip][EMERGENCY] current_norm={:.6}, max_grad_norm={:.6}, desired_scale={:.12} < EMERGENCY_SCALE_FLOOR. \
            Zeroing gradients and skipping this batch to avoid catastrophic update.",
                current_norm, self.max_grad_norm, desired_scale
            );
            for layer_grad in gradients.iter_mut() {
                for row in layer_grad.iter_mut() {
                    for g in row.iter_mut() { *g = 0.0; }
                }
            }
            for b in bias_gradients.iter_mut() { for g in b.iter_mut() { *g = 0.0; } }
            return;
        }

        let scale = desired_scale.clamp(f32::MIN_POSITIVE, 1.0);
        eprintln!(
            "[GradClip] current_norm={:.6}, max_grad_norm={:.6}, applied_scale={:.9}",
            current_norm, self.max_grad_norm, scale
        );

        for layer_grad in gradients.iter_mut() {
            for row in layer_grad.iter_mut() {
                for g in row.iter_mut() { *g *= scale; }
            }
        }
        for b in bias_gradients.iter_mut() {
            for g in b.iter_mut() { *g *= scale; }
        }

        if current_norm > self.max_grad_norm * 10.0 {
            eprintln!("[GradClip][ALERT] huge norm {:.6} at epoch {}", current_norm, self.epoch_count);
            for (li, layer_grad) in gradients.iter().enumerate() {
                let mut max_abs = 0.0f32;
                for row in layer_grad {
                    for &g in row { if g.abs() > max_abs { max_abs = g.abs(); } }
                }
                eprintln!("[GradClip][ALERT] layer {} max_abs_grad = {:.6}", li, max_abs);
            }
        }
    }

    fn reset_problem_layers(&mut self) {
        let mut layers_to_reset = Vec::new();

        for (i, layer) in self.layers.iter().enumerate() {
            let mut weight_ok = true;
            for &w in layer.weights.iter().flatten() {
                if w.is_nan() || w.is_infinite() || w.abs() > 30.0 {
                    weight_ok = false;
                    break;
                }
            }
            if let ActivationFunction::Tanh = layer.activation_func {
                let mut saturated = 0;
                for &a in &layer.activations {
                    if a.abs() > 0.99 {
                        saturated += 1;
                    }
                }
                if saturated > layer.activations.len() / 2 {
                    layers_to_reset.push(i);
                    continue;
                }
            }

            if !weight_ok {
                layers_to_reset.push(i);
            }
        }
    }

    fn adapt_learning_rate(&mut self, loss: f32) {
        if loss < self.last_loss * 0.95 {
            self.learning_rate = (self.learning_rate * 1.05).min(0.01);
            self.bad_epochs = 0;
        }
        else if loss > self.last_loss * 1.05 {
            self.learning_rate = (self.learning_rate * 0.8).max(0.0001);
            self.bad_epochs += 1;
            if self.bad_epochs >= 5 {
                println!("连续{}个epoch表现不佳，执行网络重置", self.bad_epochs);
                self.reset_problem_layers();
                self.bad_epochs = 0;
            }
        }
        else {
            self.bad_epochs = 0;
        }

        self.last_loss = loss;
    }

    fn get_weight_range(&self) -> (f32, f32) {
        let mut min_weight = f32::INFINITY;
        let mut max_weight = f32::NEG_INFINITY;

        for layer in &self.layers {
            for row in &layer.weights {
                for &w in row {
                    min_weight = min_weight.min(w);
                    max_weight = max_weight.max(w);
                }
            }
        }

        (min_weight, max_weight)
    }

    pub fn train_batch(&mut self, batch: &[(Vec<f32>, Vec<f32>)]) {
        let batch_size = batch.len();

        let mut inputs: Vec<&[f32]> = Vec::with_capacity(batch_size);
        let mut targets: Vec<&[f32]> = Vec::with_capacity(batch_size);

        for (input, target) in batch {
            inputs.push(input);
            targets.push(target);
        }
        let outputs = self.gpu_forward_batch(&inputs, batch_size);

        let mut total_gradients: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 0]; 0]; self.layers.len()];
        let mut total_bias_gradients: Vec<Vec<f32>> = vec![vec![0.0; 0]; self.layers.len()];

        for layer_idx in 0..self.layers.len() {
            let layer = &self.layers[layer_idx];
            if !layer.weights.is_empty() {
                total_gradients[layer_idx] = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            } else {
                total_gradients[layer_idx] = vec![];
            }
            total_bias_gradients[layer_idx] = vec![0.0; layer.biases.len()];
        }

        for i in 0..batch_size {
            self.backward(
                inputs[i],
                &outputs[i],
                targets[i],
                &mut total_gradients,
                &mut total_bias_gradients
            );
        }

        if batch_size > 0 {
            let inv_batch = 1.0f32 / (batch_size as f32);
            for layer_idx in 0..self.layers.len() {
                let wg = &mut total_gradients[layer_idx];
                for r in 0..wg.len() {
                    for c in 0..wg[r].len() {
                        wg[r][c] *= inv_batch;
                    }
                }
                let bg = &mut total_bias_gradients[layer_idx];
                for j in 0..bg.len() {
                    bg[j] *= inv_batch;
                }
            }
        }
        self.clip_gradients(&mut total_gradients, &mut total_bias_gradients);
        self.apply_gradients(&total_gradients, &total_bias_gradients, batch_size);
        println!("First weight value after update: {:.6}", self.layers[0].weights[0][0]);
    }


    pub fn backward(&mut self, input: &[f32], output: &[f32], target: &[f32], total_gradients: &mut [Vec<Vec<f32>>], total_bias_gradients: &mut [Vec<f32>]) {
        let mut layer_errors = vec![vec![0.0; 0]; self.layers.len()];
        if let Some(last_layer_idx) = self.layers.len().checked_sub(1) {
            let last_idx = self.layers.len() - 1;
            let out_dim = output.len() as f32;
            let mut output_errors: Vec<f32> = output.iter()
                .zip(target.iter())
                .map(|(o, t)| 2.0 * (o - t) / out_dim)
                .collect();

            // 转换：dL/da -> dL/dz（输出层）
            for i in 0..output_errors.len() {
                let z = if i < self.layers[last_idx].pre_activations.len() {
                    self.layers[last_idx].pre_activations[i]
                } else {
                    // pre_activations 不足时，尝试使用 activation 值来近似导数（针对 Sigmoid/Tanh/ReLU 可行）
                    // 如果既没有 z 也无法用 a 计算（比如 Swish/GELU），activate_derivative_from_z 的调用会作为最优路径；
                    // TODO：这里先用 0.0 做占位（下面会用 activation 备选计算）。
                    0.0
                };

                // 优先用 z（如果确实有合理的 z），否则尝试用 activation 值计算（见下面的派生逻辑）
                let deriv = if i < self.layers[last_idx].pre_activations.len() {
                    Self::activate_derivative_from_z(z, &self.layers[last_idx].activation_func)
                } else {
                    // fallback based on activation value
                    let a = if i < self.layers[last_idx].activations.len() {
                        self.layers[last_idx].activations[i]
                    } else {
                        0.0
                    };
                    match self.layers[last_idx].activation_func {
                        ActivationFunction::ReLU => if a > 0.0 { 1.0 } else { 0.0 },
                        ActivationFunction::Sigmoid => a * (1.0 - a),
                        ActivationFunction::Tanh => 1.0 - a * a,
                        // 对 Swish/GELU 没有简单的从 a 得到导数的方法 —— 打印警告并使用 1.0 的保守近似（避免进一步放大）
                        ActivationFunction::Swish | ActivationFunction::GELU => {
                            eprintln!("[Backward Warning] missing pre_activation for layer {}, using fallback derivative=1.0 for {:?}", last_idx, self.layers[last_idx].activation_func);
                            1.0
                        }
                    }
                };

                output_errors[i] *= deriv; // 现在 output_errors 存的是 dL/dz
            }

            if output_errors.iter().any(|&e| !e.is_finite()) {
                eprintln!("[Gradient Safety] NaN/Inf detected in output errors, skipping backward pass for this sample.");
                return;
            }

            layer_errors[last_layer_idx] = output_errors;
        }

        // activation a（不是 z）计算导数（可处理常见激活）
        let derivative_from_activation = |a: f32, func: &ActivationFunction| -> f32 {
            match func {
                ActivationFunction::ReLU => if a > 0.0 { 1.0 } else { 0.0 },
                ActivationFunction::Sigmoid => a * (1.0 - a),
                ActivationFunction::Tanh => 1.0 - a * a,
                // 对 Swish/GELU 无法从 a 得到精确导数，返回 1.0
                ActivationFunction::Swish | ActivationFunction::GELU => 1.0,
            }
        };

        for layer_idx in (0..self.layers.len()).rev() {
            if layer_idx < self.layers.len() - 1 {
                let next_layer = &self.layers[layer_idx + 1];
                let mut current_errors = vec![0.0; self.layers[layer_idx].activations.len()];

                // dL/da_i = sum_j w_j,i * dL/dz_j (next layer 的权重布局： next_layer.weights[j][i] )
                for i in 0..current_errors.len() {
                    for (j, error) in layer_errors[layer_idx + 1].iter().enumerate() {
                        if j < next_layer.weights.len() && i < next_layer.weights[j].len() {
                            current_errors[i] += error * next_layer.weights[j][i];
                        }
                    }
                }

                // 把 dL/da -> dL/dz，通过乘上激活导数。
                // 优先使用 pre_activations（z）；如果不存在对应 z，再尝试用 activation（a）来近似。
                let func = &self.layers[layer_idx].activation_func;
                let has_z = self.layers[layer_idx].pre_activations.len() >= current_errors.len();
                for i in 0..current_errors.len() {
                    let deriv = if has_z {
                        Self::activate_derivative_from_z(self.layers[layer_idx].pre_activations[i], func)
                    } else {
                        if matches!(func, ActivationFunction::Swish | ActivationFunction::GELU) {
                            eprintln!("[Backward Warning] layer {} missing pre_activations; using fallback derivative for {:?}", layer_idx, func);
                        }
                        let a = if i < self.layers[layer_idx].activations.len() {
                            self.layers[layer_idx].activations[i]
                        } else {
                            0.0
                        };
                        derivative_from_activation(a, func)
                    };
                    current_errors[i] *= deriv;
                }

                if current_errors.iter().any(|&e| !e.is_finite()) {
                    eprintln!("[Gradient Safety] NaN/Inf detected in current_errors at layer {}, skipping backward for this sample.", layer_idx);
                    return;
                }

                layer_errors[layer_idx] = current_errors;
            }

            let prev_acts: &[f32] = if layer_idx > 0 {
                &self.layers[layer_idx - 1].activations[..]
            } else {
                input
            };

            // 计算权重梯度 calculate_layer_gradients 期望 errors 是 dL/dz
            self.calculate_layer_gradients(
                layer_idx,
                &layer_errors[layer_idx],
                &mut total_gradients[layer_idx],
                &mut total_bias_gradients[layer_idx],
                prev_acts,
            );
        }
    }


    fn calculate_layer_gradients(
        &self,
        layer_idx: usize,
        errors: &[f32],
        gradients: &mut [Vec<f32>],
        bias_gradients: &mut [f32],
        prev_activations: &[f32],
    ) {
        // prev_activations 对应前一层的激活 (或者当 layer_idx==0 时为输入)
        // 对每个输出单元 i，遍历所有输入 j 并累加梯度： grad[i][j] += error_i * activation_j
        for (i, &error) in errors.iter().enumerate() {
            // 若没有该输出单元对应的梯度行，则跳过（保持兼容）
            if i >= gradients.len() {
                continue;
            }

            let grad_row = &mut gradients[i];
            // 只遍历 prev_activations 和 grad_row 的重叠范围，避免索引越界
            let common_len = std::cmp::min(prev_activations.len(), grad_row.len());
            for j in 0..common_len {
                grad_row[j] += error * prev_activations[j];
            }

            if i < bias_gradients.len() {
                bias_gradients[i] += error;
            }
        }
    }

    fn apply_gradients(&mut self, gradients: &[Vec<Vec<f32>>], bias_gradients: &[Vec<f32>], batch_size: usize) {
        let batch_size_f = batch_size as f32;
        const MAX_GRAD: f32 = 5.0;
        const MAX_WEIGHT: f32 = 10.0;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            for (i, weight_row) in layer.weights.iter_mut().enumerate() {
                for (j, weight) in weight_row.iter_mut().enumerate() {
                    if i < gradients[layer_idx].len() && j < gradients[layer_idx][i].len() {
                        let mut grad = gradients[layer_idx][i][j] / batch_size_f;

                        if !grad.is_finite() {
                            eprintln!("[Gradient Safety] NaN/Inf gradient detected at layer {}, weight[{}][{}], setting to 0.", layer_idx, i, j);
                            grad = 0.0;
                        }
                        grad = grad.clamp(-MAX_GRAD, MAX_GRAD);

                        layer.momentum_weights[i][j] =
                            self.momentum * layer.momentum_weights[i][j] - self.learning_rate * grad;
                        let momentum_update = layer.momentum_weights[i][j];
                        let clamped_momentum = momentum_update.clamp(-MAX_GRAD, MAX_GRAD);

                        *weight += clamped_momentum;
                        *weight = weight.clamp(-MAX_WEIGHT, MAX_WEIGHT);
                        if !weight.is_finite() {
                            eprintln!("[Gradient Safety] NaN/Inf weight detected at layer {}, weight[{}][{}], setting to 0.", layer_idx, i, j);
                            *weight = 0.0;
                        }
                    }
                }
            }

            for (i, bias) in layer.biases.iter_mut().enumerate() {
                if i < bias_gradients[layer_idx].len() {
                    let mut grad = bias_gradients[layer_idx][i] / batch_size_f;

                    if !grad.is_finite() {
                        eprintln!("[Gradient Safety] NaN/Inf gradient detected at layer {}, bias[{}], setting to 0.", layer_idx, i);
                        grad = 0.0;
                    }
                    grad = grad.clamp(-MAX_GRAD, MAX_GRAD);

                    layer.momentum_biases[i] = self.momentum * layer.momentum_biases[i] - self.learning_rate * grad;
                    let momentum_update = layer.momentum_biases[i];
                    let clamped_momentum = momentum_update.clamp(-MAX_GRAD, MAX_GRAD);

                    *bias += clamped_momentum;
                    *bias = bias.clamp(-MAX_WEIGHT, MAX_WEIGHT);

                    if !bias.is_finite() {
                        eprintln!("[Gradient Safety] NaN/Inf bias detected at layer {}, bias[{}], setting to 0.", layer_idx, i);
                        *bias = 0.0;
                    }
                }
            }
        }
        if self.device.is_none() || self.queue.is_none() {
            return;
        }
        for layer in &mut self.layers {
            let queue = self.queue.as_ref().expect("GPU queue not initialized — call init_gpu() before calling this method");
            let weights_buf = layer.weights_buffer.as_ref().expect("weights_buffer not created for layer");
            let biases_buf = layer.biases_buffer.as_ref().expect("biases_buffer not created for layer");
            let weights_flat = layer.weights.iter().flatten().cloned().collect::<Vec<f32>>();
            queue.write_buffer(weights_buf, 0, bytemuck::cast_slice(&weights_flat));
            queue.write_buffer(biases_buf, 0, bytemuck::cast_slice(&layer.biases));
        }
    }
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

    fn update_state(&mut self, new_pos: Vector2, time: f32, success: bool, note_kind: &NoteKind) {
        let time_diff = time - self.last_time;

        if time_diff > 0.001 {
            let distance = new_pos.distance_to(&self.position);
            let new_velocity = (new_pos - self.position) * (1.0 / time_diff);
            self.velocity = self.velocity * 0.7 + new_velocity * 0.3;

            // 疲劳计算
            let base_movement_cost = distance * 0.12;
            let speed_cost = (self.velocity.magnitude() / 10.0).powf(1.5) * 0.08;
            let time_factor = if time_diff < 0.1 { 2.0 } else { 1.0 };

            let total_cost = (base_movement_cost + speed_cost) * time_factor;
            self.fatigue = (self.fatigue + total_cost).min(1.0);

            // 动态恢复率，基于休息时间
            let rest_factor = if time_diff > 0.3 { 2.0 } else { 1.0 };
            let recovery = (time_diff * 0.25 * rest_factor).min(0.3);
            self.fatigue = (self.fatigue - recovery).max(0.0);
        }

        // 繁忙状态更新
        let busy_duration = match note_kind {
            NoteKind::Hold { end_time, .. } => (end_time - time + 0.1).max(0.15),
            NoteKind::Drag => 0.25,
            NoteKind::Flick => 0.2,
            NoteKind::Click => 0.12,
        };

        self.is_busy = true;
        self.busy_until = time + busy_duration;

        // 信心更新
        self.total_actions += 1;
        if success {
            self.success_streak += 1;
            // 信心增长有上限，避免过于自信.jpg
            let confidence_gain = (0.01 * (1.0 - self.confidence)).max(0.002);
            self.confidence = (self.confidence + confidence_gain).min(0.95);
        } else {
            self.success_streak = 0;
            // 失败时信心下降更明显
            let confidence_loss = (0.03 + self.confidence * 0.01).max(0.01);
            self.confidence = (self.confidence - confidence_loss).max(0.15);
        }

        // 性能评分计算改进
        let recent_window = 15.0_f32.min(self.total_actions as f32);
        let recent_success_rate = if recent_window > 0.0 {
            self.success_streak as f32 / recent_window
        } else {
            0.5
        };

        // 添加随机波动，模拟真实表现
        let randomnotess = (fastrand::f32() - 0.5) * 0.1;
        self.performance_score = (
            recent_success_rate * 0.4 +
                self.confidence * 0.35 +
                (1.0 - self.fatigue) * 0.25 +
                randomnotess
        ).clamp(0.1, 0.95);

        self.position = new_pos;
        self.last_time = time;
    }

    fn calculate_assignment_score(&self, target_pos: Vector2, time: f32, note_difficulty: f32,
                                  current_time: f32, note_kind: &NoteKind, _note_duration: f32) -> f32 {
        // 繁忙状态检查
        let is_available = current_time >= self.busy_until;
        let distance = target_pos.distance_to(&self.position);
        let time_diff = time - self.last_time;

        let mut score = 0.5; // 基础分数

        // 繁忙惩罚
        if !is_available {
            let busy_penalty = (self.busy_until - current_time).min(1.0) * 0.8;
            score -= busy_penalty;
        }

        // 位置偏好 - 增加梯度变化
        // 在2指模式下，使用更合理的移动策略，不单纯依赖x坐标的正负
        let position_weight = if self.finger.to_hand() == Hand::Left {
            // 左手优先选择x坐标较小的位置，但允许一定的灵活性
            let x_diff = self.position.x - target_pos.x;
            if x_diff > 0.2 {
                // 目标在左手左侧，非常适合
                0.4 + (x_diff * 0.6).min(0.2)
            } else if x_diff > -0.1 {
                // 目标在左手附近，适中
                0.35 * (1.0 - (x_diff + 0.1) / 0.3).max(0.0)
            } else {
                // 目标在左手右侧，不太适合
                -0.2 - ((-x_diff - 0.1) * 0.5).min(0.3)
            }
        } else {
            // 右手优先选择x坐标较大的位置，但允许一定的灵活性
            let x_diff = target_pos.x - self.position.x;
            if x_diff > 0.2 {
                // 目标在右手右侧，非常适合
                0.4 + (x_diff * 0.6).min(0.2)
            } else if x_diff > -0.1 {
                // 目标在右手附近，适中
                0.35 * (1.0 - (x_diff + 0.1) / 0.3).max(0.0)
            } else {
                // 目标在右手左侧，不太适合
                -0.2 - ((-x_diff - 0.1) * 0.5).min(0.3)
            }
        };
        score += position_weight;

        // 距离评分
        let distance_score = if distance < 0.1 {
            0.9 - distance * 2.0
        } else if distance < 0.3 {
            0.7 - (distance - 0.1) * 1.5
        } else if distance < 0.6 {
            0.4 - (distance - 0.3) * 0.8
        } else {
            0.1 - (distance - 0.6).min(0.4) * 0.25
        };
        score *= distance_score.max(0.1);

        // 疲劳影响 - 非线性
        let fatigue_penalty = self.fatigue.powf(1.5) * 0.35;
        score *= 1.0 - fatigue_penalty;

        // 时间间隔评分
        if time_diff > 0.001 {
            let time_score = if time_diff < 0.06 {
                0.3 + (time_diff / 0.06) * 0.4 // 过快惩罚
            } else if time_diff < 0.2 {
                0.7 + ((time_diff - 0.06) / 0.14) * 0.25 // 最佳区间
            } else if time_diff < 0.5 {
                0.95 - ((time_diff - 0.2) / 0.3) * 0.15 // 稍慢
            } else {
                0.8 + ((time_diff - 0.5).min(0.5) / 0.5) * 0.15 // 很慢反而好
            };
            score *= time_score;
        }

        if matches!(note_kind, NoteKind::Hold { .. }) {
            if self.is_busy {
                score *= 0.6;
            }
        }

        let performance_factor = 0.5 + self.performance_score * 0.5;
        score *= performance_factor;

        let difficulty_factor = 1.0 - (note_difficulty - 1.0).max(0.0) * (1.0 - self.confidence) * 0.15;
        score *= difficulty_factor;

        let randomnotess = (fastrand::f32() - 0.5) * 0.05;
        score += randomnotess;

        score.clamp(0.0, 1.0)
    }
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
            last_logged_bpm: None,
            last_logged_time: -1.0,
        }
    }

    pub fn extract_features(&mut self, notes: &[ProcessedNote], window_size: usize, bpm_list: &BpmList) -> Vec<f32> {
        let mut all_features = Vec::new();
        let actual_len = notes.len().min(window_size);

        // 对窗口中的每个 note 提取 28 维特征
        for i in 0..actual_len {
            let note_slice = &notes[i..i+1]; // 只传当前 note
            let mut features = Vec::new();
            features.extend(self.extract_position_features(note_slice));
            features.extend(self.extract_temporal_features(note_slice, 1, bpm_list.clone()));
            features.extend(self.extract_pattern_features(note_slice));
            features.extend(self.extract_difficulty_features(note_slice));
            features.extend(self.extract_velocity_features(note_slice));
            features.extend(self.extract_spatial_features(note_slice));
            debug_assert_eq!(features.len(), 28, "Single note feature dimension must be 28");
            all_features.extend(features);
        }

        // 填充到 window_size × 28
        let target_len = window_size * 28;
        if all_features.len() < target_len {
            all_features.resize(target_len, 0.0);
        }

        // 清理 NaN/Inf
        for f in &mut all_features {
            if !f.is_finite() { *f = 0.0; }
        }

        // 归一化（保持与原逻辑一致）
        let min_val = all_features.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = all_features.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if (max_val - min_val).abs() < f32::EPSILON {
            for f in &mut all_features { *f = 0.0; }
        } else {
            for f in &mut all_features {
                *f = 2.0 * (*f - min_val) / (max_val - min_val) - 1.0;
            }
        }

        all_features
    }

    fn extract_position_context_features(&self, note: &ProcessedNote, context: &[ProcessedNote]) -> Vec<f32> {
        let mut features = Vec::new();

        // 1. 相对位置特征 (4维)
        if let Some(prev_note) = context.iter().rev().find(|n| n.time < note.time) {
            let dx = note.position.x - prev_note.position.x;
            let dy = note.position.y - prev_note.position.y;
            features.push(dx);
            features.push(dy);
            features.push((dx * dx + dy * dy).sqrt()); // 距离
            features.push(dy.atan2(dx)); // 方向角度
        } else {
            features.extend(vec![0.0; 4]);
        }

        // 2. 局部密度特征 (3维)
        let nearby_notes: Vec<_> = context.iter()
            .filter(|n| (n.time - note.time).abs() < 0.5)
            .collect();

        let left_density = nearby_notes.iter()
            .filter(|n| n.position.x < -0.1)
            .count() as f32;
        let right_density = nearby_notes.iter()
            .filter(|n| n.position.x > 0.1)
            .count() as f32;

        features.push(left_density);
        features.push(right_density);

        let total_density = left_density + right_density;
        if total_density > 0.0 {
            features.push((right_density - left_density) / total_density);
        } else {
            features.push(0.0);
        }

        // 3. 时间序列特征 (1维): 当前手部连续分配次数
        let mut streak = 0;
        let mut last_hand = None;

        // 按时间逆序查找最近的手部分配
        let mut past_notes: Vec<_> = context.iter()
            .filter(|n| n.time < note.time && n.assigned_hand.is_some())
            .collect();
        past_notes.sort_by(|a, b| b.time.partial_cmp(&a.time).unwrap());

        for n in past_notes {
            if let Some(hand) = n.assigned_hand {
                if last_hand.is_none() {
                    last_hand = Some(hand);
                    streak = 1;
                } else if last_hand == Some(hand) {
                    streak += 1;
                } else {
                    break;
                }
            }
        }

        features.push(streak as f32);

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

    fn extract_temporal_features(&mut self, notes: &[ProcessedNote], window_size: usize, bpm_list: BpmList) -> Vec<f32> {
        if notes.is_empty() {
            return vec![0.0; 6];
        }

        let window_end = window_size.min(notes.len());
        let window = &notes[0..window_end];

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

        let rhythm_complexity = self.calculate_rhythm_complexity(&intervals);

        let current_time = notes[0].time;
        let current_bpm = bpm_list.now_bpm(current_time);

        let should_log = self.last_logged_bpm.is_none() ||
            current_bpm != self.last_logged_bpm.unwrap() ||
            current_time - self.last_logged_time > 1.0;

        if should_log {
            //println!("[BPM LOG] Time: {:.2}s, BPM: {:.1}", current_time, current_bpm);
            self.last_logged_bpm = Some(current_bpm);
            self.last_logged_time = current_time;
        }

        vec![
            avg_interval,
            interval_variance,
            rhythm_complexity,
            current_bpm,
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

            if time_diff < 0.2 && distance < 0.4 {
                stream_score += 1.0;
            }

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
        let mut consecutive_jacks = 0;

        for i in 2..notes.len() {
            let pos1 = notes[i-2].position;
            let pos2 = notes[i-1].position;
            let pos3 = notes[i].position;

            if pos1.distance_to(&pos2) < 0.15 && pos2.distance_to(&pos3) < 0.25 {
                let time_diff1 = notes[i-1].time - notes[i-2].time;
                let time_diff2 = notes[i].time - notes[i-1].time;
                let time_consistency = (time_diff1 - time_diff2).abs() < 0.05;

                if time_consistency {
                    consecutive_jacks += 1;
                    // 长连打序列得分更高
                    jack_score += 1.0 + (consecutive_jacks as f32 * 0.2).min(2.0);
                } else {
                    consecutive_jacks = 0;
                }
            } else {
                consecutive_jacks = 0;
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

        max_speed * 0.6 + avg_speed * 0.4
    }

    fn calculate_coordination_difficulty(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 3 {
            return 0.0;
        }

        let mut coordination_score = 0.0;

        for i in 2..notes.len() {
            let pos_changes = vec![
                notes[i-2].position - notes[i-1].position,
                notes[i-1].position - notes[i].position,
            ];

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

        alternating * 1.2 + stream * 2.5 + chord * 1.8 + jack * 3.0
    }

    fn extract_velocity_features(&self, notes: &[ProcessedNote]) -> Vec<f32> {
        if notes.len() < 2 {
            return vec![0.0; 4];
        }

        const MIN_TIME_DIFF: f32 = 0.01; // 防止除以接近0的时间差
        const MAX_VELOCITY: f32 = 40.0;   // 速度上限
        const MAX_ACCELERATION: f32 = 100.0; // 加速度上限

        let mut velocities = Vec::new();
        let mut accelerations = Vec::new();

        for i in 1..notes.len() {
            let distance = notes[i].position.distance_to(&notes[i-1].position);
            let time_diff = (notes[i].time - notes[i-1].time).max(MIN_TIME_DIFF);

            // 计算速度并限制范围
            let velocity = (distance / time_diff).min(MAX_VELOCITY);
            velocities.push(velocity);

            // 计算加速度
            if i > 1 && velocities.len() >= 2 {
                let prev_velocity = velocities[velocities.len() - 2];
                let time_diff_accel = (notes[i].time - notes[i-1].time).max(MIN_TIME_DIFF);
                let acceleration = ((velocity - prev_velocity) / time_diff_accel).min(MAX_ACCELERATION);
                accelerations.push(acceleration);
            }
        }

        // 安全计算统计值，避免除以0
        let avg_velocity = if !velocities.is_empty() {
            velocities.iter().sum::<f32>() / velocities.len() as f32
        } else {
            0.0
        };

        let max_velocity = if !velocities.is_empty() {
            velocities.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        } else {
            0.0
        };

        let avg_acceleration = if !accelerations.is_empty() {
            accelerations.iter().sum::<f32>() / accelerations.len() as f32
        } else {
            0.0
        };

        let max_acceleration = if !accelerations.is_empty() {
            accelerations.iter().cloned().fold(f32::NEG_INFINITY, |a, b| a.max(b.abs()))
        } else {
            0.0
        };

        vec![avg_velocity, max_velocity, avg_acceleration, max_acceleration]
    }

    fn extract_spatial_features(&mut self, notes: &[ProcessedNote]) -> Vec<f32> {
        if notes.is_empty() {
            return vec![0.0; 6];
        }

        let mut x_positions: Vec<f32> = notes.iter().map(|n| n.position.x).collect();
        let mut y_positions: Vec<f32> = notes.iter().map(|n| n.position.y).collect();

        x_positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        y_positions.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let x_range = if x_positions.is_empty() {
            0.0
        } else {
            x_positions.last().unwrap() - x_positions.first().unwrap()
        };

        let y_range = if y_positions.is_empty() {
            0.0
        } else {
            y_positions.last().unwrap() - y_positions.first().unwrap()
        };

        let center_x = if x_positions.is_empty() {
            0.0
        } else {
            x_positions[x_positions.len() / 2]
        };

        let center_y = if y_positions.is_empty() {
            0.0
        } else {
            y_positions[y_positions.len() / 2]
        };

        let mut avg_direction = Vector2::new(0.0, 0.0);
        if notes.len() > 1 {
            for i in 1..notes.len() {
                avg_direction = avg_direction + (notes[i].position - notes[i - 1].position);
            }
            avg_direction = avg_direction * (1.0 / (notes.len() as f32 - 1.0));
        }

        let spread = (x_range * x_range + y_range * y_range).sqrt();

        let magnitude = if avg_direction.x == 0.0 && avg_direction.y == 0.0 {
            0.0
        } else {
            avg_direction.magnitude()
        };

        vec![
            center_x,
            center_y,
            spread,
            avg_direction.x,
            avg_direction.y,
            magnitude,
        ]
    }
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

    fn clean(&mut self) {
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

    fn update_state(&mut self, new_pos: Vector2, time: f32, success: bool, _note_kind: &NoteKind) {
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
            let recovery = (time_diff * 0.3).min(0.15);
            self.fatigue = (self.fatigue - recovery).max(0.0);
        }

        self.total_actions += 1;
        if success {
            self.success_streak += 1;
            self.confidence = (self.confidence + 0.01).min(1.0);
        } else {
            self.success_streak = 0;
            self.confidence = (self.confidence - 0.02).max(0.1);
        }

        let recent_success_rate = if self.total_actions > 10 {
            self.success_streak as f32 / 10.0_f32.min(self.total_actions as f32)
        } else {
            self.success_streak as f32 / self.total_actions.max(1) as f32
        };

        self.performance_score = recent_success_rate * 0.5 + self.confidence * 0.3 + (1.0 - self.fatigue) * 0.2;

        self.position = new_pos;
        self.last_time = time;
    }
}

impl ExperienceReplay {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, experience: Experience) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    fn sample(&self, n: usize) -> Vec<&Experience> {
        let len = self.buffer.len();
        if len == 0 {
            return Vec::new();
        }
        let n = n.min(len);
        (0..n).map(|_| &self.buffer[fastrand::usize(0..len)]).collect()
    }

    fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl PhiTKAdvancedAI {
    const CURRENT_VERSION: u32 = 1;

    fn clean_model_data(&mut self) {
        let default = Self::new(self.rotation);
        self.main_network.clean();
        self.target_network.clean();
        self.left_hand_state.clean();
        self.right_hand_state.clean();
        if !self.exploration_rate.is_finite() { self.exploration_rate = default.exploration_rate; }
        if !self.discount_factor.is_finite() { self.discount_factor = default.discount_factor; }
        if !self.average_reward.is_finite() { self.average_reward = default.average_reward; }
        if !self.difficulty_adaptation.is_finite() { self.difficulty_adaptation = default.difficulty_adaptation; }
        if !self.learning_momentum.is_finite() { self.learning_momentum = default.learning_momentum; }
        if !self.confidence_threshold.is_finite() { self.confidence_threshold = default.confidence_threshold; }
        if !self.stability_factor.is_finite() { self.stability_factor = 0.85; }
        if !self.hand_switch_penalty.is_finite() { self.hand_switch_penalty = 0.4; }
        if !self.consistency_bonus.is_finite() { self.consistency_bonus = 0.3; }
        if !self.adaptive_learning_rate.is_finite() { self.adaptive_learning_rate = 0.001; }
        if !self.pattern_recognition_strength.is_finite() { self.pattern_recognition_strength = 1.0; }
        if !self.memory_consolidation_rate.is_finite() { self.memory_consolidation_rate = 0.1; }
        self.feature_extractor.difficulty_estimator.base_difficulty = self.feature_extractor.difficulty_estimator.base_difficulty.max(0.0).min(10.0);

        for finger_state in &mut self.finger_states {
            finger_state.clean();
        }
    }

    fn validate_for_serialization(&self) -> bool {
        let float_fields_valid = [
            self.exploration_rate,
            self.discount_factor,
            self.average_reward,
            self.difficulty_adaptation,
            self.learning_momentum,
            self.confidence_threshold
        ].iter().all(|f| f.is_finite());

        let networks_valid = self.main_network.validate() && self.target_network.validate();

        let hands_valid = self.left_hand_state.validate() && self.right_hand_state.validate();

        float_fields_valid && networks_valid && hands_valid
    }

    fn new(rotation: f32) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(32)
            .build()
            .ok().map(Arc::new);

        let rad = rotation.to_radians();
        let game_mode = GameMode::FourFinger;
        let finger_states = Self::init_finger_states(game_mode, rad);

        let mut ai = Self {
            main_network: DeepNeuralNetwork::new(),
            target_network: DeepNeuralNetwork::new(),
            thread_pool: thread_pool.clone(),
            //thread_count: if thread_pool.is_some() { 32 } else { 1 },
            feature_extractor: AdvancedFeatureExtractor::new(),
            experience_replay: ExperienceReplay::new(8000),
            left_hand_state: HandState::new(Hand::Left, Vector2::new(-0.3, 0.0).rotate(rad)),
            right_hand_state: HandState::new(Hand::Right, Vector2::new(0.3, 0.0).rotate(rad)),
            rotation,
            exploration_rate: 0.05,
            discount_factor: 0.95,
            target_update_frequency: 1000,
            total_notes_processed: 0,
            correct_predictions: 0,
            training_episodes: 0,
            average_reward: 0.7,
            difficulty_adaptation: 1.0,
            learning_momentum: 0.9,
            confidence_threshold: 0.7,
            pattern_memory: BTreeMap::new(),
            performance_history: VecDeque::with_capacity(50000),
            version: Self::CURRENT_VERSION,
            last_save_episodes: 0,
            last_update_time: -1.0,
            line_rotations: HashMap::new(),
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

        ai.target_network = DeepNeuralNetwork::new(); // 创建新实例
        ai.target_network.build_architecture();       // 构建相同架构
        ai.target_network.init_gpu_sync();

        ai.warm_thread_pool();
        ai
    }

    fn load_or_create(filepath: &str, rotation: f32) -> Self {
        let path = Path::new(filepath);
        if let Ok(bytes) = fs::read(path) {
            if let Ok(mut ai) = bincode::deserialize::<Self>(&bytes) {
                if ai.validate_for_serialization() {
                    println!("[Model] 从文件加载模型: {}, 训练回合数: {}", filepath, ai.training_episodes);
                    ai.rotation = rotation;
                    ai.update_hand_positions();

                    // 为 main_network 设置状态，强制重新初始化 GPU
                    ai.main_network.initialization_attempted = false;
                    ai.main_network.initialization_failed = false;
                    ai.main_network.gpu_initialized = false;
                    println!("[GPU/CPU SWITCH] Beginning GPU init for main_network...");

                    // 尝试初始化 GPU，失败则回退到 CPU
                    println!("[GPU/CPU SWITCH] Beginning GPU init for main_network...");
                    ai.main_network.init_gpu_sync();
                    
                    if ai.main_network.gpu_initialized {
                        println!("[GPU/CPU SWITCH] main_network GPU initialized successfully.");
                    } else {
                        println!("[GPU/CPU SWITCH] main_network GPU initialization failed, using CPU fallback.");
                    }

                    // 为 target_network 清除所有 GPU 资源并重新初始化
                    ai.target_network.device = None;
                    ai.target_network.queue = None;
                    ai.target_network.batch_size_buffer = None;
                    ai.target_network.matmul_pipeline = None;
                    ai.target_network.activation_pipelinotes = HashMap::new();
                    ai.target_network.activation_bind_group_layouts = HashMap::new();
                    // 清除所有层的GPU缓冲区
                    for layer in &mut ai.target_network.layers {
                        layer.weights_buffer = None;
                        layer.biases_buffer = None;
                        layer.activations_buffer = None;
                    }

                    // 重置 GPU 初始化状态
                    ai.target_network.initialization_attempted = false;
                    ai.target_network.initialization_failed = false;
                    ai.target_network.gpu_initialized = false;

                    println!("[GPU/CPU SWITCH] Beginning GPU init for target_network...");
                    ai.target_network.init_gpu_sync();

                    // 添加验证检查
                    if ai.target_network.gpu_initialized && ai.target_network.device.is_some() && ai.target_network.queue.is_some() {
                        println!("[GPU/CPU SWITCH] target_network GPU initialized successfully.");
                    } else {
                        println!("[GPU/CPU SWITCH] target_network GPU initialization failed, using CPU fallback.");
                    }

                    return ai;
                }
            }
        }

        // 创建新模型时也强制使用 GPU
        let mut ai = Self::new(rotation);

        // 初始化 GPU
        println!("[GPU/CPU SWITCH] Beginning GPU init for new model...");
        ai.main_network.init_gpu_sync();
        ai.target_network.init_gpu_sync();
        println!("[GPU/CPU SWITCH] New model GPU initialized successfully.");

        ai.save_model(filepath);
        ai
    }

    fn save_model(&mut self, filepath: &str) {
        println!("[Model] 尝试保存模型到: {}", filepath);

        self.clean_model_data();
        if !self.validate_for_serialization() {
            eprintln!("[Model Error] 模型验证失败，无法保存");
            return;
        }

        let path = Path::new(filepath);
        if let Ok(data) = bincode::serialize(self) {
            if let Err(e) = fs::write(path, data) {
                eprintln!("[Model Error] 保存模型失败: {:?}", e);
            } else {
                println!("[Model] 模型保存成功: {}", filepath);
            }
        } else {
            eprintln!("[Model Error] 模型序列化失败");
        }
    }

    fn init_finger_states(mode: GameMode, rotation_rad: f32) -> Vec<FingerState> {
        let mut states = Vec::new();

        match mode {
            GameMode::TwoFinger => {
                states.push(FingerState::new(
                    Finger::LeftIndex,
                    Vector2::new(-0.32, 0.0).rotate(rotation_rad)
                ));
                states.push(FingerState::new(
                    Finger::RightIndex,
                    Vector2::new(0.32, 0.0).rotate(rotation_rad)
                ));
            }
            GameMode::FourFinger => {
                states.push(FingerState::new(
                    Finger::LeftIndex,
                    Vector2::new(-0.24, 0.0).rotate(rotation_rad) //左食指
                ));
                states.push(FingerState::new(
                    Finger::LeftMiddle,
                    Vector2::new(-0.38, 0.0).rotate(rotation_rad) //左中指
                ));
                states.push(FingerState::new(
                    Finger::RightIndex,
                    Vector2::new(0.24, 0.0).rotate(rotation_rad) //右食指
                ));
                states.push(FingerState::new(
                    Finger::RightMiddle,
                    Vector2::new(0.38, 0.0).rotate(rotation_rad) //右中指
                ));
            }
        }

        states
    }

    fn update_hand_positions(&mut self) {
        let rad = self.rotation.to_radians();
        self.left_hand_state.position = Vector2::new(-0.3, 0.0).rotate(rad);
        self.right_hand_state.position = Vector2::new(0.3, 0.0).rotate(rad);
    }

    fn light_update_hand_states(&mut self, notes: &[Note]) {
        for note in notes {
            // 在chart.rs中已经进行了坐标转换，这里直接使用转换后的坐标
            let pos = Vector2::new(
                note.object.translation.0.now(),
                note.object.translation.1.now()
            );

            match note.hand {
                Hand::Left => self.left_hand_state.position = pos,
                Hand::Right => self.right_hand_state.position = pos,
                //_ => {} // 未分配的不处理
            }
        }
    }

    fn detect_and_switch_mode(&mut self, notes: &[Note]) {
        if notes.len() < 12 {  // 最小音符数量要求
            return;
        }

        let time_window = notes.last().unwrap().time - notes[0].time;
        if time_window < 1.0 {  //时间窗口要求
            return;
        }

        let note_density = notes.len() as f32 / time_window.max(0.1);

        let mut max_simultaneous = 1;
        let mut current_time = notes[0].time;
        let mut current_group_size = 1;

        for i in 1..notes.len() {
            if (notes[i].time - current_time).abs() < 0.001 {  //同时判定
                current_group_size += 1;
                max_simultaneous = max_simultaneous.max(current_group_size);
            } else {
                current_time = notes[i].time;
                current_group_size = 1;
            }
        }

        // 提高切换阈值
        let should_switch = (note_density > 48.0 && max_simultaneous >= 3) ||  // 提高密度要求
            (note_density > 8.0 && max_simultaneous >= 3) ||    // 同时音符要求更高
            max_simultaneous >= 4;                              // 只有4个以上同时音符才切换

        if should_switch && self.game_mode != GameMode::FourFinger {
            println!("[模式切换] 检测到高密度谱面，切换到四指模式。密度: {:.1}, 最大同时音符: {}",
                     note_density, max_simultaneous);
            self.game_mode = GameMode::FourFinger;
            self.finger_states = Self::init_finger_states(self.game_mode, self.rotation.to_radians());
        } else if !should_switch && self.game_mode != GameMode::TwoFinger {
            // 只有当条件明显不满足时才切换回二指模式
            if note_density < 6.0 && max_simultaneous <= 2 {
                println!("[模式切换] 谱面密度降低，切换回二指模式");
                self.game_mode = GameMode::TwoFinger;
                self.finger_states = Self::init_finger_states(self.game_mode, self.rotation.to_radians());
            }
        }
    }

    fn analyze_and_assign(&mut self, notes: &mut [Note], _config: &Config, bpm_list: &BpmList, line_id: usize) {
        //println!("[开始分析] 线路{} 音符数量:{} 游戏模式:{:?}", line_id, notes.len(), self.game_mode);
        if notes.is_empty() {
            return;
        }
        let mut processed_notes = self.preprocess_notes(notes);
        let simultaneous_groups = self.detect_simultaneous_groups(&processed_notes);
        let mut bpm_list_clone = bpm_list.clone();
        self.detect_and_switch_mode(notes);
        self.assign_simultaneous_groups(&mut processed_notes, &simultaneous_groups, &mut bpm_list_clone, line_id);
        self.ai_assign_single_notes(&mut processed_notes, &simultaneous_groups, &mut bpm_list_clone, line_id);
        self.post_process_assignments(&mut processed_notes);
        self.optimize_jack_pattern(&mut processed_notes);
        self.apply_and_learn(notes, &processed_notes);
        if self.experience_replay.len() >= 64 && self.total_notes_processed % 25 == 0 {
            self.train_network();
        }
        if self.training_episodes % self.target_update_frequency as u64 == 0 {
            self.target_network = self.main_network.clone();
        }
        self.training_episodes += 1;
        self.last_save_episodes += 1;
        if self.last_save_episodes >= 256 {
            println!("Saving episodes to {}", self.last_save_episodes);
            println!("训练回合数，已保存: {}", self.training_episodes);
            self.save_model("phitk_ai_model.bin");
            self.last_save_episodes = 0;
        }
    }

    fn preprocess_notes(&self, notes: &[Note]) -> Vec<ProcessedNote> {
        notes.iter().enumerate().map(|(i, note)| {
            let pos = Vector2::new(
                note.object.translation.0.now(),
                note.object.translation.1.now()
            );

            let difficulty = match note.kind {
                NoteKind::Click => 1.0,
                NoteKind::Drag => 1.3,
                NoteKind::Flick => 1.5,
                NoteKind::Hold { .. } => 1.8,
            };

            let duration = match &note.kind {
                NoteKind::Hold { end_time, .. } => *end_time - note.time,
                _ => 0.1,
            };

            ProcessedNote {
                index: i,
                position: pos,
                time: note.time,
                kind: note.kind.clone(),
                assigned_hand: None,
                confidence: 0.0,
                features: Vec::new(),
                judge: JudgeStatus::NotJudged,
                difficulty,
                duration,
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
                    SIMULTANEOUS_THRESHOLD * 0.5
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

    fn assign_simultaneous_groups(&mut self, notes: &mut [ProcessedNote], groups: &[Vec<usize>], bpm_list: &mut BpmList, line_id: usize) {
        for group in groups {
            if group.len() < 2 {
                continue;
            }

            if self.game_mode == GameMode::TwoFinger && group.len() > 2 {
                let left_notes: Vec<_> = group.iter().filter(|&&i| notes[i].position.x < 0.0).collect();
                let right_notes: Vec<_> = group.iter().filter(|&&i| notes[i].position.x >= 0.0).collect();

                // 如果左右都有，且总数 > 2 → 触发重分配
                if !left_notes.is_empty() && !right_notes.is_empty() {
                    // 计算哪一侧更密集
                    let dominant_side = if right_notes.len() > left_notes.len() {
                        true // right dominant
                    } else {
                        false // left dominant
                    };

                    // 全部分配到密集侧，按 x 排序后交替分配
                    let mut all_indices: Vec<_> = group.iter().copied().collect();
                    all_indices.sort_by(|&a, &b| notes[a].position.x.partial_cmp(&notes[b].position.x).unwrap_or(std::cmp::Ordering::Equal));

                    for (idx, &note_idx) in all_indices.iter().enumerate() {
                        let hand = if dominant_side {
                            // 全在右侧：按排序交替分配（模拟2指在右区）
                            if idx % 2 == 0 { Hand::Right } else { Hand::Left }
                        } else {
                            // 全在左侧
                            if idx % 2 == 0 { Hand::Left } else { Hand::Right }
                        };
                        notes[note_idx].assigned_hand = Some(hand);
                        notes[note_idx].confidence = 0.85;
                        self.recent_assignments.push_back((hand, notes[note_idx].position.x, notes[note_idx].time));
                    }
                    if self.recent_assignments.len() > 50 {
                        self.recent_assignments.pop_front();
                    }
                    continue; // 跳过原有逻辑
                }
            }

            if group.len() == 4 {
                let mut sorted_group: Vec<_> = group.iter()
                    .map(|&i| (i, notes[i].position.x))
                    .collect();
                sorted_group.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for (idx, (note_idx, _)) in sorted_group.iter().enumerate() {
                    let hand = if idx < 2 { Hand::Left } else { Hand::Right };
                    notes[*note_idx].assigned_hand = Some(hand);
                    notes[*note_idx].confidence = 0.9;
                    self.recent_assignments.push_back((hand, notes[*note_idx].position.x, notes[*note_idx].time));
                    if self.recent_assignments.len() > 78 {
                        self.recent_assignments.pop_front();
                    }
                }
                continue;
            }

            let mut sorted_group: Vec<_> = group.iter()
                .map(|&i| (i, notes[i].position.x))
                .collect();
            sorted_group.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let mid_point = sorted_group.len() / 2;

            let leftmost_x = sorted_group[0].1;
            let rightmost_x = sorted_group[sorted_group.len() - 1].1;
            let separation = rightmost_x - leftmost_x;

            if separation > 0.4 && group.len() >= 2 {
                let mut assigned_left = 0;
                let mut assigned_right = 0;

                for (pos, (note_idx, _)) in sorted_group.iter().enumerate() {
                    let hand = if pos < mid_point { Hand::Left } else { Hand::Right };

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
                            let features = self.feature_extractor.extract_features(&notes[*note_idx..*note_idx + 1], 1, bpm_list);
                            let ai_decision = self.make_ai_decision(&features, &notes[*note_idx], line_id);
                            notes[*note_idx].assigned_hand = Some(ai_decision.0);
                        }
                    }
                    notes[*note_idx].confidence = 0.9;
                    self.recent_assignments.push_back((hand, notes[*note_idx].position.x, notes[*note_idx].time));
                    if self.recent_assignments.len() > 50 {
                        self.recent_assignments.pop_front();
                    }
                }
            } else {
                // 分离度较小的情况，使用AI决策
                for &note_idx in group {
                    let start_idx = note_idx.saturating_sub(8);
                    let end_idx = (note_idx + 8).min(notes.len());
                    let context = &notes[start_idx..end_idx];

                    let features = self.feature_extractor.extract_features(
                        &notes[note_idx..note_idx + 1], // 单个音符
                        1,
                        bpm_list
                    );

                    // 新增：提取位置上下文特征
                    let position_context = self.feature_extractor.extract_position_context_features(&notes[note_idx], context);

                    // 合并特征
                    let mut all_features = features;
                    all_features.extend(position_context);

                    let ai_decision = self.make_ai_decision(&all_features, &notes[note_idx], line_id);

                    notes[note_idx].assigned_hand = Some(ai_decision.0);
                    notes[note_idx].confidence = ai_decision.1;
                }
            }

            self.update_hand_states_for_group(notes, group);
        }
    }

    fn post_process_assignments(&mut self, notes: &mut [ProcessedNote]) {
        self.smooth_hand_transitions(notes);
        self.validate_physical_feasibility(notes);
        self.optimize_recognized_patterns(notes);

        self.ensure_alternating_pattern(notes);
        self.correct_position_mismatches(notes);
        self.check_hand_consistency(notes);
        self.correct_position_mismatches(notes);
    }

    fn correct_position_mismatches(&self, notes: &mut [ProcessedNote]) {
        for note in notes {
            if let Some(hand) = note.assigned_hand {
                let mismatch = match hand {
                    Hand::Left => note.position.x > 0.0, // 左手分配但位置在右侧
                    Hand::Right => note.position.x < 0.0, // 右手分配但位置在左侧
                };

                if mismatch {
                    note.assigned_hand = Some(if note.position.x < 0.0 {
                        Hand::Left
                    } else {
                        Hand::Right
                    });
                    note.confidence = 0.7;
                }
            }
        }
    }

    fn ensure_alternating_pattern(&mut self, notes: &mut [ProcessedNote]) {
        const MAX_CONSECUTIVE: usize = 3;
        const TIME_THRESHOLD: f32 = 0.3;

        let mut consecutive_count = 0;
        let mut last_hand = None;

        let mut indices_to_switch = Vec::new();

        for (i, note) in notes.iter().enumerate() {
            if let Some(current_hand) = note.assigned_hand {
                if last_hand != Some(current_hand) {
                    consecutive_count = 0;
                    last_hand = Some(current_hand);
                    continue;
                }

                consecutive_count += 1;
                if consecutive_count > MAX_CONSECUTIVE {
                    let prev_alt_time = notes[..i]
                        .iter()
                        .rev()
                        .find(|n| n.assigned_hand != Some(current_hand))
                        .map(|n| n.time);

                    if let Some(prev_time) = prev_alt_time {
                        if note.time - prev_time < TIME_THRESHOLD * 2.0 {
                            indices_to_switch.push(i);
                            consecutive_count = 0;
                        }
                    }
                }
            }
        }

        for i in indices_to_switch {
            if let Some(current_hand) = notes[i].assigned_hand {
                notes[i].assigned_hand = Some(match current_hand {
                    Hand::Left => Hand::Right,
                    Hand::Right => Hand::Left,
                });
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

        if !left_positions.is_empty() {
            let avg_pos = left_positions.iter().fold(Vector2::new(0.0, 0.0), |acc, &pos| acc + pos) * (1.0 / left_positions.len() as f32);
            let success = notes[group[0]].judge == JudgeStatus::Judged;
            self.left_hand_state.update_state(avg_pos, group_time, success, &notes[group[0]].kind);
        }

        if !right_positions.is_empty() {
            let avg_pos = right_positions.iter().fold(Vector2::new(0.0, 0.0), |acc, &pos| acc + pos) * (1.0 / right_positions.len() as f32);
            let success = notes[group[0]].judge == JudgeStatus::Judged;
            self.right_hand_state.update_state(avg_pos, group_time, success, &notes[group[0]].kind);
        }
    }

    fn make_ai_decision_with_density(
        &mut self,
        features: &[f32],
        note: &ProcessedNote,
        _line_id: usize,
        dominant_side: Option<Hand>,
    ) -> (Hand, f32, Finger) {
        // 如果存在显著密度偏移，直接使用排序交替逻辑（2指模式）
        if self.game_mode == GameMode::TwoFinger && dominant_side.is_some() {
            // 在偏移区域，两只手都进该侧，按 x 排序交替分配
            let hand = match dominant_side.unwrap() {
                Hand::Left => {
                    // 在左区：x 越小越靠前，先左后右
                    if note.position.x <= -0.15 {
                        Hand::Left
                    } else {
                        Hand::Right
                    }
                }
                Hand::Right => {
                    // 在右区：x 越大越靠前，先右后左
                    if note.position.x >= 0.15 {
                        Hand::Right
                    } else {
                        Hand::Left
                    }
                }
            };
            // 返回高置信度（因为这是策略性分配）
            return (hand, 0.88, match hand {
                Hand::Left => Finger::LeftIndex,
                Hand::Right => Finger::RightIndex,
            });
        }

        // 否则，走原有 AI 决策逻辑
        self.make_ai_decision(features, note, _line_id)
    }

    fn ai_assign_single_notes(&mut self, notes: &mut [ProcessedNote], simultaneous_groups: &[Vec<usize>], bpm_list: &mut BpmList, line_id: usize) {
        let assigned_indices: std::collections::HashSet<usize> = simultaneous_groups.iter().flatten().copied().collect();
        const CONTEXT_WINDOW: usize = 64;
        const BATCH_SIZE: usize = 24;

        let mut unassigned_indices = Vec::new();
        for i in 0..notes.len() {
            if !assigned_indices.contains(&i) {
                unassigned_indices.push(i);
            }
        }

        for batch_indices in unassigned_indices.chunks(BATCH_SIZE) {
            let mut features_batch: Vec<Vec<f32>> = Vec::with_capacity(batch_indices.len());
            let mut note_data: Vec<(usize, f32, f32, JudgeStatus, NoteKind)> = Vec::with_capacity(batch_indices.len());

            let results: Vec<_> = if let Some(pool) = &self.thread_pool {
                pool.install(|| {
                    batch_indices.par_iter().map(|&idx| {
                        let start_idx = idx.saturating_sub(CONTEXT_WINDOW / 2);
                        let end_idx = (idx + CONTEXT_WINDOW / 2 + 1).min(notes.len());
                        let context = &notes[start_idx..end_idx];

                        // 计算局部密度偏移（保留原有逻辑）
                        let mut left_count = 0;
                        let mut right_count = 0;
                        for note in context {
                            if note.position.x < -0.1 {
                                left_count += 1;
                            } else if note.position.x > 0.1 {
                                right_count += 1;
                            }
                        }
                        let dominant_side = if left_count > 0 && right_count > 0 {
                            if (right_count as f32) / (left_count as f32) >= 2.0 {
                                Some(Hand::Right)
                            } else if (left_count as f32) / (right_count as f32) >= 2.0 {
                                Some(Hand::Left)
                            } else {
                                None
                            }
                        } else if left_count > 0 {
                            Some(Hand::Left)
                        } else if right_count > 0 {
                            Some(Hand::Right)
                        } else {
                            None
                        };

                        // 提取基础特征
                        let mut feature_extractor = self.feature_extractor.clone();
                        let features = feature_extractor.extract_features(&context, CONTEXT_WINDOW, bpm_list);

                        // 新增：提取位置上下文特征
                        let position_context = feature_extractor.extract_position_context_features(&notes[idx], &context);

                        // 合并特征
                        let mut all_features = features;
                        all_features.extend(position_context);

                        (
                            all_features,  // 使用合并后的特征
                            idx,
                            notes[idx].position.x,
                            notes[idx].time,
                            notes[idx].judge.clone(),
                            notes[idx].kind.clone(),
                            dominant_side,
                        )
                    }).collect()
                })
            } else {
                batch_indices.iter().map(|&idx| {
                    let start_idx = idx.saturating_sub(CONTEXT_WINDOW / 2);
                    let end_idx = (idx + CONTEXT_WINDOW / 2 + 1).min(notes.len());
                    let context = &notes[start_idx..end_idx];

                    // 计算局部密度偏移（保留原有逻辑）
                    let mut left_count = 0;
                    let mut right_count = 0;
                    for note in context {
                        if note.position.x < -0.1 {
                            left_count += 1;
                        } else if note.position.x > 0.1 {
                            right_count += 1;
                        }
                    }
                    let dominant_side = if left_count > 0 && right_count > 0 {
                        if (right_count as f32) / (left_count as f32) >= 2.0 {
                            Some(Hand::Right)
                        } else if (left_count as f32) / (right_count as f32) >= 2.0 {
                            Some(Hand::Left)
                        } else {
                            None
                        }
                    } else if left_count > 0 {
                        Some(Hand::Left)
                    } else if right_count > 0 {
                        Some(Hand::Right)
                    } else {
                        None
                    };

                    // 提取基础特征
                    let features = self.feature_extractor.extract_features(&context, CONTEXT_WINDOW, bpm_list);

                    // 新增：提取位置上下文特征
                    let position_context = self.feature_extractor.extract_position_context_features(&notes[idx], &context);

                    // 合并特征
                    let mut all_features = features;
                    all_features.extend(position_context);

                    (
                        all_features,  // 使用合并后的特征
                        idx,
                        notes[idx].position.x,
                        notes[idx].time,
                        notes[idx].judge.clone(),
                        notes[idx].kind.clone(),
                        dominant_side,
                    )
                }).collect()
            };

            let feature_slices: Vec<&[f32]> = results.iter().map(|r| r.0.as_slice()).collect();
            let outputs = if self.main_network.gpu_initialized {
                self.main_network.gpu_forward_batch(&feature_slices, results.len())
            } else {
                results.iter().map(|r| self.main_network.forward(&r.0)).collect()
            };

            for (i, output) in outputs.iter().enumerate() {
                let (features, note_idx, position_x, time, judge, kind, dominant_side) = &results[i];

                // 先克隆当前note用于AI决策（避免借用冲突）
                let current_note = notes[*note_idx].clone();

                // 使用增强的特征进行决策
                let ai_decision = self.make_ai_decision_with_density(
                    output, &current_note, line_id, *dominant_side
                );

                // 然后获取可变引用进行修改
                let note = &mut notes[*note_idx];
                note.features = features.clone();
                note.assigned_hand = Some(ai_decision.0);
                note.confidence = ai_decision.1;

                self.recent_assignments.push_back((ai_decision.0, *position_x, *time));
                if self.recent_assignments.len() > 78 {
                    self.recent_assignments.pop_front();
                }

                if let Some(finger_state) = self.finger_states.iter_mut().find(|fs| fs.finger == ai_decision.2) {
                    let success = *judge == JudgeStatus::Judged;
                    finger_state.update_state(note.position, note.time, success, &kind);
                }

                let success = *judge == JudgeStatus::Judged;
                match ai_decision.0 {
                    Hand::Left => self.left_hand_state.update_state(note.position, note.time, success, &kind),
                    Hand::Right => self.right_hand_state.update_state(note.position, note.time, success, &kind),
                }

                self.record_experience(&features, note, ai_decision.0, ai_decision.1);
            }
        }
    }

    fn make_ai_decision(&mut self, features: &[f32], note: &ProcessedNote, line_id: usize) -> (Hand, f32, Finger) {
        let token = TOTAL_TOKENS_USED.fetch_add(1, Ordering::Relaxed);
        if token % 500 == 0 {
            println!("Decision数量: {}", token + 1);
        }

        // AI推理
        let output = self.main_network.forward(features);

        let left_prob = output.get(0).copied().unwrap_or(0.5);
        let right_prob = output.get(1).copied().unwrap_or(0.5);
        let confidence = output.get(3).copied().unwrap_or(0.7);

        let position_hand = if left_prob > right_prob { Hand::Left } else { Hand::Right };
        let position_confidence = confidence;
        
        let mut finger_scores: Vec<(Finger, f32)> = self.finger_states.iter()
            .map(|fs| {
                let score = fs.calculate_assignment_score(
                    note.position, note.time, note.difficulty,
                    note.time, &note.kind, note.duration
                );
                (fs.finger, score)
            })
            .collect();

        finger_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let best_finger = finger_scores[0].0;
        let finger_hand = best_finger.to_hand();

        // 融合位置决策和手指决策
        let final_hand = if position_confidence > 0.7 {
            position_hand
        } else {
            finger_hand
        };

        let final_confidence = (position_confidence * 0.6 + finger_scores[0].1 * 0.4)
            .clamp(0.2, 0.95);

        (final_hand, final_confidence, best_finger)
    }

    fn record_experience(&mut self, features: &[f32], note: &ProcessedNote, chosen_hand: Hand, confidence: f32) {
        //println!("Recording experience, replay size now: {}", self.experience_replay.len());
        let reward = self.calculate_reward(note, chosen_hand, confidence);
        let feature_diversity = features.iter().map(|&x| (x - 0.5).abs()).sum::<f32>() / features.len() as f32;
        let priority = if feature_diversity > 0.3 { 2.0 } else { 1.0 };

        let experience = Experience {
            state: features.to_vec(),
            action: if chosen_hand == Hand::Left { 0 } else { 1 },
            reward: reward * priority,
            next_state: features.to_vec(),
            done: false,
            timestamp: note.time,
        };

        self.experience_replay.push(experience);
    }

    fn calculate_reward(&self, note: &ProcessedNote, chosen_hand: Hand, confidence: f32) -> f32 {
        let mut reward = 0.0;

        let ideal_hand = if note.position.x < 0.0 { Hand::Left } else { Hand::Right };
        let position_reward = if chosen_hand == ideal_hand {
            1.0
        } else {
            -0.8
        };

        reward += position_reward;

        let difficulty_bonus = (note.difficulty - 1.0).max(0.0) * 0.3;
        reward += difficulty_bonus;

        let noise = (fastrand::f32() - 0.5) * 0.1; // 减小噪声
        reward += noise;

        reward.clamp(-1.0, 1.5) // 调整范围
    }

    fn smooth_hand_transitions(&self, notes: &mut [ProcessedNote]) {
        if notes.len() < 3 {
            return;
        }

        for i in 1..notes.len() - 1 {
            if let (Some(prev_hand), Some(curr_hand), Some(next_hand)) = (notes[i - 1].assigned_hand, notes[i].assigned_hand, notes[i + 1].assigned_hand) {
                if curr_hand != prev_hand && curr_hand != next_hand && prev_hand == next_hand {
                    let time_gap_prev = notes[i].time - notes[i - 1].time;
                    let time_gap_next = notes[i + 1].time - notes[i].time;

                    if time_gap_prev > 0.15 && time_gap_next > 0.15 {
                        let position_reasonable = match prev_hand {
                            Hand::Left => notes[i].position.x < 0.3,
                            Hand::Right => notes[i].position.x > -0.3,
                        };

                        if position_reasonable && notes[i].confidence < 0.8 {
                            notes[i].assigned_hand = Some(prev_hand);
                            notes[i].confidence = 0.75;
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
            if let (Some(prev_hand), Some(curr_hand)) = (notes[i - 1].assigned_hand, notes[i].assigned_hand) {
                if prev_hand == curr_hand {
                    let distance = notes[i].position.distance_to(&notes[i - 1].position);
                    let time_diff = notes[i].time - notes[i - 1].time;

                    if time_diff > 0.001 {
                        let required_speed = distance / time_diff;

                        if required_speed > MAX_SPEED || time_diff < MIN_TIME_GAP {
                            let other_hand = if curr_hand == Hand::Left { Hand::Right } else { Hand::Left };

                            let switch_reasonable = match other_hand {
                                Hand::Left => notes[i].position.x < 0.3,
                                Hand::Right => notes[i].position.x > -0.3,
                            };

                            if switch_reasonable {
                                notes[i].assigned_hand = Some(other_hand);
                                notes[i].confidence = 0.6;
                            }
                        }
                    }
                }
            }
        }
    }

    fn optimize_recognized_patterns(&mut self, notes: &mut [ProcessedNote]) {
        let mut i = 0;
        while i < notes.len() {
            let window_end = (i + 8).min(notes.len());
            let window = &notes[i..window_end];

            if window.len() < 3 {
                i += 1;
                continue;
            }

            let alternating_score = self.feature_extractor.detect_alternating_pattern(window);
            let stream_score = self.feature_extractor.detect_stream_pattern(window);
            let chord_score = self.feature_extractor.detect_chord_pattern(window);
            let jack_score = self.feature_extractor.detect_jack_pattern(window);
            let crossing_score = self.detect_crossing_pattern(window);

            if alternating_score > 0.7 {
                self.optimize_alternating_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else if stream_score > 0.6 {
                self.optimize_stream_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else if chord_score > 0.5 {
                self.optimize_chord_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else if jack_score > 0.4 {
                self.optimize_jack_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else if crossing_score > 0.2 {
                self.optimize_crossing_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else {
                i += 1;
            }
        }
    }

    fn optimize_jack_pattern(&mut self, notes: &mut [ProcessedNote]) {
        if notes.len() < 3 {
            return;
        }
        let avg_x: f32 = notes.iter().map(|n| n.position.x).sum::<f32>() / notes.len() as f32;
        let mut current_hand = if avg_x < 0.0 { Hand::Left } else { Hand::Right };

        for (index, note) in notes.iter_mut().enumerate() {
            let position_based_hand = if note.position.x < 0.0 { Hand::Left } else { Hand::Right };
            if index > 0 && position_based_hand != current_hand {
                note.assigned_hand = Some(position_based_hand);
                current_hand = position_based_hand;
            } else {
                note.assigned_hand = Some(current_hand);
                current_hand = match current_hand {
                    Hand::Left => Hand::Right,
                    Hand::Right => Hand::Left,
                };
            }
            note.confidence = 0.85;
        }
    }

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

    fn optimize_stream_pattern(&self, notes: &mut [ProcessedNote]) {
        if notes.is_empty() {
            return;
        }

        let avg_x = notes.iter().map(|n| n.position.x).sum::<f32>() / notes.len() as f32;
        let dominant_hand = if avg_x < 0.0 { Hand::Left } else { Hand::Right };

        let mut can_use_same_hand = true;
        for i in 1..notes.len() {
            let distance = notes[i].position.distance_to(&notes[i - 1].position);
            let time_diff = notes[i].time - notes[i - 1].time;

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
        let mut time_groups = HashMap::new();

        for (i, note) in notes.iter().enumerate() {
            let time_key = (note.time * 20.0).round() as i32;
            time_groups.entry(time_key).or_insert(Vec::new()).push(i);
        }

        for indices in time_groups.values() {
            if indices.len() > 1 {
                let mut sorted_indices = indices.clone();
                sorted_indices.sort_by(|&a, &b| notes[a].position.x.partial_cmp(&notes[b].position.x).unwrap_or(std::cmp::Ordering::Equal));

                let mid = sorted_indices.len() / 2;
                for (pos, &idx) in sorted_indices.iter().enumerate() {
                    notes[idx].assigned_hand = Some(if pos < mid { Hand::Left } else { Hand::Right });
                    notes[idx].confidence = 0.9;
                }
            }
        }
    }

    fn detect_crossing_pattern(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 2 {
            return 0.0;
        }
        let mut crossing_score = 0.0;
        let mut consecutive_crossings = 0;
        let mut max_consecutive = 0;
        
        for i in 1..notes.len() {
            let prev_x = notes[i-1].position.x;
            let curr_x = notes[i].position.x;
            // 检查是否从正到负或负到正
            if prev_x * curr_x < 0.0 {
                consecutive_crossings += 1;
                crossing_score += 1.0;
            } else {
                max_consecutive = max_consecutive.max(consecutive_crossings);
                consecutive_crossings = 0;
            }
        }
        max_consecutive = max_consecutive.max(consecutive_crossings);
        
        // 如果有连续的跨越，给予更高的分数
        let base_score = crossing_score / (notes.len() - 1) as f32;
        if max_consecutive >= 2 {
            // 连续跨越序列，提高检测分数
            (base_score * 1.5).min(1.0)
        } else {
            base_score
        }
    }

    fn optimize_crossing_pattern(&mut self, notes: &mut [ProcessedNote]) {
        if notes.is_empty() {
            return;
        }
        
        // 检测是否为真正的跨越序列（从正到负或负到正的连续序列）
        let mut is_crossing_sequence = false;
        let mut crossing_start = 0;
        let mut crossing_end = 0;
        
        // 查找跨越序列的开始和结束
        for i in 0..notes.len() {
            if i > 0 {
                let prev_x = notes[i-1].position.x;
                let curr_x = notes[i].position.x;
                // 检查是否发生跨越（符号改变）
                if prev_x * curr_x < 0.0 {
                    if !is_crossing_sequence {
                        is_crossing_sequence = true;
                        crossing_start = i.saturating_sub(1);
                    }
                    crossing_end = i;
                } else if is_crossing_sequence {
                    // 如果跨越序列中断，结束检测
                    break;
                }
            }
        }
        
        if is_crossing_sequence && crossing_end > crossing_start {
            // 确定起始手：根据跨越序列开始前的音符位置
            let start_hand = if notes[crossing_start].position.x < 0.0 { Hand::Left } else { Hand::Right };
            let mut current_hand = start_hand;
            
            // 处理跨越序列之前的音符
            for i in 0..=crossing_start {
                notes[i].assigned_hand = Some(start_hand);
                notes[i].confidence = 0.85;
            }
            
            // 处理跨越序列：使用起始手完成整个跨越
            for i in (crossing_start + 1)..=crossing_end {
                notes[i].assigned_hand = Some(start_hand);
                notes[i].confidence = 0.8;
            }
            
            // 跨越完成后，根据最后一个音符的位置决定是否切换回默认手
            let last_note_x = notes[crossing_end].position.x;
            let ideal_hand_after_crossing = if last_note_x < 0.0 { Hand::Left } else { Hand::Right };
            
            // 如果跨越后的理想手与起始手不同，则切换
            if ideal_hand_after_crossing != start_hand {
                current_hand = ideal_hand_after_crossing;
            }
            
            // 处理跨越序列之后的音符
            for i in (crossing_end + 1)..notes.len() {
                notes[i].assigned_hand = Some(current_hand);
                notes[i].confidence = 0.85;
            }
        } else {
            // 非跨越序列，使用原有逻辑
            let start_hand = if notes[0].position.x < 0.0 { Hand::Left } else { Hand::Right };
            let mut current_hand = start_hand;
            for note in notes.iter_mut() {
                note.assigned_hand = Some(current_hand);
                note.confidence = 0.8;
                // 如果位置与手部相反且距离较大，考虑切换
                if note.position.x * current_hand.sign() < -2.4 {
                    current_hand = match current_hand {
                        Hand::Left => Hand::Right,
                        Hand::Right => Hand::Left,
                    };
                }
            }
        }
    }

    fn check_hand_consistency(&self, notes: &mut [ProcessedNote]) {
        const MAX_CONSECUTIVE_SAME_HAND: usize = 5;
        let mut consecutive_count = 0;
        let mut last_hand = None;
        for note in notes.iter_mut() {
            if let Some(hand) = note.assigned_hand {
                if last_hand == Some(hand) {
                    consecutive_count += 1;
                } else {
                    consecutive_count = 1;
                    last_hand = Some(hand);
                }
                if consecutive_count > MAX_CONSECUTIVE_SAME_HAND {
                    // 强制切换手
                    note.assigned_hand = Some(match hand {
                        Hand::Left => Hand::Right,
                        Hand::Right => Hand::Left,
                    });
                    note.confidence = 0.6;
                    consecutive_count = 1;
                    last_hand = note.assigned_hand;
                }
            }
        }
    }

    fn apply_and_learn(&mut self, original_notes: &mut [Note], processed_notes: &[ProcessedNote]) {
        let _timestamp = if !processed_notes.is_empty() {
            processed_notes.last().unwrap().time
        } else {
            0.0
        };
        let mut correct_predictions = 0;
        let mut total_predictions = 0;

        for (i, processed) in processed_notes.iter().enumerate() {
            if let Some(hand) = processed.assigned_hand {
                original_notes[i].hand = hand;

                let success = match processed.assigned_hand {
                    Some(Hand::Left) => processed.position.x < 0.0,
                    Some(Hand::Right) => processed.position.x >= 0.0,
                    None => false,
                };

                total_predictions += 1;
                if success {
                    correct_predictions += 1;
                }
            }
        }

        self.total_notes_processed += original_notes.len() as u64;
        self.correct_predictions += correct_predictions;

        let current_accuracy = if total_predictions > 0 {
            correct_predictions as f32 / total_predictions as f32
        } else {
            1.0
        };

        self.average_reward = self.average_reward * 0.95 + current_accuracy * 0.05;

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

        if self.feature_extractor.detect_alternating_pattern(notes) > 0.6 { patterns.push("alternating".to_string()); }
        if self.feature_extractor.detect_stream_pattern(notes) > 0.6 { patterns.push("stream".to_string()); }
        if self.feature_extractor.detect_chord_pattern(notes) > 0.5 { patterns.push("chord".to_string()); }
        if self.feature_extractor.detect_jack_pattern(notes) > 0.5 { patterns.push("jack".to_string()); }

        patterns
    }

    fn adapt_parameters(&mut self, current_accuracy: f32) {
        debug_assert!((0.0..=1.0).contains(&current_accuracy));

        self.average_reward = 0.99 * self.average_reward + 0.01 * current_accuracy;
        self.average_reward = self.average_reward.clamp(0.0, 1.0);

        let lr = self.main_network.learning_rate;
        let new_lr = if current_accuracy < 0.3 {
            (lr * 1.2).clamp(0.0005, 0.01)   
        } else if current_accuracy < self.average_reward {
            (lr * 1.05).clamp(0.0005, 0.01)
        } else {
            (lr * 0.98).max(0.0001)
        };
        self.main_network.learning_rate = new_lr;

        self.exploration_rate = (0.3 - 0.25 * current_accuracy).clamp(0.05, 0.3);

        if current_accuracy > 0.85 {
            self.confidence_threshold = (self.confidence_threshold + 0.01).min(0.9);
        } else if current_accuracy < 0.65 {
            self.confidence_threshold = (self.confidence_threshold - 0.01).max(0.5);
        }
    }

    fn train_network(&mut self) {
        let batch_size = 24;
        let experiences: Vec<_> = self.experience_replay.sample(batch_size).into_iter().cloned().collect();
        //println!("Training network with experience replay size: {}, sampled: {}", self.experience_replay.len(), experiences.len());
        let mut training_data = Vec::with_capacity(batch_size);

        for exp in &experiences {
            let target_value = (exp.reward + self.discount_factor * self.estimate_future_value(&exp.next_state)).clamp(0.0, 1.0);

            let normalized_reward = (exp.reward.clamp(-1.0, 1.0) + 1.0) / 2.0;

            let mut target_output = vec![0.5, 0.5, 1.0, normalized_reward];
            if exp.action < target_output.len() {
                target_output[exp.action] = target_value;
            }
            training_data.push((exp.state.clone(), target_output));
        }
       // println!("Training network with experience replay size: {}", self.experience_replay.len());

        println!("网络训练完成: Epoch {}, 学习率: {:.6}, 探索率: {:.3}, 准确率: {:.3}",
                 self.main_network.epoch_count,
                 self.main_network.learning_rate,
                 self.exploration_rate,
                 self.average_reward);

        self.main_network.train(&training_data);
    }

    fn estimate_future_value(&mut self, state: &[f32]) -> f32 {
        let output = self.target_network.forward(state);
        output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

    pub fn warm_thread_pool(&self) {
        if let Some(pool) = &self.thread_pool {
            pool.install(|| {
                let _ : Vec<u8> = (0..1).into_par_iter().map(|x| (x*2) as u8).collect();
            });
        }
    }
}

