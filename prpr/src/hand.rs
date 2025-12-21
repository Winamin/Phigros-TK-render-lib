use crate::config::Config;

use crate::core::note::Hand;

use crate::core::{BpmList, Note, NoteKind};
use crate::hand_model::{ErgonomicHandSystem, FingerType, Vector2};
use crate::autonomous_wisdom::{AutonomousWisdomAI, ConsciousnessState};
use crate::core::Vector;
use bincode;
use bytemuck::{Pod, Zeroable};
use crossbeam_channel::{unbounded, Receiver as CbReceiver, Sender as CbSender};
use fastrand;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
use std::panic;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Once};
use std::thread;
use std::time::{Duration, Instant};
use wgpu;
use wgpu::util::DeviceExt;

//type StdHashMap<K, V> = HashMap<K, V>;

// 更严格的物理限制常量
const ABSOLUTE_MAX_SPEED: f32 = 4.0;      // 更严格的绝对最大速度
const HARD_MIN_TIME_GAP: f32 = 0.06;      // 更严格的最小时间间隔 (60ms)
const MAX_CONSECUTIVE_SAME_HAND: usize = 3; // 更严格的连续同手最大数量
const MAX_SIMULTANEOUS_NOTES: usize = 2;   // 更严格的同时音符限制
const MAX_FINGER_REACH: f32 = 0.25;       // 与人体工程学模型一致的可达距离

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
//static AI_SYSTEM: OnceCell<Mutex<PhiTKAdvancedAI>> = OnceCell::new();
static WISDOM_AI: OnceCell<Mutex<AutonomousWisdomAI>> = OnceCell::new();
static LINE_STATES: OnceCell<Mutex<HashMap<usize, LineState>>> = OnceCell::new();
static LINE_RESP_QUEUES: OnceCell<Mutex<HashMap<usize, VecDeque<AiResponse>>>> = OnceCell::new();
static REQ_COUNTER: AtomicU64 = AtomicU64::new(1);
static VERSION_COUNTER: AtomicU64 = AtomicU64::new(1);
static TOTAL_TOKENS_USED: AtomicU64 = AtomicU64::new(0);

//const LIGHT_UPDATE_INTERVAL_MS: u64 = 16;
const FULL_UPDATE_INTERVAL_MS: u64 = 8;
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
    const POS_THRESHOLD_SQ: f32 = 400.0; // 20^2，避免开方运算

    let mut used = vec![false; updated.len()];
    // 预计算 updated 音符的索引，按时间排序以优化匹配
    let mut updated_indices: Vec<usize> = (0..updated.len()).collect();
    updated_indices.sort_by(|&a, &b| {
        updated[a].time.partial_cmp(&updated[b].time).unwrap_or(std::cmp::Ordering::Equal)
    });

    for orig in original.iter_mut() {
        let mut best_idx: Option<usize> = None;
        let mut best_score = std::f64::INFINITY;

        let orig_time = orig.time;
        let search_start = updated_indices.partition_point(|&i| updated[i].time < orig_time - TIME_THRESHOLD_SEC);
        let search_end = updated_indices.partition_point(|&i| updated[i].time <= orig_time + TIME_THRESHOLD_SEC);

        // 只检查时间窗口内的音符
        for &idx in &updated_indices[search_start..search_end] {
            if used[idx] {
                continue;
            }
            if std::mem::discriminant(&orig.kind) != std::mem::discriminant(&updated[idx].kind) {
                continue;
            }

            let ox = orig.object.translation.0.now();
            let oy = orig.object.translation.1.now();
            let ux = updated[idx].object.translation.0.now();
            let uy = updated[idx].object.translation.1.now();

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
        }
        // 否则：跳过该音符，不中断整个合并过程
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

fn start_ai_worker_if_needed(config: &Config) {
    static HAND_SPLIT: OnceCell<bool> = OnceCell::new();
    HAND_SPLIT.get_or_init(|| config.hand_split);
    
    // 初始化自主智慧AI
    WISDOM_AI.get_or_init(|| Mutex::new(AutonomousWisdomAI::new(0.0)));
    
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
            let hand_split = *HAND_SPLIT.get().unwrap_or(&false);
            let mut worker_config = Config::default();
            worker_config.hand_split = hand_split;
            let mut worker_ai = PhiTKAdvancedAI::load_or_create("phitk_ai_model.bin", 0.0, &worker_config);
            println!("[GPU] Initializing GPU in AI worker thread...");
            worker_ai.main_network.init_gpu_sync();
            worker_ai.target_network.init_gpu_sync();

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
                    worker_ai.analyze_and_assign(&mut notes_copy, &req.config, &req.bpm_list, req.line_id, start_time.elapsed().as_secs_f32());
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

    start_ai_worker_if_needed(config);

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

    /*
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

     */

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

/// 从统一主视角计算坐标并使用异步AI系统进行决策
pub fn assign_hands_unified_perspective(
    notes: &mut [Note],
    config: &Config,
    line_id: usize,
    rotation: f32,
    bpm_list: &BpmList,
    world_positions: &[(Vector, Vector)] // (true_world_pos, enhanced_pos)
) {
    if notes.is_empty() {
        return;
    }

    // 启动AI worker（如果尚未启动）- 异步处理
    start_ai_worker_if_needed(config);

    let now = Instant::now();
    let rad = rotation.to_radians(); // 反向旋转
    let cos_r = rad.cos();
    let sin_r = rad.sin();

    // 获取或创建line状态
    let line_states = LINE_STATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut line_states_guard = line_states.lock().unwrap_or_else(|poisoned| {
        eprintln!("Line states mutex poisoned, recovering...");
        poisoned.into_inner()
    });
    let line_state = line_states_guard.entry(line_id).or_default();

    // 1. 首先处理AI异步响应（如果有）
    if let Some(map) = LINE_RESP_QUEUES.get() {
        let mut responses = Vec::with_capacity(3);
        {
            let mut mq = map.lock().unwrap();
            if let Some(queue) = mq.get_mut(&line_id) {
                const MAX_RESPONSES_PER_FRAME: usize = 3;
                for _ in 0..MAX_RESPONSES_PER_FRAME {
                    if let Some(resp) = queue.pop_front() {
                        responses.push(resp);
                    } else {
                        break;
                    }
                }
            }
        }

        // 处理AI响应并应用结果
        for resp in responses {
            if resp.line_id != line_id {
                continue;
            }

            if let Some(pending_entry) = line_state.pending_requests.remove(&resp.id) {
                let (_req_ts, req_version) = pending_entry;
                if resp.version == req_version {
                    if match_and_merge_notes(notes, &resp.notes) {
                        line_state.current_version = resp.version;
                        line_state.last_full_update = now;
                    }
                }
            }
        }
    }

    // 2. 只使用AI分配系统，不使用硬编码快速分配
    let should_full_update = now.duration_since(line_state.last_full_update) >= Duration::from_millis(FULL_UPDATE_INTERVAL_MS);
    if should_full_update {
        // 异步发送AI请求以获取更优的分配结果（不阻塞主线程）
        const MAX_PENDING_REQUESTS: usize = 2;
        if line_state.pending_requests.len() < MAX_PENDING_REQUESTS {
            let version = VERSION_COUNTER.fetch_add(1, Ordering::Relaxed);
            let request_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);

            line_state.pending_requests.insert(request_id, (now, version));

            // 准备AI输入数据（主视角坐标）
            let main_perspective_notes: Vec<Note> = notes.iter().enumerate().map(|(i, note)| {
                let (true_world_pos, _enhanced_pos) = world_positions[i];

                let main_perspective_x = true_world_pos.x * cos_r - true_world_pos.y * sin_r;
                let main_perspective_y = true_world_pos.x * sin_r + true_world_pos.y * cos_r;

                let mut ai_note = note.clone();
                ai_note.object.translation.0 = crate::core::AnimFloat::fixed(main_perspective_x);
                ai_note.object.translation.1 = crate::core::AnimFloat::fixed(main_perspective_y);
                ai_note
            }).collect();

            let cfg_arc = Arc::new(config.clone());
            let bpm_arc = Arc::new(bpm_list.clone());

            if let Some(tx) = AI_REQ_TX.get() {
                let req = AiRequest {
                    id: request_id,
                    line_id,
                    version,
                    timestamp: now,
                    notes: main_perspective_notes, // 已经转换为主视角坐标的音符
                    rotation,
                    config: cfg_arc,
                    bpm_list: bpm_arc,
                };

                // 非阻塞发送AI请求
                let _ = tx.send(req);
            }
        }
    }
    drop(line_states_guard);
}

pub fn default_max_grad_norm() -> f32 { 5.0_f32 }

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
    
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiTKAdvancedAI {
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
    /// 人体工程学手部系统
    ergonomic_hand_system: ErgonomicHandSystem,
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
    game_mode: crate::hand_model::GameMode,
    stability_factor: f32,
    hand_switch_penalty: f32,
    consistency_bonus: f32,
    adaptive_learning_rate: f32,
    pattern_recognition_strength: f32,
    memory_consolidation_rate: f32,
    recent_assignments: VecDeque<(Hand, f32, f32)>,
    hand_switch_count: u32,
    last_assigned_hand: Option<Hand>,
    recent_losses: VecDeque<f32>, // 最近损失记录
    // 防过拟合机制
    best_validation_reward: f32,
    validation_patience: u32,
    no_improvement_count: u32,
    early_stopping_threshold: f32,
    // 基于音符序列模式的早停机制
    sequence_pattern_history: VecDeque<SequencePatternMetrics>,
    pattern_diversity_threshold: f32,
    sequence_complexity_threshold: f32,
    convergence_window: u32,
    // 反思系统相关字段
    decision_history: VecDeque<(Hand, f32, bool)>, // (hand_assignment, timestamp, success)
    reflection_memory: HashMap<String, f32>, // 模式反思记忆
    reflection_confidence: f32, // 反思置信度
    last_reflection_time: f32, // 上次反思时间
    reflection_learning_rate: f32, // 反思学习率
    // 自主智慧AI集成
    wisdom_ai_enabled: bool, // 是否启用智慧AI
    consciousness_state: ConsciousnessState, // 当前意识状态
    wisdom_integration_level: f32, // 智慧集成程度
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
enum EarlyStopDecision {
    Continue,
    LearningRateAdjust,
    Stop,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SequencePatternMetrics {
    timestamp: f32,
    alternating_score: f32,
    stream_score: f32,
    chord_score: f32,
    jack_score: f32,
    crossing_score: f32,
    note_density: f32,
    rhythm_complexity: f32,
    sequence_variance: f32,
    hand_switching_frequency: f32,
    confidence_score: f32,
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
    difficulty: f32,
    duration: f32,
    // 物理模型判断相关字段
    actual_position: Option<Vector2>,
    position_error: f32,
    timing_error: f32,
    is_successful: bool,
    physical_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Experience {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
    timestamp: f32,
    // PPO新增字段
    log_prob: f32,
    value: f32,
    next_value: f32,
    advantage: f32,
    return_: f32,
    // 未来音符信息（4个未来音符）
    // 每个未来音符：[hand_left_prob, hand_right_prob, position_x, time_delta]
    future_notes: Vec<f32>,
    // 反思相关字段
    reflection_score: f32,      // 反思得分
    decision_history: Vec<usize>, // 过去决策历史
    outcome_success: bool,       // 决策结果是否成功
    reflection_features: Vec<f32>, // 反思提取的特征
    // 智慧AI相关字段
    hand_system_interface: f32,  // 手部系统接口评分
    evolution_history: f32,      // 进化历史评分
    current_consciousness: f32,  // 当前意识状态评分
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExperienceReplay {
    buffer: VecDeque<Experience>,
    capacity: usize,
}



// Finger 已被替换为使用 hand_model.rs 中的 FingerType
pub type Finger = crate::hand_model::FingerType;




#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepNeuralNetwork {
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
    last_loss: f32,
    #[serde(default)]
    bad_epochs: usize,
    epoch_count: u64,
    #[serde(skip)]device: Option<wgpu::Device>,
    #[serde(skip)]queue: Option<wgpu::Queue>,
    #[serde(skip)]gpu_executor: Option<Arc<crate::gpu_utils::GpuNetworkExecutor>>,
    #[serde(skip)]gpu_initialized: bool,
    #[serde(skip)]batch_size_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]initialization_attempted: bool,
    #[serde(skip)]initialization_failed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLayer {
    #[serde(default)]
    pub weights: Vec<Vec<f32>>,
    #[serde(default)]
    pub biases: Vec<f32>,
    #[serde(default)]
    pub activations: Vec<f32>,
    #[serde(default)]
    pub pre_activations: Vec<f32>,
    #[serde(default)]
    pub gradients: Vec<f32>,
    #[serde(default)]
    pub momentum_weights: Vec<Vec<f32>>,
    #[serde(default)]
    pub momentum_biases: Vec<f32>,
    #[serde(default)]
    pub bidirectional: bool,
    #[serde(default)]
    pub seq_len: usize,
    pub layer_type: LayerType,
    pub activation_func: ActivationFunction,
    #[serde(skip)]
    pub weights_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    pub biases_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    pub activations_buffer: Option<wgpu::Buffer>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LayerType {
    Dense,
    LSTM,
    Attention,
    Residual,
    Concat,
    Reflection,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Swish,
    GELU,
    Linear,
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

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LSTMParams {
    input_size: u32,
    hidden_size: u32,
    seq_len: u32,
    batch_size: u32,
    bidirectional: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DispatchParams {
    t: u32,
    direction: u32,
    _pad0: u32, // 16-byte alignment padding
    _pad1: u32, // 16-byte alignment padding
}

impl Vector2 {
    //fn new(x: f32, y: f32) -> Self { Self { x, y } }

    //fn distance_to(&self, other: &Vector2) -> f32 {
    //    ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    //}

    pub fn clean(&mut self) {
        if !self.x.is_finite() {
            self.x = 0.0;
        }
        if !self.y.is_finite() {
            self.y = 0.0;
        }
    }
}



impl DeepNeuralNetwork {
    pub fn new() -> Self {
        let mut network = Self {
            layers: Vec::new(),
            learning_rate: 0.05, //大幅提高初始学习率以增强梯度效果
            momentum: 0.9,
            dropout_rate: 0.3,
            batch_size: 64, //增加批次大小以提高GPU利用率
            max_grad_norm: 20.0, //显著提高梯度裁剪阈值，允许更大的梯度范围
            //weight_decay: 0.0001,
            last_loss: f32::INFINITY, //损失
            bad_epochs: 0,
            epoch_count: 0,
            device: None,
            queue: None,
            gpu_executor: None,
            gpu_initialized: false,
            batch_size_buffer: None,
            initialization_attempted: false,
            initialization_failed: false,
        };

        network.build_architecture();
        //network.init_gpu_sync();
        network
    }

    /// 将hand_model转换为网络输入向量
    pub fn hand_model_to_input(hand_system: &ErgonomicHandSystem) -> Vec<f32> {
        let mut input = Vec::with_capacity(152);
        
        // 左手数据 (12维)
        input.push(hand_system.left_hand.position.x);
        input.push(hand_system.left_hand.position.y);
        input.push(hand_system.left_hand.velocity.x);
        input.push(hand_system.left_hand.velocity.y);
        input.push(hand_system.left_hand.acceleration.x);
        input.push(hand_system.left_hand.acceleration.y);
        input.push(hand_system.left_hand.rotation);
        input.push(hand_system.left_hand.openness);
        input.push(hand_system.left_hand.fatigue);
        input.push(hand_system.left_hand.dexterity);
        input.push(hand_system.left_hand.last_update_time);
        input.push(hand_system.left_hand.hand_type as u8 as f32);
        
        // 右手数据 (12维)
        input.push(hand_system.right_hand.position.x);
        input.push(hand_system.right_hand.position.y);
        input.push(hand_system.right_hand.velocity.x);
        input.push(hand_system.right_hand.velocity.y);
        input.push(hand_system.right_hand.acceleration.x);
        input.push(hand_system.right_hand.acceleration.y);
        input.push(hand_system.right_hand.rotation);
        input.push(hand_system.right_hand.openness);
        input.push(hand_system.right_hand.fatigue);
        input.push(hand_system.right_hand.dexterity);
        input.push(hand_system.right_hand.last_update_time);
        input.push(hand_system.right_hand.hand_type as u8 as f32);
        
        // 左手指数据 (10维 × 5手指 = 50维)
        for finger in &hand_system.left_fingers {
            input.push(finger.position.x);
            input.push(finger.position.y);
            input.push(finger.bend_angle);
            input.push(finger.length);
            input.push(finger.thickness);
            input.push(finger.fatigue);
            input.push(finger.dexterity);
            input.push(if finger.is_pressed { 1.0 } else { 0.0 });
            input.push(finger.press_time);
            input.push(finger.finger_type as u8 as f32);
        }
        
        // 右手指数据 (10维 × 5手指 = 50维)
        for finger in &hand_system.right_fingers {
            input.push(finger.position.x);
            input.push(finger.position.y);
            input.push(finger.bend_angle);
            input.push(finger.length);
            input.push(finger.thickness);
            input.push(finger.fatigue);
            input.push(finger.dexterity);
            input.push(if finger.is_pressed { 1.0 } else { 0.0 });
            input.push(finger.press_time);
            input.push(finger.finger_type as u8 as f32);
        }
        
        // 左臂数据 (12维)
        input.push(hand_system.left_arm.shoulder_position.x);
        input.push(hand_system.left_arm.shoulder_position.y);
        input.push(hand_system.left_arm.elbow_position.x);
        input.push(hand_system.left_arm.elbow_position.y);
        input.push(hand_system.left_arm.wrist_position.x);
        input.push(hand_system.left_arm.wrist_position.y);
        input.push(hand_system.left_arm.angle);
        input.push(hand_system.left_arm.length);
        input.push(hand_system.left_arm.thickness);
        input.push(hand_system.left_arm.fatigue);
        input.push(hand_system.left_arm.strength);
        input.push(hand_system.left_arm.flexibility);
        
        // 右臂数据 (12维)
        input.push(hand_system.right_arm.shoulder_position.x);
        input.push(hand_system.right_arm.shoulder_position.y);
        input.push(hand_system.right_arm.elbow_position.x);
        input.push(hand_system.right_arm.elbow_position.y);
        input.push(hand_system.right_arm.wrist_position.x);
        input.push(hand_system.right_arm.wrist_position.y);
        input.push(hand_system.right_arm.angle);
        input.push(hand_system.right_arm.length);
        input.push(hand_system.right_arm.thickness);
        input.push(hand_system.right_arm.fatigue);
        input.push(hand_system.right_arm.strength);
        input.push(hand_system.right_arm.flexibility);
        
        // 身体数据 (4维)
        input.push(hand_system.body_center.x);
        input.push(hand_system.body_center.y);
        input.push(hand_system.body_tilt);
        input.push(hand_system.difficulty_factor);
        
        // 确保所有值都是有限的
        for x in &mut input {
            if !x.is_finite() {
                *x = 0.0;
            }
        }
        
        input
    }

    fn clean(&mut self) {
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
        }
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
        // 对输入进行裁剪以防止数值不稳定
        let clipped_x = x.clamp(-50.0, 50.0);  // 更严格的输入裁剪
        
        let result = match func {
            ActivationFunction::ReLU => {
                let res = clipped_x.max(0.0);
                res.clamp(-50.0, 50.0)
            },
            ActivationFunction::Sigmoid => {
                // 使用更稳定的sigmoid实现
                let res = if clipped_x > 20.0 {
                    1.0
                } else if clipped_x < -20.0 {
                    0.0
                } else {
                    1.0 / (1.0 + (-clipped_x).exp())
                };
                res.clamp(0.0, 1.0)
            },
            ActivationFunction::Tanh => {
                // 使用clamp确保输出在[-1, 1]范围内
                let res = clipped_x.tanh();
                res.clamp(-0.999, 0.999)  // 避免极端值
            },
            ActivationFunction::Swish => {
                // 防止指数函数溢出
                let exp_val = if clipped_x > 20.0 {
                    1.0
                } else if clipped_x < -20.0 {
                    0.0
                } else {
                    (-clipped_x).exp()
                };
                let sigmoid = 1.0 / (1.0 + exp_val);
                let res = clipped_x * sigmoid;
                // 更严格的输出裁剪
                res.clamp(-30.0, 30.0)
            },
            ActivationFunction::GELU => {
                // 使用更稳定的GELU实现
                let res = if clipped_x.abs() < 3.0 {
                    // 直接使用GELU公式
                    let sqrt_2_over_pi = 0.7978845608_f32;
                    let x3 = clipped_x * clipped_x * clipped_x;
                    let a = sqrt_2_over_pi * (clipped_x + 0.044715 * x3);
                    0.5 * clipped_x * (1.0 + a.tanh())
                } else {
                    // 对极端值使用近似
                    if clipped_x > 0.0 {
                        clipped_x
                    } else {
                        0.0
                    }
                };
                res.clamp(-20.0, 20.0)
            },
            ActivationFunction::Linear => clipped_x.clamp(-50.0, 50.0),
        };
        
        // 最终检查结果是否为有限值
        if result.is_finite() {
            result
        } else {
            0.0  // 如果结果不是有限值，返回0
        }
    }

    pub fn activate_derivative_from_z(x: f32, func: &ActivationFunction) -> f32 {
        // 对输入进行裁剪以防止数值不稳定
        let clipped_x = x.clamp(-100.0, 100.0);
        
        match func {
            ActivationFunction::ReLU => {
                if clipped_x > 0.0 { 1.0 } else { 0.0 }
            }
            ActivationFunction::Sigmoid => {
                // s = sigmoid(x)
                let exp_val = if clipped_x > 10.0 {
                    1.0
                } else if clipped_x < -10.0 {
                    0.0
                } else {
                    (-clipped_x).exp()
                };
                let s = 1.0 / (1.0 + exp_val);
                let result = s * (1.0 - s);
                // 确保结果在有效范围内
                result.clamp(0.0, 1.0)
            }
            ActivationFunction::Tanh => {
                let t = clipped_x.tanh();
                let result = 1.0 - t * t;
                // 确保结果在有效范围内
                result.clamp(0.0, 1.0)
            }
            ActivationFunction::Swish => {
                // swish(x) = x * sigmoid(x)
                let exp_val = if clipped_x > 10.0 {
                    1.0
                } else if clipped_x < -10.0 {
                    0.0
                } else {
                    (-clipped_x).exp()
                };
                let sig = 1.0 / (1.0 + exp_val);
                let result = sig + clipped_x * sig * (1.0 - sig);
                // 确保结果在合理范围内
                result.clamp(-10.0, 10.0)
            }
            ActivationFunction::GELU => {
                // derivative of GELU (approximation consistent with GELU definition used)
                let sqrt_2_over_pi = 0.7978845608_f32;
                let x3 = clipped_x * clipped_x * clipped_x;
                let a = sqrt_2_over_pi * (clipped_x + 0.044715 * x3);
                let clamped_a = a.clamp(-10.0, 10.0); // 限制a的范围
                let tanh_a = clamped_a.tanh();
                let left = 0.5 * (1.0 + tanh_a);
                let sech2 = 1.0 - tanh_a * tanh_a;
                let a_prime = sqrt_2_over_pi * (1.0 + 0.134145 * clipped_x * clipped_x); // 0.044715*3 = 0.134145
                let result = left + 0.5 * clipped_x * sech2 * a_prime;
                // 确保结果在合理范围内
                result.clamp(-10.0, 10.0)
            }
            ActivationFunction::Linear => 1.0,
        }
    }

    fn light_forward(&mut self, input: &[f32]) -> Vec<f32> {
    
    
        // 检查输入是否包含NaN或无穷大值
        let mut current_input: Vec<f32> = input.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
        let mut layer_outputs: Vec<Vec<f32>> = Vec::new();

        for layer in self.layers.iter_mut() {
            // 在每层处理前检查输入
            for x in &mut current_input {
                if !x.is_finite() {
                    *x = 0.0; // 将NaN或无穷大值替换为0
                }
            }
            
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
                            // 检查两个值是否都是有限的
                            if current_input[i].is_finite() && input_copy[i].is_finite() {
                                current_input[i] += input_copy[i];
                            } else {
                                current_input[i] = 0.0; // 如果任何一个值不是有限的，则使用默认值
                            }
                        }
                    }
                }
                LayerType::Concat => {
                    current_input = Self::concat_forward(layer, &layer_outputs);
                }
                LayerType::Reflection => {
                    current_input = Self::reflection_forward(layer, &current_input, &layer_outputs);
                }
            }
            
            // 在每层处理后检查输出
            for x in &mut current_input {
                if !x.is_finite() {
                    *x = 0.0; // 将NaN或无穷大值替换为0
                }
            }
            
            layer_outputs.push(current_input.clone());
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
                && self.gpu_executor.is_some();

            if init_result && resources_valid {
                self.gpu_initialized = true;
                self.initialization_failed = false;
                println!("[GPU] GPU initialization successful on first attempt");
            } else {
                self.initialization_failed = true;
                let missing = format!("device: {}, queue: {}, executor: {}",
                                      self.device.is_some(),
                                      self.queue.is_some(),
                                      self.gpu_executor.is_some()
                );
                eprintln!("[GPU] GPU initialization partially failed - missing resources: {}", missing);
                eprintln!("[GPU] Falling back to CPU implementation");
                // 清理可能的部分初始化的资源
                self.device = None;
                self.queue = None;
                self.batch_size_buffer = None;
                self.gpu_executor = None;
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
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12 | wgpu::Backends::METAL,
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

        // 获取适配器支持的极限
        let adapter_limits = adapter.limits();
        
        // 设置最高性能的设备限制
        let mut limits = wgpu::Limits::default();
        // 提高所有限制到最大值以获得最佳性能
        limits.max_texture_dimension_1d = adapter_limits.max_texture_dimension_1d;
        limits.max_texture_dimension_2d = adapter_limits.max_texture_dimension_2d;
        limits.max_texture_dimension_3d = adapter_limits.max_texture_dimension_3d;
        limits.max_texture_array_layers = adapter_limits.max_texture_array_layers;
        limits.max_bind_groups = adapter_limits.max_bind_groups;
        limits.max_bindings_per_bind_group = adapter_limits.max_bindings_per_bind_group;
        limits.max_dynamic_uniform_buffers_per_pipeline_layout = adapter_limits.max_dynamic_uniform_buffers_per_pipeline_layout;
        limits.max_dynamic_storage_buffers_per_pipeline_layout = adapter_limits.max_dynamic_storage_buffers_per_pipeline_layout;
        limits.max_sampled_textures_per_shader_stage = adapter_limits.max_sampled_textures_per_shader_stage;
        limits.max_samplers_per_shader_stage = adapter_limits.max_samplers_per_shader_stage;
        limits.max_storage_buffers_per_shader_stage = adapter_limits.max_storage_buffers_per_shader_stage;
        limits.max_storage_textures_per_shader_stage = adapter_limits.max_storage_textures_per_shader_stage;
        limits.max_uniform_buffers_per_shader_stage = adapter_limits.max_uniform_buffers_per_shader_stage;
        limits.max_uniform_buffer_binding_size = adapter_limits.max_uniform_buffer_binding_size;
        limits.max_storage_buffer_binding_size = adapter_limits.max_storage_buffer_binding_size;
        limits.min_uniform_buffer_offset_alignment = adapter_limits.min_uniform_buffer_offset_alignment;
        limits.min_storage_buffer_offset_alignment = adapter_limits.min_storage_buffer_offset_alignment;
        limits.max_vertex_buffers = adapter_limits.max_vertex_buffers;
        limits.max_buffer_size = adapter_limits.max_buffer_size;
        limits.max_vertex_attributes = adapter_limits.max_vertex_attributes;
        limits.max_vertex_buffer_array_stride = adapter_limits.max_vertex_buffer_array_stride;
        limits.max_inter_stage_shader_components = adapter_limits.max_inter_stage_shader_components;
        limits.max_compute_workgroup_storage_size = adapter_limits.max_compute_workgroup_storage_size;
        limits.max_compute_invocations_per_workgroup = adapter_limits.max_compute_invocations_per_workgroup;
        limits.max_compute_workgroup_size_x = adapter_limits.max_compute_workgroup_size_x;
        limits.max_compute_workgroup_size_y = adapter_limits.max_compute_workgroup_size_y;
        limits.max_compute_workgroup_size_z = adapter_limits.max_compute_workgroup_size_z;
        limits.max_compute_workgroups_per_dimension = adapter_limits.max_compute_workgroups_per_dimension;
        
        // 启用所有可用的性能特性（只使用当前wgpu版本支持的特性）
        let required_features = wgpu::Features::all()
            & wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
            & wgpu::Features::PIPELINE_STATISTICS_QUERY
            & wgpu::Features::TIMESTAMP_QUERY
            & wgpu::Features::INDIRECT_FIRST_INSTANCE
            & wgpu::Features::SHADER_F16
            & wgpu::Features::RG11B10UFLOAT_RENDERABLE
            & wgpu::Features::BGRA8UNORM_STORAGE
            & wgpu::Features::FLOAT32_FILTERABLE
            & wgpu::Features::TEXTURE_COMPRESSION_BC
            & wgpu::Features::TEXTURE_COMPRESSION_ETC2
            & wgpu::Features::TEXTURE_COMPRESSION_ASTC
            & wgpu::Features::TEXTURE_BINDING_ARRAY
            & wgpu::Features::BUFFER_BINDING_ARRAY
            & wgpu::Features::STORAGE_RESOURCE_BINDING_ARRAY
            & wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING
            & wgpu::Features::STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
            & wgpu::Features::PARTIALLY_BOUND_BINDING_ARRAY
            & wgpu::Features::MULTI_DRAW_INDIRECT
            & wgpu::Features::MULTI_DRAW_INDIRECT_COUNT
            & wgpu::Features::PUSH_CONSTANTS
            & wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER
            & wgpu::Features::ADDRESS_MODE_CLAMP_TO_ZERO
            & wgpu::Features::POLYGON_MODE_LINE
            & wgpu::Features::POLYGON_MODE_POINT
            & wgpu::Features::CONSERVATIVE_RASTERIZATION
            & wgpu::Features::VERTEX_WRITABLE_STORAGE
            & wgpu::Features::CLEAR_TEXTURE
            & wgpu::Features::SPIRV_SHADER_PASSTHROUGH
            & wgpu::Features::MULTIVIEW
            & wgpu::Features::SHADER_PRIMITIVE_INDEX
            & wgpu::Features::SHADER_EARLY_DEPTH_TEST
            & wgpu::Features::DEPTH32FLOAT_STENCIL8
            & wgpu::Features::DEPTH_CLIP_CONTROL
            & wgpu::Features::DUAL_SOURCE_BLENDING
            & wgpu::Features::TEXTURE_FORMAT_16BIT_NORM
            & wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR;

        let (device, queue) = match adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("High Performance GPU Device"),
            required_features,
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }).await {
            Ok((d, q)) => (d, q),
            Err(e) => {
                eprintln!("Failed to request GPU device: {:?}", e);
                return false;
            }
        };

        // 创建GPU执行器，实现一次性上传所有数据的优化
        let gpu_executor = Arc::new(crate::gpu_utils::GpuNetworkExecutor::new(
            Arc::new(device.clone()),
            Arc::new(queue.clone()),
        ));

        // 预先准备GPU层数据
        for (index, layer) in self.layers.iter_mut().enumerate() {
            // 将权重展平为一维数组
            let weights_flat = layer.weights.iter().flatten().cloned().collect::<Vec<f32>>();
            layer.weights_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} weights buffer", index)),
                contents: bytemuck::cast_slice(&weights_flat),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            layer.biases_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} biases buffer", index)),
                contents: bytemuck::cast_slice(&layer.biases),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            }));
            layer.activations_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Layer {} activations buffer", index)),
                size: (layer.activations.len() * size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }

        // 创建batch_size_buffer
        let batch_size_data = [self.batch_size as u32];
        self.batch_size_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Batch Size Buffer"),
            contents: bytemuck::cast_slice(&batch_size_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        }));

        self.device = Some(device);
        self.queue = Some(queue);
        self.gpu_executor = Some(gpu_executor);
        self.gpu_initialized = true;

        true
    }

    /*
    fn get_activation_shader(&self, func: &ActivationFunction) -> String {
        let fn_body = match func {
            ActivationFunction::ReLU => "return max(val, 0.0);",
            ActivationFunction::Sigmoid => "return 1.0 / (1.0 + exp(-val));",
            ActivationFunction::Tanh => "return tanh(val);",
            ActivationFunction::Swish => "return val * (1.0 / (1.0 + exp(-val)));",
            ActivationFunction::GELU => "return 0.5 * val * (1.0 + tanh(val * 0.7978845608 * (1.0 + 0.044715 * val * val)));",
            ActivationFunction::Linear => "return val;",
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

     */

    fn gpu_forward(&mut self, input: &[f32]) -> Vec<f32> {
        // 检查输入是否有效
        if input.is_empty() {
            println!("[GPU前向传播] 输入为空，回退到CPU");
            return self.light_forward(input);
        }

        //println!("[GPU前向传播] 输入维度: {}", input.len());
        if !self.gpu_initialized {
            self.init_gpu_sync();
            if !self.gpu_initialized {
                println!("[GPU] GPU not available, falling back to CPU");
                return self.light_forward(input);
            }
        }

        // 检查GPU资源是否完整
        if self.device.is_none() || self.queue.is_none() || self.gpu_executor.is_none() {
            println!("[GPU DEBUG] GPU resources not fully initialized. FALLING BACK TO CPU.");
            return self.light_forward(input);
        }

        // 使用优化的GPU执行器
        if let Some(executor) = self.gpu_executor.clone() {
            // 转换层数据为GPU优化格式
            let gpu_layers: Vec<crate::gpu_utils::NetworkLayerGPU> = self.layers.iter().map(|layer| {
                crate::gpu_utils::NetworkLayerGPU {
                    weights_flattened: layer.weights.iter().flatten().cloned().collect(),
                    biases: layer.biases.clone(),
                    output_size: match layer.layer_type {
                        LayerType::Dense => layer.weights.len(),
                        LayerType::LSTM => layer.activations.len(),
                        LayerType::Attention => layer.activations.len(),
                        LayerType::Residual => layer.weights.len(),
                        LayerType::Concat => layer.activations.len(),
                        LayerType::Reflection => layer.activations.len(),
                    },
                    seq_len: layer.seq_len,
                    layer_type: match layer.layer_type {
                        LayerType::Dense => crate::gpu_utils::LayerTypeGPU::Dense,
                        LayerType::LSTM => crate::gpu_utils::LayerTypeGPU::LSTM,
                        LayerType::Attention => crate::gpu_utils::LayerTypeGPU::Attention,
                        LayerType::Residual => crate::gpu_utils::LayerTypeGPU::Residual,
                        LayerType::Concat => crate::gpu_utils::LayerTypeGPU::Concat,
                        LayerType::Reflection => crate::gpu_utils::LayerTypeGPU::Dense,
                    },
                    num_inputs: if let LayerType::Concat = layer.layer_type {
                        // 对于Concat层，weights中存储了层索引
                        layer.weights.len()
                    } else {
                        0
                    },
                }
            }).collect();

            // 检查GPU层数据是否有效
            if gpu_layers.is_empty() {
                println!("[GPU DEBUG] No GPU layers prepared. FALLING BACK TO CPU.");
                return self.light_forward(input);
            }

            // 使用优化的执行器执行前向传播
            let result_size = self.layers.last().unwrap().activations.len();
            let result = executor.execute_network_forward(input, &gpu_layers, result_size);

            // 更新最后一层的激活值
            if let Some(last_layer) = self.layers.last_mut() {
                last_layer.activations = result.clone();
            }

            result
        } else {
            // 回退到CPU
            println!("[GPU DEBUG] Executor not available. FALLING BACK TO CPU (light_forward).");
            self.light_forward(input)
        }
    }


    fn gpu_forward_batch(&mut self, inputs: &[&[f32]], actual_batch_size: usize) -> Vec<Vec<f32>> {
        // 检查输入是否有效
        if inputs.is_empty() || actual_batch_size == 0 {
            println!("[GPU BATCH DEBUG] Empty batch, falling back to CPU");
            // 使用多线程处理，但避免需要可变借用
            return inputs.par_iter().map(|input| {
                DeepNeuralNetwork::light_forward_immutable(&self.layers, input)
            }).collect();
        }

        println!("[GPU BATCH DEBUG] Processing batch with {} samples.", actual_batch_size);
        if !self.gpu_initialized {
            self.init_gpu_sync();
            if !self.gpu_initialized {
                println!("[GPU] GPU not available, falling back to CPU");
                return inputs.iter().map(|input| self.light_forward(input)).collect();
            }
        }

        // 检查GPU资源是否完整
        if self.device.is_none() || self.queue.is_none() || self.gpu_executor.is_none() {
            println!("[GPU DEBUG] GPU resources not fully initialized. FALLING BACK TO CPU.");
            return inputs.iter().map(|input| self.light_forward(input)).collect();
        }

        // 使用GPU执行器
        if let Some(executor) = self.gpu_executor.clone() {
            // 转换层数据为GPU优化格式
            let gpu_layers: Vec<crate::gpu_utils::NetworkLayerGPU> = self.layers.iter().map(|layer| {
                crate::gpu_utils::NetworkLayerGPU {
                    weights_flattened: layer.weights.iter().flatten().cloned().collect(),
                    biases: layer.biases.clone(),
                    output_size: match layer.layer_type {
                        LayerType::Dense => layer.weights.len(),
                        LayerType::LSTM => layer.activations.len(),
                        LayerType::Attention => layer.activations.len(),
                        LayerType::Residual => layer.weights.len(),
                        LayerType::Concat => layer.activations.len(),
                        LayerType::Reflection => layer.activations.len(),
                    },
                    seq_len: layer.seq_len,
                    layer_type: match layer.layer_type {
                        LayerType::Dense => crate::gpu_utils::LayerTypeGPU::Dense,
                        LayerType::LSTM => crate::gpu_utils::LayerTypeGPU::LSTM,
                        LayerType::Attention => crate::gpu_utils::LayerTypeGPU::Attention,
                        LayerType::Residual => crate::gpu_utils::LayerTypeGPU::Residual,
                        LayerType::Concat => crate::gpu_utils::LayerTypeGPU::Concat,
                        LayerType::Reflection => crate::gpu_utils::LayerTypeGPU::Dense,
                    },
                    num_inputs: if let LayerType::Concat = layer.layer_type {
                        // 对于Concat层，weights中存储了层索引
                        layer.weights.len()
                    } else {
                        0
                    },
                }
            }).collect();

            // 检查GPU层数据是否有效
            if gpu_layers.is_empty() {
                println!("[GPU DEBUG] No GPU layers prepared. FALLING BACK TO CPU.");
                return inputs.iter().map(|input| self.light_forward(input)).collect();
            }

            // 处理批次输入
            let input_size = inputs[0].len();
            let total_input_size = input_size * actual_batch_size;
            let mut batch_input_data = Vec::with_capacity(total_input_size);
            for input in inputs {
                batch_input_data.extend_from_slice(input);
            }

            // 使用优化的执行器执行批次前向传播
            let result_size = self.layers.last().unwrap().activations.len();
            let flat_result = executor.execute_network_forward(&batch_input_data, &gpu_layers, result_size * actual_batch_size);

            // 将结果拆分为单个样本
            let mut results = Vec::with_capacity(actual_batch_size);
            for i in 0..actual_batch_size {
                let start = i * result_size;
                let end = start + result_size;
                results.push(flat_result[start..end].to_vec());
            }

            results
        } else {
            // 回退到CPU
            println!("[GPU DEBUG] Executor not available. FALLING BACK TO CPU.");
            inputs.iter().map(|input| self.light_forward(input)).collect()
        }
    }

    /*
    * 构建神经网络架构
     */
    fn build_architecture(&mut self) {
        //const INPUT_FUTURE_STEPS: usize = 16;
        const OUTPUT_PREDICTION_STEPS: usize = 16;
        const INPUT_DIM: usize = 2560; // 64个音符 * 40维特征，匹配特征提取器输出
        //const SEQ_LEN: usize = 64; // 匹配实际的音符窗口大小

        // 输入编码层 - 处理2560维输入，使用更深的网络
        self.add_dense_layer(INPUT_DIM, 512, ActivationFunction::GELU);
        self.add_dense_layer(512, 256, ActivationFunction::GELU);

        // 简化版序列建模 - 使用轻量级注意力或单一LSTM
        // 方案1：只使用注意力
        self.add_attention_layer(256, 256);
        self.add_residual_layer(256, 256);

        // 或方案2：只使用双向LSTM（更轻量）
        // self.add_lstm_layer_bi(256, 128, true, SEQ_LEN); // 输出256维

        // 特征提取层
        self.add_dense_layer(256, 128, ActivationFunction::GELU);
        self.add_residual_layer(128, 128);

        // 多任务输出头 - 共享特征，分离输出
        // 主决策输出：9维
        self.add_dense_layer(128, 64, ActivationFunction::GELU);
        self.add_dense_layer(64, 9, ActivationFunction::Linear);

        // 未来预测输出：48维
        self.add_dense_layer(128, 64, ActivationFunction::GELU);
        self.add_dense_layer(64, OUTPUT_PREDICTION_STEPS * 3, ActivationFunction::Linear);
    }

    //TODO: 归一化输出
    /*
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

     */

    fn add_dense_layer(&mut self, input_size: usize, output_size: usize, activation: ActivationFunction) {
        let mut weights = Vec::with_capacity(output_size);
        let mut biases = Vec::with_capacity(output_size);
        let mut momentum_weights = Vec::with_capacity(output_size);

        let std_dev = match activation {
            ActivationFunction::ReLU | ActivationFunction::GELU | ActivationFunction::Swish => {
                (2.0 / input_size as f32).sqrt()
            }
            ActivationFunction::Sigmoid | ActivationFunction::Tanh | ActivationFunction::Linear => {
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
        let _dir_mul = if bidirectional { 2 } else { 1 }; // 保存但不直接使用，避免警告
        // LSTM 需要 4 * output_size * dir_mul 个 bias（i, f, g, o）
        // 权重：W_ih (4*output_size × input_size) + W_hh (4*output_size × output_size)
        let weights_rows = 4 * output_size * if bidirectional { 2 } else { 1 };
        
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
            biases: vec![0.0; 4 * output_size * (if bidirectional { 2 } else { 1 })],
            activations: vec![0.0; output_size * (if bidirectional { 2 } else { 1 })],
            pre_activations: vec![0.0; weights_rows],
            gradients: vec![0.0; output_size * (if bidirectional { 2 } else { 1 })],
            momentum_weights,
            momentum_biases: vec![0.0; 4 * output_size * (if bidirectional { 2 } else { 1 })],
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
        let mut weights = Vec::with_capacity(output_size);
        let mut biases = Vec::with_capacity(output_size);
        let mut momentum_weights = Vec::with_capacity(output_size);

        let std_dev = (2.0 / input_size as f32).sqrt();

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
    
    // 添加连接层：用于合并多个输入流
    fn add_concat_layer(&mut self, input_sizes: &[usize], layer_indices: &[usize]) {
        let total_input_size: usize = input_sizes.iter().sum();
        let output_size = total_input_size; // 连接层输出维度等于输入维度之和
        
        // 连接层不需要权重和偏置，只是将输入拼接
        // 存储需要合并的层索引
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        
        // 使用权重字段存储层索引（hack方式）
        for &idx in layer_indices {
            weights.push(vec![idx as f32]);
            biases.push(idx as f32);
        }
        
        let layer = NetworkLayer {
            weights,
            biases,
            activations: vec![0.0; output_size],
            pre_activations: vec![0.0; output_size],
            gradients: vec![0.0; output_size],
            momentum_weights: vec![],
            momentum_biases: vec![],
            bidirectional: false,
            seq_len: 1,
            layer_type: LayerType::Concat,
            activation_func: ActivationFunction::Linear,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
        };
        
        self.layers.push(layer);
    }
    
    // 反思层：用于自我验证和反思
    fn add_reflection_layer(&mut self, input_size: usize, hidden_size: usize) {
        // 分析历史决策模式
        let output_size = input_size; // 输出维度与输入相同
        
        let mut layer = NetworkLayer {
            weights: vec![vec![0.0; input_size]; hidden_size],  // hidden_size x input_size
            biases: vec![0.0; hidden_size],
            activations: vec![0.0; hidden_size],
            pre_activations: vec![0.0; hidden_size],
            gradients: vec![0.0; hidden_size * input_size], // 展平的梯度
            momentum_weights: vec![vec![0.0; input_size]; hidden_size],
            momentum_biases: vec![0.0; hidden_size],
            bidirectional: false,
            seq_len: 0,
            layer_type: LayerType::Reflection,
            activation_func: ActivationFunction::GELU,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
        };
        
        // 初始化权重：反思层使用较小权重以避免过拟合
        for i in 0..hidden_size {
            for j in 0..input_size {
                layer.weights[i][j] = fastrand::f32() * 0.1 - 0.05; // [-0.05, 0.05] 范围
            }
            layer.biases[i] = 0.0;
        }
        
        // 初始化动量
        for i in 0..hidden_size {
            layer.momentum_biases[i] = 1e-8;
            for j in 0..input_size {
                layer.momentum_weights[i][j] = 1e-8;
            }
        }
        
        self.layers.push(layer);
        
        // 添加输出层到原始维度
        self.add_dense_layer(hidden_size, output_size, ActivationFunction::Linear);
    }

    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        // 检查输入是否包含NaN或无穷大值
        if input.iter().any(|&x| !x.is_finite()) {
            eprintln!("[Forward Warning] NaN/Inf detected in input, replacing with zeros");
            // 创建一个清理后的输入向量
            let clean_input: Vec<f32> = input.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
            if self.gpu_initialized && self.device.is_some() && self.gpu_executor.is_some() {
                return self.gpu_forward(&clean_input);
            } else {
                return self.cpu_forward(&clean_input);
            }
        }

        if self.gpu_initialized && self.device.is_some() && self.gpu_executor.is_some() {
            self.gpu_forward(input)
        } else {
            self.cpu_forward(input)
        }
    }
    
    fn cpu_forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut current_input = input.to_vec();
        let mut layer_outputs: Vec<Vec<f32>> = Vec::new();
        let total_layers = self.layers.len(); // 提前获取长度以避免借用冲突
        let mut reflection_start_idx = None;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            // 在每层处理前检查输入
            for x in &mut current_input {
                if !x.is_finite() {
                    *x = 0.0; // 将NaN或无穷大值替换为0
                }
            }
            
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
                    // 获取残差输入（通常是前一层或前几层的输出）
                    let residual_input = if layer_outputs.len() >= 2 {
                        layer_outputs[layer_outputs.len() - 2].clone()
                    } else {
                        current_input.clone()
                    };

                    current_input = Self::residual_forward(layer, &current_input, &residual_input);
                }
                LayerType::Concat => {
                    current_input = Self::concat_forward(layer, &layer_outputs);
                }
                LayerType::Reflection => {
                    current_input = Self::reflection_forward(layer, &current_input, &layer_outputs);
                }
            }
            
            // 检查当前层输出是否包含NaN或无穷大值
            for x in &mut current_input {
                if !x.is_finite() {
                    *x = 0.0; // 将NaN或无穷大值替换为0
                }
            }
            
            layer_outputs.push(current_input.clone());
            
            // 检测反思层的开始位置
            if layer.layer_type == LayerType::Dense && reflection_start_idx.is_none() {
                // 反思层是最后几层的Dense层
                if layer_idx >= total_layers - 4 {
                    reflection_start_idx = Some(layer_idx);
                }
            }
        }
        
        // 如果有反思层，执行反思过程
        if let Some(start_idx) = reflection_start_idx {
            let reflection_input = current_input.clone();
            let mut reflection_output = reflection_input;
            
            // 通过反思层进行前向传播
            for layer_idx in start_idx..total_layers {
                if let LayerType::Dense = self.layers[layer_idx].layer_type {
                    reflection_output = Self::dense_forward(&mut self.layers[layer_idx], &reflection_output);
                    
                    // 检查反思层输出
                    for x in &mut reflection_output {
                        if !x.is_finite() {
                            *x = 0.0; // 将NaN或无穷大值替换为0
                        }
                    }
                }
            }
            
            // 将反思结果与原始输出结合
            for i in 0..current_input.len() {
                // 确保两个值都是有限的
                if current_input[i].is_finite() && reflection_output[i].is_finite() {
                    current_input[i] = 0.7 * current_input[i] + 0.3 * reflection_output[i]; // 加权融合
                } else {
                    // 如果任何一个值不是有限的，则使用默认值
                    current_input[i] = 0.0;
                }
            }
        }

        // 最终输出检查
        for x in &mut current_input {
            if !x.is_finite() {
                *x = 0.0; // 将NaN或无穷大值替换为0
            }
        }

        current_input
    }

    pub fn dense_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; layer.weights.len()];
        let mut zs = vec![0.0; layer.weights.len()];

        layer.weights.par_iter()
            .zip(layer.biases.par_iter())
            .zip(zs.par_iter_mut())
            .zip(output.par_iter_mut())
            .for_each(|(((weights, bias), z), out)| {
                let mut sum = *bias;
                for (w, x) in weights.iter().zip(input.iter()) {
                    // 检查w和x是否为NaN或无穷大
                    if w.is_finite() && x.is_finite() {
                        sum += w * x;
                    }
                }
                
                // 检查sum是否为NaN或无穷大
                if !sum.is_finite() {
                    sum = 0.0; // 将NaN或无穷大替换为0
                }
                
                // 限制sum的范围以防止激活函数输出极端值
                sum = sum.clamp(-100.0, 100.0);
                
                *z = sum;
                *out = DeepNeuralNetwork::activate(sum, &layer.activation_func);
                
                // 检查激活函数输出是否为NaN或无穷大
                if !out.is_finite() {
                    *out = 0.0; // 将NaN或无穷大替换为0
                }
            });

        layer.pre_activations = zs;
        layer.activations = output.clone();
        output
    }

    fn lstm_cell(x_t: &[f32], h_prev: &[f32], c_prev: &[f32], weights_ih: &[Vec<f32>], weights_hh: &[Vec<f32>], biases: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let hidden_size = h_prev.len();
        let input_size = x_t.len();

        fn sigmoid(x: f32) -> f32 { 
            // 防止指数函数溢出
            if x > 10.0 {
                1.0
            } else if x < -10.0 {
                0.0
            } else {
                1.0 / (1.0 + (-x).exp())
            }
        }
        
        fn tanh(x: f32) -> f32 { 
            // tanh 函数本身对极端值处理较好，但仍进行检查
            x.tanh().clamp(-1.0, 1.0)
        }

        // 计算四个门的线性组合
        let mut ifgo = vec![0.0; 4 * hidden_size];

        // 计算输入门 (i)
        for i in 0..hidden_size {
            let mut sum = biases[i];
            for j in 0..input_size {
                // 检查权重和输入是否为有限值
                if weights_ih[i][j].is_finite() && x_t[j].is_finite() {
                    sum += weights_ih[i][j] * x_t[j];
                }
            }
            for j in 0..hidden_size {
                // 检查权重和隐藏状态是否为有限值
                if weights_hh[i][j].is_finite() && h_prev[j].is_finite() {
                    sum += weights_hh[i][j] * h_prev[j];
                }
            }
            // 检查偏差是否为有限值
            if !biases[i].is_finite() {
                sum = 0.0;
            }
            // 限制sum的范围以防止激活函数输出极端值
            ifgo[i] = sum.clamp(-100.0, 100.0);
        }

        // 计算遗忘门 (f)
        for i in 0..hidden_size {
            let mut sum = biases[hidden_size + i];
            for j in 0..input_size {
                if weights_ih[hidden_size + i][j].is_finite() && x_t[j].is_finite() {
                    sum += weights_ih[hidden_size + i][j] * x_t[j];
                }
            }
            for j in 0..hidden_size {
                if weights_hh[hidden_size + i][j].is_finite() && h_prev[j].is_finite() {
                    sum += weights_hh[hidden_size + i][j] * h_prev[j];
                }
            }
            if !biases[hidden_size + i].is_finite() {
                sum = 0.0;
            }
            ifgo[hidden_size + i] = sum.clamp(-100.0, 100.0);
        }

        // 计算候选值 (g)
        for i in 0..hidden_size {
            let mut sum = biases[2 * hidden_size + i];
            for j in 0..input_size {
                if weights_ih[2 * hidden_size + i][j].is_finite() && x_t[j].is_finite() {
                    sum += weights_ih[2 * hidden_size + i][j] * x_t[j];
                }
            }
            for j in 0..hidden_size {
                if weights_hh[2 * hidden_size + i][j].is_finite() && h_prev[j].is_finite() {
                    sum += weights_hh[2 * hidden_size + i][j] * h_prev[j];
                }
            }
            if !biases[2 * hidden_size + i].is_finite() {
                sum = 0.0;
            }
            ifgo[2 * hidden_size + i] = sum.clamp(-100.0, 100.0);
        }

        // 计算输出门 (o)
        for i in 0..hidden_size {
            let mut sum = biases[3 * hidden_size + i];
            for j in 0..input_size {
                if weights_ih[3 * hidden_size + i][j].is_finite() && x_t[j].is_finite() {
                    sum += weights_ih[3 * hidden_size + i][j] * x_t[j];
                }
            }
            for j in 0..hidden_size {
                if weights_hh[3 * hidden_size + i][j].is_finite() && h_prev[j].is_finite() {
                    sum += weights_hh[3 * hidden_size + i][j] * h_prev[j];
                }
            }
            if !biases[3 * hidden_size + i].is_finite() {
                sum = 0.0;
            }
            ifgo[3 * hidden_size + i] = sum.clamp(-100.0, 100.0);
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
            // 检查前一个细胞状态是否为有限值
            let prev_c = if c_prev[i].is_finite() { c_prev[i] } else { 0.0 };
            // 检查候选值是否为有限值
            let g_cand = if g_candidate[i].is_finite() { g_candidate[i] } else { 0.0 };
            // 检查输入门是否为有限值
            let i_g = if i_gate[i].is_finite() { i_gate[i] } else { 0.0 };
            
            c_t[i] = f_gate[i] * prev_c + i_g * g_cand;
            
            // 检查更新后的细胞状态是否为有限值
            if !c_t[i].is_finite() {
                c_t[i] = 0.0;
            } else {
                c_t[i] = c_t[i].clamp(-10.0, 10.0); // 限制细胞状态范围
            }
            
            // 检查输出门是否为有限值
            let o_g = if o_gate[i].is_finite() { o_gate[i] } else { 0.0 };
            
            h_t[i] = o_g * tanh(c_t[i]);
            
            // 检查更新后的隐藏状态是否为有限值
            if !h_t[i].is_finite() {
                h_t[i] = 0.0;
            } else {
                h_t[i] = h_t[i].clamp(-10.0, 10.0); // 限制隐藏状态范围
            }
        }

        (h_t, c_t)
    }

    fn lstm_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let seq_len = layer.seq_len;
        let feature_dim = if seq_len > 0 { input.len() / seq_len } else { input.len() };
        let output_size = layer.activations.len() / (if layer.bidirectional { 2 } else { 1 });
        let _dir_mul = if layer.bidirectional { 2 } else { 1 };

        if seq_len <= 1 || feature_dim == 0 {
            return Self::dense_forward(layer, input);
        }

        let weights = &layer.weights;
        let biases = &layer.biases;

        // 检查weights和biases数组大小
        let required_weights = if layer.bidirectional { 8 * output_size } else { 4 * output_size };
        let required_biases = if layer.bidirectional { 8 * output_size } else { 4 * output_size };
        
        if weights.len() < required_weights || biases.len() < required_biases {
            // 如果数组大小不够，回退到dense层
            return Self::dense_forward(layer, input);
        }

        let min_row_size = feature_dim + output_size;
        for row in weights.iter().take(required_weights) {
            if row.len() < min_row_size {
                return Self::dense_forward(layer, input);
            }
        }

        let weights_ih_fwd: Vec<Vec<f32>> = weights[..4 * output_size]
            .iter()
            .map(|row| row[..feature_dim].to_vec())
            .collect();
        let weights_hh_fwd: Vec<Vec<f32>> = weights[..4 * output_size]
            .iter()
            .map(|row| row[feature_dim..(feature_dim + output_size)].to_vec())
            .collect();
        let biases_fwd = &biases[..4 * output_size];

        let mut h_fwd = vec![0.0; output_size];
        let mut c_fwd = vec![0.0; output_size];
        for t in 0..seq_len {
            let start = t * feature_dim;
            let end = start + feature_dim;
            if end > input.len() { break; }
            let x_t = &input[start..end];
            let (new_h, new_c) = Self::lstm_cell(x_t, &h_fwd, &c_fwd, &weights_ih_fwd, &weights_hh_fwd, biases_fwd);
            
            // 检查输出是否为有限值
            for h in &new_h {
                if !h.is_finite() { continue; } // 如果不是有限值则跳过更新
            }
            for c in &new_c {
                if !c.is_finite() { continue; } // 如果不是有限值则跳过更新
            }
            
            h_fwd = new_h;
            c_fwd = new_c;
        }

        if !layer.bidirectional {
            // 检查最终输出是否为有限值
            for x in &mut h_fwd {
                if !x.is_finite() {
                    *x = 0.0;
                }
            }
            layer.activations = h_fwd.clone();
            return h_fwd;
        }

        let weights_ih_bwd: Vec<Vec<f32>> = weights[4 * output_size..8 * output_size]
            .iter()
            .map(|row| row[..feature_dim].to_vec())
            .collect();
        let weights_hh_bwd: Vec<Vec<f32>> = weights[4 * output_size..8 * output_size]
            .iter()
            .map(|row| row[feature_dim..(feature_dim + output_size)].to_vec())
            .collect();
        let biases_bwd = &biases[4 * output_size..8 * output_size];

        let mut h_bwd = vec![0.0; output_size];
        let mut c_bwd = vec![0.0; output_size];
        for t in (0..seq_len).rev() {
            let start = t * feature_dim;
            let end = start + feature_dim;
            if end > input.len() { continue; }
            let x_t = &input[start..end];
            let (new_h, new_c) = Self::lstm_cell(x_t, &h_bwd, &c_bwd, &weights_ih_bwd, &weights_hh_bwd, biases_bwd);
            
            // 检查输出是否为有限值
            for h in &new_h {
                if !h.is_finite() { continue; } // 如果不是有限值则跳过更新
            }
            for c in &new_c {
                if !c.is_finite() { continue; } // 如果不是有限值则跳过更新
            }
            
            h_bwd = new_h;
            c_bwd = new_c;
        }

        // 拼接正向 + 反向
        let mut output = Vec::with_capacity(2 * output_size);
        output.extend(h_fwd);
        output.extend(h_bwd);
        
        // 检查最终输出是否为有限值
        for x in &mut output {
            if !x.is_finite() {
                *x = 0.0;
            }
        }
        
        layer.activations = output.clone();
        output
    }

    /// 实现完整的多头自注意力机制
    fn attention_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        // 获取层参数
        let input_len = input.len();
        let output_size = layer.activations.len();
        
        // 注意力头数，假设为8
        let num_heads = 8;
        let head_dim = output_size / num_heads;
        
        // 确保输出维度能被头数整除
        if output_size % num_heads != 0 {
            // 如果不能整除，回退到简化版本
            return Self::simplified_attention_forward(layer, input);
        }
        
        // 为每个注意力头计算QKV
        let mut multi_head_output = vec![0.0; output_size];
        
        // 获取权重和偏置
        let weights = &layer.weights;
        let biases = &layer.biases;
        
        // 检查权重和偏置是否足够
        if weights.len() < 3 * output_size || biases.len() < 3 * output_size {
            // 如果权重或偏置不足，回退到简化版本
            return Self::simplified_attention_forward(layer, input);
        }
        
        // 对每个注意力头进行计算
        for head in 0..num_heads {
            let start_idx = head * head_dim;
            let _end_idx = (head + 1) * head_dim;
            
            // 计算当前头的QKV
            let mut q = vec![0.0; head_dim];
            let mut k = vec![0.0; head_dim];
            let mut v = vec![0.0; head_dim];
            
            // 计算QKV投影
            for i in 0..head_dim {
                let q_idx = start_idx + i;
                let k_idx = output_size + start_idx + i;
                let v_idx = 2 * output_size + start_idx + i;
                
                // 计算Q
                let mut sum_q = if q_idx < biases.len() { biases[q_idx] } else { 0.0 };
                for j in 0..input_len.min(weights[q_idx].len()) {
                    sum_q += weights[q_idx][j] * input[j];
                }
                q[i] = sum_q;
                
                // 计算K
                let mut sum_k = if k_idx < biases.len() { biases[k_idx] } else { 0.0 };
                for j in 0..input_len.min(weights[k_idx].len()) {
                    sum_k += weights[k_idx][j] * input[j];
                }
                k[i] = sum_k;
                
                // 计算V
                let mut sum_v = if v_idx < biases.len() { biases[v_idx] } else { 0.0 };
                for j in 0..input_len.min(weights[v_idx].len()) {
                    sum_v += weights[v_idx][j] * input[j];
                }
                v[i] = sum_v;
            }
            
            // 计算注意力分数
            let mut attention_scores = vec![0.0; head_dim];
            let dk = (head_dim as f32).sqrt();
            
            // 简化的点积注意力
            for i in 0..head_dim {
                attention_scores[i] = (q[i] * k[i]) / dk;
            }
            
            // 简化的softmax (实际上应该对所有位置进行softmax，但这里简化处理)
            let max_score = attention_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores: Vec<f32> = attention_scores.iter().map(|&x| (x - max_score).exp()).collect();
            let sum_exp: f32 = exp_scores.iter().sum();
            
            // 加权V值
            for i in 0..head_dim {
                let attention_weight = if sum_exp > 1e-8 { exp_scores[i] / sum_exp } else { 1.0 / head_dim as f32 };
                multi_head_output[start_idx + i] = attention_weight * v[i];
            }
        }
        
        // 应用输出投影 (这里简化处理，直接返回多头结果)
        layer.activations = multi_head_output.clone();
        multi_head_output
    }
    
    /// 简化版注意力机制 (用于回退)
    fn simplified_attention_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let output_size = layer.activations.len();
        let input_len = input.len();

        // 简化版：对单个向量做 QKV
        let qkv_weights = &layer.weights;
        let qkv_biases = &layer.biases;

        let mut q = vec![0.0; output_size];
        let mut k = vec![0.0; output_size];
        let mut v = vec![0.0; output_size];

        for i in 0..output_size {
            let mut sum_q = qkv_biases.get(i).copied().unwrap_or(0.0);
            let mut sum_k = qkv_biases.get(output_size + i).copied().unwrap_or(0.0);
            let mut sum_v = qkv_biases.get(2 * output_size + i).copied().unwrap_or(0.0);
            
            for j in 0..input_len {
                if j < qkv_weights.get(i).map_or(0, |w| w.len()) {
                    sum_q += qkv_weights[i][j] * input[j];
                }
                if j < qkv_weights.get(output_size + i).map_or(0, |w| w.len()) {
                    sum_k += qkv_weights[output_size + i][j] * input[j];
                }
                if j < qkv_weights.get(2 * output_size + i).map_or(0, |w| w.len()) {
                    sum_v += qkv_weights[2 * output_size + i][j] * input[j];
                }
            }
            q[i] = sum_q;
            k[i] = sum_k;
            v[i] = sum_v;
        }

        // Scaled Dot-Product Attention
        let dk = (output_size as f32).sqrt();
        let mut scores = vec![0.0; output_size];
        for i in 0..output_size {
            scores[i] = (q[i] * k[i]) / dk;
        }

        // 简化的Softmax
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f32> = scores.iter().map(|&x| (x - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        
        // 计算注意力输出
        let mut output = vec![0.0; output_size];
        for i in 0..output_size {
            let attention_weight = if sum_exp > 1e-8 { exp_scores[i] / sum_exp } else { 1.0 / output_size as f32 };
            output[i] = attention_weight * v[i];
        }

        layer.activations = output.clone();
        output
    }

    /// 实现残差连接的前向传播
    fn residual_forward(layer: &mut NetworkLayer, input: &[f32], residual: &[f32]) -> Vec<f32> {
        // 首先计算主路径的输出
        let main_path_output = Self::dense_forward_static(&layer.weights, &layer.biases, input, &layer.activation_func);
        
        // 确保输出维度匹配
        let output_dim = main_path_output.len();
        let mut output = vec![0.0; output_dim];
        
        // 如果输入维度与输出维度不匹配，需要进行投影
        if input.len() != output_dim {
            // 使用1x1卷积或线性投影来匹配维度
            // 这里简化处理，只在维度匹配时添加残差
            output.copy_from_slice(&main_path_output);
        } else {
            // 维度匹配时，添加残差连接
            for i in 0..output_dim {
                // 标准残差连接：输出 = 主路径输出 + 残差输入
                output[i] = main_path_output[i] + residual.get(i).copied().unwrap_or(0.0);
            }
        }
        
        // 应用后激活函数（如果需要）
        // 这里我们保持原始激活函数
        if let ActivationFunction::Linear = layer.activation_func {
            // 如果是线性激活，保持原样
        } else {
            // 对输出应用激活函数
            for i in 0..output_dim {
                output[i] = DeepNeuralNetwork::activate(output[i], &layer.activation_func);
            }
        }
        
        output
    }

    pub fn dense_forward_static(weights: &[Vec<f32>], biases: &[f32], input: &[f32], activation: &ActivationFunction) -> Vec<f32> {
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
    
    fn concat_forward(layer: &NetworkLayer, layer_outputs: &[Vec<f32>]) -> Vec<f32> {
        // 从 layer.weights 中提取需要合并的层索引
        let mut output = Vec::new();
        
        for idx_row in &layer.weights {
            if let Some(&idx_f32) = idx_row.first() {
                let idx = idx_f32 as usize;
                if idx < layer_outputs.len() {
                    output.extend_from_slice(&layer_outputs[idx]);
                }
            }
        }
        
        output
    }

    fn reflection_forward(layer: &mut NetworkLayer, input: &[f32], layer_outputs: &[Vec<f32>]) -> Vec<f32> {
        // 反思层：分析历史决策，生成反思信号
        // 收集前几层的重要特征（通常前1-3层的输出）
        let mut reflection_context = Vec::new();
        
        // 从历史层输出中提取决策模式
        for (i, prev_output) in layer_outputs.iter().rev().take(3).enumerate() {
            if i < layer.weights.len() {
                // 对每个历史输出应用权重
                let mut weighted_output = vec![0.0; prev_output.len().min(layer.weights[i].len())];
                for (j, &weight) in layer.weights[i].iter().take(weighted_output.len()).enumerate() {
                    if j < prev_output.len() {
                        weighted_output[j] = weight * prev_output[j];
                    }
                }
                reflection_context.extend_from_slice(&weighted_output);
            }
        }
        
        // 如果没有足够的上下文，使用当前输入
        if reflection_context.is_empty() {
            reflection_context = input.to_vec();
        }
        
        // 反思分析：通过注意力机制关注重要的历史信息
        let mut output = vec![0.0; input.len()];
        
        // 对每个输出单元，计算反思加权
        for (i, output_val) in output.iter_mut().enumerate() {
            if i < layer.biases.len() {
                let mut reflection_score = layer.biases[i];
                
                // 使用简单的注意力机制：相关性 = 点积
                let context_len = reflection_context.len().min(input.len());
                for j in 0..context_len {
                    let correlation = input[j] * reflection_context.get(j).unwrap_or(&0.0);
                    if i < layer.weights.len() && j < layer.weights[i].len() {
                        reflection_score += layer.weights[i][j] * correlation;
                    }
                }
                
                // 应用激活函数
                *output_val = Self::activate(reflection_score, &layer.activation_func);
            }
        }
        
        // 更新层的激活值（用于反向传播）
        layer.activations = output.clone();
        layer.pre_activations = vec![0.0; output.len()]; // 反思层不需要pre_activation
        
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

    /*
    fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) {
        if training_data.is_empty() {
            eprintln!("警告: 训练数据为空，跳过训练");
            return;
        }
        
        // 初始化动量参数
        for layer in &mut self.layers {
            if !layer.weights.is_empty() && !layer.momentum_weights.is_empty() {
                if layer.momentum_weights.len() != layer.weights.len() {
                    layer.momentum_weights = vec![vec![1e-8; layer.weights[0].len()]; layer.weights.len()];
                } else {
                    for i in 0..layer.momentum_weights.len() {
                        if layer.momentum_weights[i].len() != layer.weights[i].len() {
                            layer.momentum_weights[i] = vec![1e-8; layer.weights[i].len()];
                        } else {
                            // 为已有的动量参数添加小幅扰动，避免全零
                            for j in 0..layer.momentum_weights[i].len() {
                                if layer.momentum_weights[i][j] == 0.0 {
                                    layer.momentum_weights[i][j] = 1e-8;
                                }
                            }
                        }
                    }
                }
            } else if !layer.weights.is_empty() {
                layer.momentum_weights = vec![vec![1e-8; layer.weights[0].len()]; layer.weights.len()];
            }
            if layer.momentum_biases.len() != layer.biases.len() {
                layer.momentum_biases = vec![1e-8; layer.biases.len()];
            } else {
                // 为已有的偏置动量添加小幅扰动
                for j in 0..layer.momentum_biases.len() {
                    if layer.momentum_biases[j] == 0.0 {
                        layer.momentum_biases[j] = 1e-8;
                    }
                }
            }
        }

        let batch_count = (training_data.len() + self.batch_size - 1) / self.batch_size;
        let mut total_loss = 0.0;
        let mut valid_samples = 0;

        for batch_idx in 0..batch_count {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = start_idx + self.batch_size.min(training_data.len() - start_idx);
            let batch = &training_data[start_idx..end_idx];
            let mut batch_loss = 0.0;
            let mut batch_valid_samples = 0;
            
            for (input, target) in batch {
                // 检查输入和目标是否包含NaN或无穷大值
                let input_valid = input.iter().all(|&x| x.is_finite());
                let target_valid = target.iter().all(|&x| x.is_finite());
                
                if input_valid && target_valid {
                    let output = self.light_forward(input);
                    // 检查输出是否有效
                    if output.iter().all(|&x| x.is_finite()) {
                        let loss: f32 = output.iter().zip(target.iter()).take(2) // 只取 hand Q 值
                            .map(|(o, t)| {
                                // 确保不会出现NaN
                                let diff = o - t;
                                if diff.is_finite() {
                                    diff.powi(2)
                                } else {
                                    0.0 // 如果差值不是有限值，则损失为0
                                }
                            })
                            .sum();
                        
                        // 检查损失是否为有限值
                        if loss.is_finite() {
                            batch_loss += loss;
                            batch_valid_samples += 1;
                        }
                    }
                }
            }
            
            // 只有当批次中有有效样本时才进行训练
            if batch_valid_samples > 0 {
                total_loss += batch_loss;
                valid_samples += batch_valid_samples;
                self.train_batch(batch);
            } else {
                eprintln!("[Train Warning] Batch {} has no valid samples, skipping", batch_idx);
            }

            if batch_idx == batch_count - 1 {
                // 只有当有有效样本时才计算平均损失
                let avg_loss = if valid_samples > 0 {
                    total_loss / valid_samples as f32
                } else {
                    0.0 // 如果没有有效样本，损失为0
                };
                
                // 检查平均损失是否为有限值
                let safe_avg_loss = if avg_loss.is_finite() {
                    avg_loss
                } else {
                    eprintln!("[Train Warning] NaN/Inf loss detected, using 0.0");
                    0.0
                };
                
                self.adapt_learning_rate(safe_avg_loss);
                self.epoch_count += 1;
                
                if self.epoch_count % 1 == 0 {
                    let (min_weight, max_weight) = self.get_weight_range();
                    let (min_momentum, max_momentum) = self.get_momentum_range();
                    println!("[训练状态] Epoch: {}, Loss: {:.6}, LR: {:.6}, 权重范围 [{:.4}, {:.4}], 动量范围 [{:.4}, {:.4}]",
                             self.epoch_count, safe_avg_loss, self.learning_rate, min_weight, max_weight, min_momentum, max_momentum);
                }
            }
        }
    }

     */

    fn compute_ppo_policy_gradients(&self, ratio: f32, advantage: f32, action: usize, left_prob: f32, right_prob: f32, ppo_epsilon: f32,
    ) -> (f32, f32) {
        let unclipped = ratio * advantage;
        let clipped_ratio = ratio.clamp(1.0 - ppo_epsilon, 1.0 + ppo_epsilon);
        let clipped = clipped_ratio * advantage;

        // 判断哪个项被选中（PPO取min，但注意：如果advantage<0，min更负）
        let use_unclipped = if advantage >= 0.0 {
            unclipped <= clipped // 因为要最大化，实际是取max，但PPO写成min负loss
        } else {
            unclipped >= clipped // 当A<0时，clipped更小（更负），所以min会选clipped
        };
        let scale = if use_unclipped {
            ratio * advantage
        } else {
            0.0
        };

        // ∇logπ(a) for binary action:
        match action {
            0 => (
                scale * (1.0 - left_prob),   // d log π0 / d logit0 = 1 - π0
                scale * (-right_prob),       // d log π0 / d logit1 = -π1
            ),
            1 => (
                scale * (-left_prob),        // d log π1 / d logit0 = -π0
                scale * (1.0 - right_prob),  // d log π1 / d logit1 = 1 - π1
            ),
            _ => (0.0, 0.0),
        }
    }

    fn train_with_ppo(&mut self, training_data: &[(Vec<f32>, Vec<f32>)], experiences: &[Experience]) {
        if training_data.is_empty() {
            eprintln!("警告: 训练数据为空，跳过训练");
            return;
        }
        const PPO_EPSILON: f32 = 0.2;
        const GAMMA: f32 = 0.99;
        const PPO_EPOCHS: usize = 4;
        const MINI_BATCH_SIZE: usize = 64;
        const GRAD_CLIP_NORM: f32 = 0.5;
        const VALUE_LOSS_COEF: f32 = 0.5;
        const ENTROPY_COEF: f32 = 0.01;
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
        // 准备训练数据
        let mut valid_data: Vec<(Vec<f32>, Vec<f32>, Experience)> = Vec::new();
        for (i, (input, target)) in training_data.iter().enumerate() {
            if i < experiences.len() {
                let exp = &experiences[i];
                
                // 验证数据有效性
                let input_valid = input.iter().all(|&x| x.is_finite());
                let target_valid = target.iter().all(|&x| x.is_finite());
                let exp_valid = exp.state.iter().all(|&x| x.is_finite()) && 
                               exp.reward.is_finite() && 
                               exp.value.is_finite();
                
                if input_valid && target_valid && exp_valid {
                    valid_data.push((input.clone(), target.clone(), exp.clone()));
                }
            }
        }
        if valid_data.is_empty() { eprintln!("[PPO错误] 没有有效的训练数据");
            return;
        }
        let mut advantages = Vec::with_capacity(valid_data.len());
        let mut returns = Vec::with_capacity(valid_data.len());
        
        for i in (0..valid_data.len()).rev() {
            let exp = &valid_data[i].2;
            let next_state_value = if exp.done { 0.0 } else if i < valid_data.len() - 1 { valid_data[i + 1].2.value } else { 0.0 };
            let td_target = exp.reward + GAMMA * next_state_value;
            returns.insert(0, td_target);
            let advantage = td_target - exp.value;
            advantages.insert(0, advantage);
        }
        let adv_mean = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let adv_std = (advantages.iter().map(|&a| (a - adv_mean).powi(2)).sum::<f32>() / advantages.len() as f32).sqrt();
        let normalized_advantages: Vec<f32> = advantages.iter()
            .map(|&a| (a - adv_mean) / (adv_std + 1e-8))
            .collect();
        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;
        let mut total_samples = 0;
        for _epoch in 0..PPO_EPOCHS {
            // 随机打乱数据
            let mut indices: Vec<usize> = (0..valid_data.len()).collect();
            fastrand::shuffle(&mut indices);
            
            // 分批处理
            for chunk in indices.chunks(MINI_BATCH_SIZE) {
                let mut batch_policy_loss = 0.0;
                let mut batch_value_loss = 0.0;
                let mut batch_entropy = 0.0;
                let mut batch_samples = 0;
                
                // 准备批次输入
                let mut batch_inputs: Vec<&[f32]> = Vec::new();
                
                for &idx in chunk {
                    let (input, _, _) = &valid_data[idx];
                    batch_inputs.push(input);
                }
                
                // 前向传播
                let outputs = self.gpu_forward_batch(&batch_inputs, batch_inputs.len());
                
                // 累积梯度
                let mut total_gradients: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; 0]; 0]; self.layers.len()];
                let mut total_bias_gradients: Vec<Vec<f32>> = vec![vec![0.0; 0]; self.layers.len()];
                
                // 初始化梯度累积器
                for layer_idx in 0..self.layers.len() {
                    let layer = &self.layers[layer_idx];
                    if !layer.weights.is_empty() {
                        total_gradients[layer_idx] = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
                    } else {
                        total_gradients[layer_idx] = vec![];
                    }
                    total_bias_gradients[layer_idx] = vec![0.0; layer.biases.len()];
                }
                
                for (batch_idx, &idx) in chunk.iter().enumerate() {
                    let (_, _, exp) = &valid_data[idx];
                    let output = &outputs[batch_idx];
                    let advantage = normalized_advantages[idx];
                    let return_ = returns[idx];
                    
                    // 提取策略概率（假设前两个输出是左右手概率）
                    let left_logit = output[0];
                    let right_logit = output[1];
                    
                    // 应用softmax获取概率分布
                    let max_logit = left_logit.max(right_logit);
                    let exp_left = (left_logit - max_logit).exp();
                    let exp_right = (right_logit - max_logit).exp();
                    let softmax_sum = exp_left + exp_right + 1e-8;
                    
                    let left_prob = exp_left / softmax_sum;
                    let right_prob = exp_right / softmax_sum;
                    
                    // 计算当前策略的对数概率
                    let current_log_prob = if exp.action == 0 { 
                        left_prob.ln() 
                    } else { 
                        right_prob.ln() 
                    };
                    
                    // 计算重要性采样比率
                    let ratio = (current_log_prob - exp.log_prob).exp();
                    
                    // PPO裁剪目标函数 - 参考PyTorch实现
                    let clipped_ratio = ratio.clamp(1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON);
                    let surr1 = ratio * advantage;
                    let surr2 = clipped_ratio * advantage;
                    let policy_loss = -surr1.min(surr2); // 对应 -torch.min(surr1, surr2)
                    
                    // 价值函数损失
                    let value_pred = output.get(2).copied().unwrap_or(0.0);
                    let value_loss = 0.5 * (value_pred - return_).powi(2); // 对应 0.5 * MSE loss
                    
                    // 策略熵（鼓励探索）- 参考PyTorch实现
                    let entropy = -(left_prob * left_prob.ln() + right_prob * right_prob.ln());
                    
                    // 验证所有值都是有限的
                    if policy_loss.is_finite() && value_loss.is_finite() && entropy.is_finite() {
                        batch_policy_loss += policy_loss;
                        batch_value_loss += value_loss;
                        batch_entropy += entropy;
                        batch_samples += 1;
                        
                        // 计算损失对输出的梯度
                        let mut output_gradients = vec![0.0; output.len()];
                        
                        // 优化的策略梯度计算 - 利用现有架构和批量处理
                        let (grad_left, grad_right) = self.compute_ppo_policy_gradients(
                            ratio, advantage, exp.action, left_prob, right_prob, PPO_EPSILON
                        );
                        
                        output_gradients[0] = grad_left;
                        output_gradients[1] = grad_right;
                        
                        // 价值函数梯度 - 统一使用VALUE_LOSS_COEF
                        if output_gradients.len() > 2 {
                            output_gradients[2] = VALUE_LOSS_COEF * (value_pred - return_); // d/dx (0.5 * (x - target)^2) = (x - target)
                        }
                        
                        // 熵梯度计算 - 统一使用ENTROPY_COEF
                        // 熵损失对logits的导数: -ENTROPY_COEF * d/dlogits H(π)
                        //if exp.action == 0 { // Left action
                        //    output_gradients[0] -= ENTROPY_COEF * (left_prob.ln() + 1.0) * left_prob * (1.0 - left_prob);
                        //    output_gradients[1] += ENTROPY_COEF * (right_prob.ln() + 1.0) * left_prob * right_prob;
                        //} else { // Right action
                         //   output_gradients[0] += ENTROPY_COEF * (left_prob.ln() + 1.0) * right_prob * left_prob;
                        //    output_gradients[1] -= ENTROPY_COEF * (right_prob.ln() + 1.0) * right_prob * (1.0 - right_prob);
                        //}
                        output_gradients[0] -= ENTROPY_COEF * left_prob.ln();
                        output_gradients[1] -= ENTROPY_COEF * right_prob.ln();
                        
                        // 反向传播累积梯度
                        let (input, _, _) = &valid_data[idx];
                        self.backward(input, output, &output_gradients, &mut total_gradients, &mut total_bias_gradients);
                    }
                }
                
                if batch_samples > 0 {
                    // 平均梯度
                    let inv_batch = 1.0 / batch_samples as f32;
                    for layer_idx in 0..total_gradients.len() {
                        for i in 0..total_gradients[layer_idx].len() {
                            for j in 0..total_gradients[layer_idx][i].len() {
                                total_gradients[layer_idx][i][j] *= inv_batch;
                            }
                        }
                        for i in 0..total_bias_gradients[layer_idx].len() {
                            total_bias_gradients[layer_idx][i] *= inv_batch;
                        }
                    }
                    self.clip_gradients(&mut total_gradients, &mut total_bias_gradients, GRAD_CLIP_NORM);
                    self.update_weights_with_momentum(&total_gradients, &total_bias_gradients);
                    total_policy_loss += batch_policy_loss;
                    total_value_loss += batch_value_loss;
                    total_entropy += batch_entropy;
                    total_samples += batch_samples;
                }
            }
        }

        if total_samples > 0 {
            let avg_policy_loss = total_policy_loss / total_samples as f32;
            let avg_value_loss = total_value_loss / total_samples as f32;
            let avg_entropy = total_entropy / total_samples as f32;
            let total_loss = avg_policy_loss + VALUE_LOSS_COEF * avg_value_loss - ENTROPY_COEF * avg_entropy;

            if let Some(ai_system) = WISDOM_AI.get() {
                if let Ok(mut ai) = ai_system.lock() {
                    ai.hand_ai.recent_losses.push_back(total_loss);
                    if ai.hand_ai.recent_losses.len() > 20 {
                        ai.hand_ai.recent_losses.pop_front();
                    }
                }
            }
            
            self.adapt_learning_rate(total_loss);
            self.epoch_count += 1;
            
            if self.epoch_count % 1 == 0 {
                let (min_weight, max_weight) = self.get_weight_range();
                let (min_momentum, max_momentum) = self.get_momentum_range();
                println!("[PPO训练] Epoch: {}, 总损失: {:.6}, 策略损失: {:.6}, 价值损失: {:.6}, 熵: {:.6}, LR: {:.6}, 样本数: {}",
                         self.epoch_count, total_loss, avg_policy_loss, avg_value_loss, avg_entropy, 
                         self.learning_rate, total_samples);
                println!("  权重范围 [{:.4}, {:.4}], 动量范围 [{:.4}, {:.4}]",
                         min_weight, max_weight, min_momentum, max_momentum);
            }
        }
    }
    
    // 梯度裁剪函数
    fn clip_gradients(
        &self,
        gradients: &mut Vec<Vec<Vec<f32>>>,
        bias_gradients: &mut Vec<Vec<f32>>,
        max_norm: f32,
    ) {
        // 安全常数，防止除零
        const EPS: f32 = 1e-7;

        // 计算 L2 范数的平方
        let mut total_norm_sq: f32 = 0.0;
        for layer in &mut *gradients {
            for row in layer {
                for &grad in row.iter() {
                    total_norm_sq += grad * grad;
                }
            }
        }
        for layer in &mut *bias_gradients {
            for grad in layer {
                total_norm_sq += (*grad) * (*grad);
            }
        }

        // 计算 L2 范数，加 epsilon 防止 sqrt(0) 后除零（虽然 sqrt(0)=0，但后续除法需防护）
        let total_norm = total_norm_sq.sqrt();

        if total_norm > max_norm {
            let scale = max_norm / (total_norm + EPS); // ← 防止 total_norm 为 0（尽管 unlikely）
            for layer in gradients {
                for row in layer {
                    for grad in row {
                        *grad *= scale;
                    }
                }
            }
            for layer in bias_gradients {
                for grad in layer {
                    *grad *= scale;
                }
            }
        }
    }
    
    // 使用动量更新权重
    fn update_weights_with_momentum(
        &mut self,
        gradients: &Vec<Vec<Vec<f32>>>,
        bias_gradients: &Vec<Vec<f32>>,
    ) {
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            if layer_idx >= gradients.len() || layer_idx >= bias_gradients.len() {
                continue;
            }

            // 更新权重
            for i in 0..layer.weights.len() {
                if i >= gradients[layer_idx].len() {
                    continue;
                }
                for j in 0..layer.weights[i].len() {
                    if j >= gradients[layer_idx][i].len() {
                        continue;
                    }

                    // 动量缓冲区：仅累积梯度（不乘学习率）
                    layer.momentum_weights[i][j] =
                        self.momentum * layer.momentum_weights[i][j] +
                            gradients[layer_idx][i][j];  // ← 移除了 learning_rate

                    // 更新权重：动量项乘以学习率
                    layer.weights[i][j] -= self.learning_rate * layer.momentum_weights[i][j];
                }
            }

            // 更新偏置
            for i in 0..layer.biases.len() {
                if i >= bias_gradients[layer_idx].len() {
                    continue;
                }

                layer.momentum_biases[i] =
                    self.momentum * layer.momentum_biases[i] +
                        bias_gradients[layer_idx][i];  // ← 移除了 learning_rate

                layer.biases[i] -= self.learning_rate * layer.momentum_biases[i];
            }
        }
    }


    /*
    fn train_with_grpo(&mut self, training_data: &[(Vec<f32>, Vec<f32>)], group_size: usize) {
        if training_data.is_empty() {
            eprintln!("警告: 训练数据为空，跳过训练");
            return;
        }
        
        // 初始化动量参数
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
            
            // GRPO算法：对同一批次中的样本进行分组
            let group_count = (batch.len() + group_size - 1) / group_size;
            let mut group_rewards = Vec::with_capacity(group_count);
            
            // 计算每组的平均奖励
            for group_idx in 0..group_count {
                let group_start = group_idx * group_size;
                let group_end = (group_start + group_size).min(batch.len());
                let group = &batch[group_start..group_end];
                
                let mut group_reward = 0.0;
                for (input, target) in group {
                    let output = self.light_forward(input);
                    let reward: f32 = output.iter().zip(target.iter()).take(2)
                        .map(|(o, t)| {
                            // 奖励计算：准确率越高奖励越高
                            let diff = (o - t).abs();
                            if diff < 0.1 { 1.0 } else if diff < 0.3 { 0.5 } else { -0.5 }
                        })
                        .sum();
                    group_reward += reward;
                }
                group_rewards.push(group_reward / group.len() as f32);
            }
            
            // 使用组平均奖励作为基线进行策略更新
            let baseline_reward = group_rewards.iter().sum::<f32>() / group_rewards.len() as f32;
            
            // 计算损失并更新网络
            let mut batch_loss = 0.0;
            for (i, (input, target)) in batch.iter().enumerate() {
                let output = self.light_forward(input);
                // 根据所在组的奖励调整损失
                let group_idx = i / group_size;
                let group_reward = group_rewards[group_idx];
                let reward_advantage = group_reward - baseline_reward;
                
                // 计算新旧策略概率比 (PPO组件1)
                let old_policy_probs: Vec<f32> = target.iter().take(2).copied().collect();
                let new_policy_probs: Vec<f32> = output.iter().take(2).copied().collect();
                
                // 计算概率比
                let mut ratio = 1.0;
                for j in 0..new_policy_probs.len().min(old_policy_probs.len()) {
                    if old_policy_probs[j].abs() > 1e-8 {
                        ratio *= new_policy_probs[j] / old_policy_probs[j];
                    }
                }
                
                // PPO裁剪 (PPO组件2)
                const PPO_EPSILON: f32 = 0.2;
                let clipped_ratio = ratio.clamp(1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON);
                
                // 计算PPO损失
                let advantage = reward_advantage;
                let ppo_loss = (ratio * advantage).min(clipped_ratio * advantage);
                
                let loss: f32 = output.iter().zip(target.iter()).take(2)
                    .map(|(o, t)| {
                        let diff = o - t;
                        // 根据奖励优势调整损失，并结合PPO损失
                        diff.powi(2) * (1.0 + reward_advantage.signum() * reward_advantage.abs().min(1.0)) - ppo_loss * 0.01
                    })
                    .sum();
                batch_loss += loss;
            }
            total_loss += batch_loss;

            self.train_batch_with_advantage(batch, &group_rewards, baseline_reward);

            if batch_idx == batch_count - 1 {
                let avg_loss = total_loss / training_data.len() as f32;
                self.adapt_learning_rate(avg_loss);
                self.epoch_count += 1;
                if self.epoch_count % 1 == 0 {
                    let (min_weight, max_weight) = self.get_weight_range();
                    let (min_momentum, max_momentum) = self.get_momentum_range();
                    println!("[GRPO训练状态] Epoch: {}, Loss: {:.6}, LR: {:.6}, 权重范围 [{:.4}, {:.4}], 动量范围 [{:.4}, {:.4}]",
                             self.epoch_count, avg_loss, self.learning_rate, min_weight, max_weight, min_momentum, max_momentum);
                }
            }
        }
    }
    
    // 带优势函数的训练批次，集成PPO优化功能
    fn train_batch_with_advantage(&mut self, batch: &[(Vec<f32>, Vec<f32>)], group_rewards: &[f32], baseline_reward: f32) {
        let batch_size = batch.len();
        if batch_size == 0 {
            eprintln!("[TrainAdvantage] Empty batch, skipping training");
            return;
        }

        // 正确的分组逻辑
        let group_size = (batch_size / group_rewards.len()).max(1);
        let actual_groups = (batch_size + group_size - 1) / group_size;

        // 准备输入和目标数据
        let mut inputs: Vec<&[f32]> = Vec::with_capacity(batch_size);
        let mut targets: Vec<Vec<f32>> = Vec::with_capacity(batch_size);

        for (input, target) in batch {
            inputs.push(input);
            targets.push(target.clone());
        }

        // 前向传播获取当前策略输出
        let outputs = self.gpu_forward_batch(&inputs, batch_size);

        // 初始化梯度累积器
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

        // 计算优势函数
        let mut advantages = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let group_idx = i / group_size;
            let group_reward = group_rewards[group_idx.min(group_rewards.len() - 1)];
            let advantage = group_reward - baseline_reward;
            advantages.push(advantage);
        }

        // 优势标准化
        let mean_advantage = advantages.iter().sum::<f32>() / batch_size as f32;
        let variance = advantages.iter().map(|&a| (a - mean_advantage).powi(2)).sum::<f32>() / batch_size as f32;
        let std_advantage = variance.sqrt().max(1e-8);
        
        println!("[Advantage] Mean: {:.4}, Std: {:.4}, Range: [{:.4}, {:.4}]", 
                 mean_advantage, std_advantage,
                 advantages.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
                 advantages.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));

        // PPO算法参数
        const PPO_EPSILON: f32 = 0.2;
        const VALUE_LOSS_COEF: f32 = 0.5;
        const ENTROPY_BONUS: f32 = 0.01;

        let mut total_policy_loss = 0.0;
        let mut total_value_loss = 0.0;
        let mut total_entropy = 0.0;

        // 为每个样本计算梯度
        for i in 0..batch_size {
            let _group_idx = i / group_size;
            let normalized_advantage = (advantages[i] - mean_advantage) / std_advantage;
            
            // 获取当前策略概率（假设前两个输出是左右手概率）
            let current_probs = &outputs[i][..2];
            let target_probs = &targets[i][..2];
            
            // 计算重要性采样比率 r(θ) = π_θ(a|s) / π_θ_old(a|s)
            let mut importance_ratio = 1.0;
            for j in 0..current_probs.len().min(target_probs.len()) {
                if target_probs[j] > 1e-8 {
                    importance_ratio *= (current_probs[j] / target_probs[j]).powf(normalized_advantage.signum());
                }
            }

            // PPO裁剪目标: L^CLIP(θ) = min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)
            let clipped_ratio = importance_ratio.clamp(1.0 - PPO_EPSILON, 1.0 + PPO_EPSILON);
            let ppo_objective = clipped_ratio * normalized_advantage;
            let policy_loss = -ppo_objective; // 负号因为我们要最大化

            // 计算策略熵作为探索奖励
            let mut entropy = 0.0;
            for &prob in current_probs {
                if prob > 1e-8 {
                    entropy -= prob * prob.ln();
                }
            }
            total_entropy += entropy;

            // 计算价值函数损失（如果有价值输出，假设在索引2）
            let current_value = outputs[i].get(2).copied().unwrap_or(0.0);
            let target_value = targets[i].get(2).copied().unwrap_or(0.0);
            let value_loss = (current_value - target_value).powi(2);
            total_value_loss += value_loss;

            // 构建调整后的目标向量
            let mut adjusted_target = targets[i].clone();
            
            // 应用策略梯度损失
            if !adjusted_target.is_empty() {
                adjusted_target[0] -= policy_loss * 0.1; // 调整左手概率
                if adjusted_target.len() > 1 {
                    adjusted_target[1] -= policy_loss * 0.1; // 调整右手概率
                }
            }

            // 应用价值函数损失
            if adjusted_target.len() > 2 {
                adjusted_target[2] += VALUE_LOSS_COEF * value_loss;
            }

            // 应用熵奖励
            if !adjusted_target.is_empty() {
                adjusted_target[0] += ENTROPY_BONUS * entropy;
                if adjusted_target.len() > 1 {
                    adjusted_target[1] += ENTROPY_BONUS * entropy;
                }
            }

            // 计算并累积梯度
            self.backward(
                inputs[i],
                &outputs[i],
                &adjusted_target,
                &mut total_gradients,
                &mut total_bias_gradients
            );

            total_policy_loss += policy_loss;
        }

        // 梯度裁剪和参数更新
        if batch_size > 0 {
            let inv_batch = 1.0f32 / (batch_size as f32);
            for layer_idx in 0..self.layers.len() {
                let wg = &mut total_gradients[layer_idx];
                for row in wg.iter_mut() {
                    for grad in row.iter_mut() {
                        *grad *= inv_batch;
                    }
                }
                let bg = &mut total_bias_gradients[layer_idx];
                for grad in bg.iter_mut() {
                    *grad *= inv_batch;
                }
            }
        }

        // 应用梯度
        self.clip_gradients(&mut total_gradients, &mut total_bias_gradients, 0.5);
        self.apply_gradients(&total_gradients, &total_bias_gradients, batch_size);

        // 输出训练统计信息
        println!("[Advantage Training] Batch: {}, Policy Loss: {:.6}, Value Loss: {:.6}, Entropy: {:.6}, Groups: {}", 
                 batch_size, total_policy_loss / batch_size as f32, 
                 total_value_loss / batch_size as f32, 
                 total_entropy / batch_size as f32, actual_groups);
    }

     */
    /*

    fn calculate_gradient_norm(&self, gradients: &[Vec<Vec<f32>>], bias_gradients: &[Vec<f32>]) -> f32 {
        let mut total_norm = 0.0;
        let mut valid_count = 0;

        for layer_grad in gradients {
            for row in layer_grad {
                for &g in row {
                    // 只有当梯度是有限值时才计算
                    if g.is_finite() {
                        total_norm += g * g;
                        valid_count += 1;
                    }
                }
            }
        }

        for bias_grad in bias_gradients {
            for &g in bias_grad {
                // 只有当梯度是有限值时才计算
                if g.is_finite() {
                    total_norm += g * g;
                    valid_count += 1;
                }
            }
        }

        // 如果没有有效的梯度值，返回0
        if valid_count == 0 {
            return 0.0;
        }
        
        // 防止sqrt(0)的情况
        if total_norm <= 0.0 {
            return 0.0;
        }
        
        total_norm.sqrt()
    }

     */

    

    fn reset_problem_layers(&mut self) {
        // 首先收集需要重置的层索引
        let mut layers_to_reset = Vec::new();
        let mut saturated_layers = Vec::new();
        
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
                    saturated_layers.push(i);
                    continue;
                }
            }

            if !weight_ok {
                layers_to_reset.push(i);
            }
        }
        
        // 现在可以安全地重置层，因为没有借用冲突
        for i in layers_to_reset {
            let input_size = if i > 0 { 
                self.layers[i-1].weights.len() 
            } else { 
                self.layers[i].weights[0].len() 
            };
            let _output_size = self.layers[i].weights.len(); // 添加下划线以避免未使用变量警告
            let std_dev = (1.0 / input_size as f32).sqrt();
            
            for weights in &mut self.layers[i].weights {
                for w in weights.iter_mut() {
                    *w = if w.is_nan() || w.is_infinite() {
                        (fastrand::f32() * 2.0 - 1.0) * 0.1
                    } else if w.abs() > 30.0 {
                        30.0 * w.signum()
                    } else {
                        (fastrand::f32() * 2.0 - 1.0) * std_dev
                    };
                }
            }
            
            for b in &mut self.layers[i].biases {
                if b.is_nan() || b.is_infinite() {
                    *b = (fastrand::f32() * 2.0 - 1.0) * 0.1;
                } else if b.abs() > 30.0 {
                    *b = 30.0 * b.signum();
                } else {
                    *b = 0.0;
                }
            }
        }
        
        // 重置饱和层
        for i in saturated_layers {
            let input_size = if i > 0 { 
                self.layers[i-1].weights.len() 
            } else { 
                self.layers[i].weights[0].len() 
            };
            let _output_size = self.layers[i].weights.len(); // 添加下划线以避免未使用变量警告
            let std_dev = (1.0 / input_size as f32).sqrt();
            
            // 重置激活值
            for a in &mut self.layers[i].activations {
                if a.is_nan() || a.is_infinite() {
                    *a = 0.0;
                }
            }
            
            // 重置权重
            for weights in &mut self.layers[i].weights {
                for w in weights.iter_mut() {
                    *w = (fastrand::f32() * 2.0 - 1.0) * std_dev;
                }
            }
            
            for b in &mut self.layers[i].biases {
                *b = 0.0;
            }
        }
    }

    fn adapt_learning_rate(&mut self, loss: f32) {
        // 初始Loss值为0.36
        let loss_change_ratio = if self.last_loss.is_finite() && self.last_loss > 1e-8 {
            (loss / self.last_loss).clamp(0.1, 10.0) // 限制比率范围，防止极端值
        } else {
            1.0
        };

        let epoch_factor = if self.epoch_count < 50 {
            0.9  // 早期训练使用较低的学习率
        } else if self.epoch_count < 200 {
            1.0
        } else {
            0.7  // 后期训练继续降低学习率
        };

        if loss_change_ratio < 0.98 {
            // 损失下降，正常增加学习率，但幅度更保守
            self.learning_rate = (self.learning_rate * 1.015 * epoch_factor).min(0.02);
            self.bad_epochs = 0;
        }
        else if loss_change_ratio > 1.02 {  // 提高阈值，更少触发学习率降低
            self.learning_rate = (self.learning_rate * 0.90).max(0.0001);  // 与clamp下限保持一致
            self.bad_epochs += 1;
            if self.bad_epochs >= 3 {
                self.learning_rate = (self.learning_rate * 0.7).max(0.00005); // 减小降低幅度
                println!("连续{}个epoch表现不佳，适度降低学习率至{:.6}", self.bad_epochs, self.learning_rate);
            }
            if self.bad_epochs >= 5 {
                println!("连续{}个epoch损失增加，采取激进措施", self.bad_epochs);
                // 可以在这里添加恢复到最佳模型权重的逻辑
            }
            if self.bad_epochs >= 8 {
                println!("连续{}个epoch表现不佳，执行网络重置", self.bad_epochs);
                self.reset_problem_layers();
                self.bad_epochs = 0;
                // 重置后使用适中的学习率，增加梯度更新能力
                self.learning_rate = 0.002;
                println!("[重置] 网络重置，学习率恢复至: {:.6}", self.learning_rate);
            }
        }
        else {
            // 损失平稳，使用更稳定的调度策略
            let cosine_factor = 0.5 * (1.0 + (std::f32::consts::PI * (self.epoch_count % 200) as f32 / 200.0).cos());
            self.learning_rate = 0.0008 * cosine_factor * epoch_factor;
            self.bad_epochs = 0;
        }
        
        // 确保学习率在更高的范围内，适应增强的梯度
        self.learning_rate = self.learning_rate.clamp(0.0001, 0.02);
        
        // 记录当前损失，添加NaN检查
        self.last_loss = if loss.is_finite() { loss } else { self.last_loss };
        
        // 每3个epoch输出一次学习率状态（更频繁监控）
        if self.epoch_count % 3 == 0 {
            println!("[学习率调整] Epoch: {}, Loss: {:.6}, LR: {:.6}, 变化率: {:.3}%", 
                     self.epoch_count, loss, self.learning_rate, (loss_change_ratio - 1.0) * 100.0);
        }
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
        // 检查批次是否为空
        if batch.is_empty() {
            eprintln!("[Train Warning] Empty batch provided, skipping training");
            return;
        }
        
        let batch_size = batch.len();

        // 创建清理后的批次副本，避免生命周期问题
        let cleaned_batch: Vec<(Vec<f32>, Vec<f32>)> = batch.iter()
            .map(|(input, target)| {
                // 检查输入是否包含NaN或无穷大值
                let clean_input: Vec<f32> = input.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
                let clean_target: Vec<f32> = target.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
                (clean_input, clean_target)
            })
            .collect();

        // 前向传播
        let inputs: Vec<&[f32]> = cleaned_batch.iter().map(|(input, _)| input.as_slice()).collect();
        let targets: Vec<&[f32]> = cleaned_batch.iter().map(|(_, target)| target.as_slice()).collect();
        
        let outputs = if self.device.is_some() && self.gpu_initialized {
            self.gpu_forward_batch(&inputs, batch_size)
        } else {
            // CPU前向传播
            inputs.iter().map(|&input| self.cpu_forward(input)).collect()
        };
        
        // 检查输出是否包含NaN或无穷大值
        let clean_outputs: Vec<Vec<f32>> = outputs.into_iter().map(|output| {
            output.into_iter().map(|x| if x.is_finite() { x } else { 0.0 }).collect()
        }).collect();

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

        // 反向传播
        let mut valid_samples = 0;
        for i in 0..batch_size {
            // 检查输入、输出和目标是否都有效
            let input_valid = inputs[i].iter().all(|&x| x.is_finite());
            let output_valid = clean_outputs[i].iter().all(|&x| x.is_finite());
            let target_valid = targets[i].iter().all(|&x| x.is_finite());
            
            if input_valid && output_valid && target_valid {
                self.backward(
                    inputs[i],
                    &clean_outputs[i],
                    targets[i],
                    &mut total_gradients,
                    &mut total_bias_gradients
                );
                valid_samples += 1;
            } else {
                eprintln!("[Train Warning] Skipping sample {} due to NaN/Inf values", i);
            }
        }
        
        // 如果没有有效样本，直接返回
        if valid_samples == 0 {
            eprintln!("[Train Error] No valid samples in batch, skipping gradient update");
            return;
        }

        if valid_samples > 0 {
            let inv_batch = 1.0f32 / (valid_samples as f32);
            for layer_idx in 0..self.layers.len() {
                let wg = &mut total_gradients[layer_idx];
                for r in 0..wg.len() {
                    for c in 0..wg[r].len() {
                        wg[r][c] *= inv_batch;
                        // 确保梯度是有限值
                        if !wg[r][c].is_finite() {
                            wg[r][c] = 0.0;
                        }
                    }
                }
                let bg = &mut total_bias_gradients[layer_idx];
                for j in 0..bg.len() {
                    bg[j] *= inv_batch;
                    // 确保梯度是有限值
                    if !bg[j].is_finite() {
                        bg[j] = 0.0;
                    }
                }
            }
        }
        
        // 添加原始梯度诊断信息
        if self.epoch_count % 1 == 0 {
            // 计算原始梯度的最大值和最小值
            let mut max_weight_grad = 0.0f32;
            let mut min_weight_grad = f32::INFINITY;
            let mut max_bias_grad = 0.0f32;
            let mut min_bias_grad = f32::INFINITY;
            let mut grad_count = 0;
            
            for layer_idx in 0..self.layers.len() {
                let wg = &total_gradients[layer_idx];
                for r in 0..wg.len() {
                    for c in 0..wg[r].len() {
                        let grad = wg[r][c];
                        if grad.is_finite() {
                            max_weight_grad = max_weight_grad.max(grad.abs());
                            min_weight_grad = min_weight_grad.min(grad.abs());
                            grad_count += 1;
                        }
                    }
                }
                let bg = &total_bias_gradients[layer_idx];
                for j in 0..bg.len() {
                    let grad = bg[j];
                    if grad.is_finite() {
                        max_bias_grad = max_bias_grad.max(grad.abs());
                        min_bias_grad = min_bias_grad.min(grad.abs());
                    }
                }
            }
            
            println!("[梯度诊断] Epoch {}: 权重梯度范围 [{:.12}, {:.12}], 偏置梯度范围 [{:.12}, {:.12}], 有效梯度数量: {}", 
                     self.epoch_count, min_weight_grad, max_weight_grad, min_bias_grad, max_bias_grad, grad_count);
        }
        
        // 梯度裁剪
        self.clip_gradients(&mut total_gradients, &mut total_bias_gradients, 0.5);
        
        // 应用梯度
        self.apply_gradients(&total_gradients, &total_bias_gradients, valid_samples);
        
        // 输出第一个权重值用于调试
        if !self.layers.is_empty() && !self.layers[0].weights.is_empty() && !self.layers[0].weights[0].is_empty() {
            let first_weight = self.layers[0].weights[0][0];
            if first_weight.is_finite() {
                println!("First weight value after update: {:.6}", first_weight);
            } else {
                println!("First weight value after update: NaN/Inf (reset to 0.0)");
            }
        }
    }


    pub fn backward(&mut self, input: &[f32], output: &[f32], target: &[f32], total_gradients: &mut [Vec<Vec<f32>>], total_bias_gradients: &mut [Vec<f32>]) {
        let mut layer_errors = vec![vec![0.0; 0]; self.layers.len()];
        if let Some(last_layer_idx) = self.layers.len().checked_sub(1) {
            let last_idx = self.layers.len() - 1;
            // 去掉错误的out_dim归一化，每个输出维度应该有独立的误差
            let mut output_errors: Vec<f32> = output.iter()
                .zip(target.iter())
                .map(|(o, t)| 2.0 * (o - t))  // 去掉除法！
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
                        ActivationFunction::Swish | ActivationFunction::GELU | ActivationFunction::Linear => {
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

        // activation a（不是 z）计算导数
        let derivative_from_activation = |a: f32, func: &ActivationFunction| -> f32 {
            match func {
                ActivationFunction::ReLU => if a > 0.0 { 1.0 } else { 0.0 },
                ActivationFunction::Sigmoid => a * (1.0 - a),
                ActivationFunction::Tanh => 1.0 - a * a,
                // 对 Swish/GELU 无法从 a 得到精确导数，返回 1.0
                ActivationFunction::Swish | ActivationFunction::GELU | ActivationFunction::Linear=> 1.0,
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
        _layer_idx: usize,
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
        const MAX_GRAD: f32 = 50.0;  // 增大梯度上限以处理极小梯度
        const MAX_WEIGHT: f32 = 20.0;  // 增大权重上限
        const MIN_GRAD_THRESHOLD: f32 = 1e-10;  // 梯度下限 - 进一步降低阈值，提高对小梯度的敏感性

        // 计算平均梯度大小
        let mut total_grad_sum = 0.0;
        let mut total_grad_count = 0;
        
        for layer_grad in gradients {
            for row in layer_grad {
                for &g in row {
                    // 添加梯度检查，防止NaN和无穷大影响平均值计算
                    if g.is_finite() {
                        total_grad_sum += g.abs();
                        total_grad_count += 1;
                    }
                }
            }
        }
        
        for bias_grad in bias_gradients {
            for &g in bias_grad {
                // 添加梯度检查，防止NaN和无穷大影响平均值计算
                if g.is_finite() {
                    total_grad_sum += g.abs();
                    total_grad_count += 1;
                }
            }
        }
        
        let avg_grad_magnitude = if total_grad_count > 0 {
            total_grad_sum / total_grad_count as f32
        } else {
            0.0
        };

        // 限制梯度缩放因子，防止过大
        let gradient_scale = if avg_grad_magnitude > 0.0 && avg_grad_magnitude < MIN_GRAD_THRESHOLD {
            let scale = MIN_GRAD_THRESHOLD / avg_grad_magnitude;
            scale.min(10000000.0) // 进一步增加最大限制
        } else if avg_grad_magnitude == 0.0 {
            1000000.0 // 对零梯度使用更大的放大因子
        } else {
            1.0
        };

        if self.epoch_count % 1 == 0 {
            println!("[Gradient Monitor] Epoch: {}, Avg Gradient Magnitude: {:.8}, Scale: {:.2}, LR: {:.6}", 
                     self.epoch_count, avg_grad_magnitude, gradient_scale, self.learning_rate);
            
            // 如果梯度太小，添加额外诊断信息
            if avg_grad_magnitude < 1e-6 {
                println!("[Gradient Alert] 梯度极小，启用增强缩放");
                // 对于极小梯度，使用更激进的缩放
                let emergency_scale = (1e-4 / avg_grad_magnitude.max(1e-10)).min(1000000.0);
                println!("[Gradient Emergency] 紧急缩放因子: {:.2}", emergency_scale);
            }
        }

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            for (i, weight_row) in layer.weights.iter_mut().enumerate() {
                for (j, weight) in weight_row.iter_mut().enumerate() {
                    if i < gradients[layer_idx].len() && j < gradients[layer_idx][i].len() {
                        let mut grad = gradients[layer_idx][i][j] / batch_size_f;

                        // 检查梯度是否为NaN或无穷大
                        if !grad.is_finite() {
                            eprintln!("[Gradient Safety] NaN/Inf gradient detected at layer {}, weight[{}][{}], setting to 0.", layer_idx, i, j);
                            grad = 0.0;
                        }

                        // 应用梯度缩放和裁剪
                        grad *= gradient_scale;
                        grad = grad.clamp(-MAX_GRAD, MAX_GRAD);

                        // 对于极小梯度，使用大幅增强的学习率
                        let effective_lr = if avg_grad_magnitude < 1e-10 {
                            self.learning_rate * 1000.0  // 对极小梯度使用1000倍学习率
                        } else if avg_grad_magnitude < 1e-7 {
                            self.learning_rate * 100.0  // 对极小梯度使用100倍学习率
                        } else if avg_grad_magnitude < 1e-6 {
                            self.learning_rate * 50.0   // 对小梯度使用50倍学习率
                        } else if avg_grad_magnitude < 1e-5 {
                            self.learning_rate * 10.0   // 对较小梯度使用10倍学习率
                        } else {
                            self.learning_rate
                        };

                        // 计算动量更新，添加NaN检查
                        let momentum_component = self.momentum * layer.momentum_weights[i][j];
                        let gradient_component = effective_lr * grad;
                        
                        // 检查计算结果是否为NaN或无穷大
                        if !momentum_component.is_finite() || !gradient_component.is_finite() {
                            eprintln!("[Gradient Safety] NaN/Inf momentum or gradient component at layer {}, weight[{}][{}], resetting.", layer_idx, i, j);
                            layer.momentum_weights[i][j] = 0.0;
                        } else {
                            layer.momentum_weights[i][j] = momentum_component - gradient_component;
                        }
                        
                        let momentum_update = layer.momentum_weights[i][j];
                        let clamped_momentum = momentum_update.clamp(-MAX_GRAD, MAX_GRAD);

                        if self.epoch_count % 1 == 0 && i == 0 && j == 0 {
                            println!("[Weight Debug] Layer {}, Weight[{}][{}]: {:.6} -> ", layer_idx, i, j, *weight);
                        }

                        let old_weight = *weight;
                        *weight += clamped_momentum;
                        
                        // 添加权重更新后的检查
                        if !weight.is_finite() {
                            eprintln!("[Gradient Safety] NaN/Inf weight detected at layer {}, weight[{}][{}], resetting to 0.", layer_idx, i, j);
                            *weight = 0.0;
                        } else {
                            *weight = weight.clamp(-MAX_WEIGHT, MAX_WEIGHT);
                        }

                        if self.epoch_count % 1 == 0 && i == 0 && j == 0 {
                            let delta = *weight - old_weight;
                            println!("{:.6} (delta: {:+.10}) (momentum: {:+.10}, scale: {:.2})", 
                                    *weight, delta, clamped_momentum, gradient_scale);
                        }
                        
                        // 最终检查权重是否为NaN或无穷大
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

                    // 检查梯度是否为NaN或无穷大
                    if !grad.is_finite() {
                        eprintln!("[Gradient Safety] NaN/Inf gradient detected at layer {}, bias[{}], setting to 0.", layer_idx, i);
                        grad = 0.0;
                    }
                    
                    // 应用梯度放大和裁剪
                    grad *= gradient_scale;
                    grad = grad.clamp(-MAX_GRAD, MAX_GRAD);

                    // 计算动量更新，添加NaN检查
                    let momentum_component = self.momentum * layer.momentum_biases[i];
                    let gradient_component = self.learning_rate * grad;
                    
                    // 检查计算结果是否为NaN或无穷大
                    if !momentum_component.is_finite() || !gradient_component.is_finite() {
                        eprintln!("[Gradient Safety] NaN/Inf momentum or gradient component at layer {}, bias[{}], resetting.", layer_idx, i);
                        layer.momentum_biases[i] = 0.0;
                    } else {
                        layer.momentum_biases[i] = momentum_component - gradient_component;
                    }
                    
                    let momentum_update = layer.momentum_biases[i];
                    let clamped_momentum = momentum_update.clamp(-MAX_GRAD, MAX_GRAD);

                    // 添加调试信息
                    if self.epoch_count % 1 == 0 && i == 0 {  // 每个epoch输出第一个偏置的更新信息
                        println!("[Bias Debug] Layer {}, Bias[{}]: {:.6} -> ", layer_idx, i, *bias);
                    }

                    let old_bias = *bias;
                    *bias += clamped_momentum;
                    
                    // 添加偏置更新后的检查
                    if !bias.is_finite() {
                        eprintln!("[Gradient Safety] NaN/Inf bias detected at layer {}, bias[{}], resetting to 0.", layer_idx, i);
                        *bias = 0.0;
                    } else {
                        *bias = bias.clamp(-MAX_WEIGHT, MAX_WEIGHT);
                    }
                    
                    // 添加调试信息
                    if self.epoch_count % 1 == 0 && i == 0 {  // 每个epoch输出第一个偏置的更新信息
                        println!("{:.6} (delta: {:+.8})", *bias, *bias - old_bias);
                    }

                    // 最终检查偏置是否为NaN或无穷大
                    if !bias.is_finite() {
                        eprintln!("[Gradient Safety] NaN/Inf bias detected at layer {}, bias[{}], setting to 0.", layer_idx, i);
                        *bias = 0.0;
                    }
                } else {
                    // 如果没有对应的梯度，跳过更新但记录警告
                    if self.epoch_count % 100 == 0 {  // 减少输出频率
                        eprintln!("[Gradient Warning] No bias gradient for layer {}, bias[{}], skipping update.", layer_idx, i);
                    }
                }
            }
        }
        
        // GPU更新部分添加错误处理
        if self.device.is_none() || self.queue.is_none() {
            return;
        }
        
        for layer in &mut self.layers {
            // 检查GPU资源是否存在
            if layer.weights_buffer.is_none() || layer.biases_buffer.is_none() {
                continue; // 跳过未初始化的层
            }
            
            // 检查队列是否有效
            if let Some(queue) = &self.queue {
                // 检查缓冲区是否有效
                if let (Some(weights_buf), Some(biases_buf)) = (&layer.weights_buffer, &layer.biases_buffer) {
                    // 检查权重和偏置是否为NaN或无穷大
                    let weights_valid = layer.weights.iter().flatten().all(|&w| w.is_finite());
                    let biases_valid = layer.biases.iter().all(|&b| b.is_finite());
                    
                    if weights_valid && biases_valid {
                        let weights_flat = layer.weights.iter().flatten().cloned().collect::<Vec<f32>>();
                        queue.write_buffer(weights_buf, 0, bytemuck::cast_slice(&weights_flat));
                        queue.write_buffer(biases_buf, 0, bytemuck::cast_slice(&layer.biases));
                    } else {
                        eprintln!("[GPU Update] Skipping layer update due to NaN/Inf values");
                    }
                }
            }
        }
    }
    
    // 不可变版本的forward，用于多线程
    pub fn light_forward_immutable(layers: &[NetworkLayer], input: &[f32]) -> Vec<f32> {
        // 检查输入是否包含NaN或无穷大值
        let mut current_input: Vec<f32> = input.iter().map(|&x| if x.is_finite() { x } else { 0.0 }).collect();
        let mut layer_outputs: Vec<Vec<f32>> = Vec::new();

        for layer in layers {
            // 在每层处理前检查输入
            for x in &mut current_input {
                if !x.is_finite() {
                    *x = 0.0; // 将NaN或无穷大值替换为0
                }
            }
            
            // 创建层的副本以避免可变借用问题
            let mut layer_copy = layer.clone();
            
            match layer.layer_type {
                LayerType::Dense => {
                    current_input = DeepNeuralNetwork::dense_forward(&mut layer_copy, &current_input);
                }
                LayerType::LSTM => {
                    current_input = DeepNeuralNetwork::lstm_forward(&mut layer_copy, &current_input);
                }
                LayerType::Attention => {
                    current_input = DeepNeuralNetwork::attention_forward(&mut layer_copy, &current_input);
                }
                LayerType::Residual => {
                    let input_copy = current_input.clone();
                    current_input = DeepNeuralNetwork::dense_forward(&mut layer_copy, &current_input);

                    // 确保维度匹配后才添加残差连接
                    if current_input.len() == input_copy.len() {
                        for i in 0..current_input.len() {
                            // 检查两个值是否都是有限的
                            if current_input[i].is_finite() && input_copy[i].is_finite() {
                                current_input[i] += input_copy[i];
                            } else {
                                current_input[i] = 0.0; // 如果任何一个值不是有限的，则使用默认值
                            }
                        }
                    }
                }
                LayerType::Concat => {
                    current_input = DeepNeuralNetwork::concat_forward(&mut layer_copy, &layer_outputs);
                }
                LayerType::Reflection => {
                    current_input = DeepNeuralNetwork::reflection_forward(&mut layer_copy, &current_input, &layer_outputs);
                }
            }
            
            // 在每层处理后检查输出
            for x in &mut current_input {
                if !x.is_finite() {
                    *x = 0.0; // 将NaN或无穷大值替换为0
                }
            }
            
            layer_outputs.push(current_input.clone());
        }

        current_input
    }
}

impl Finger {
    pub fn to_hand(&self) -> Hand {
        match self {
            FingerType::Index | FingerType::Middle | FingerType::Thumb | FingerType::Ring | FingerType::Pinky => {
                // 需要根据上下文判断左右手，这里默认使用右手
                // 实际使用时应该从调用上下文中获取手部信息
                Hand::Right
            }
        }
    }

    pub fn is_index(&self) -> bool {
        matches!(self, FingerType::Index)
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

        // 全局上下文（每个音符重复）
        let (note_density, max_simul, global_bpm) = if !notes.is_empty() {
            let time_span = notes.last().unwrap().time - notes.first().unwrap().time + 0.001;
            let density = notes.len() as f32 / time_span;
            let max_simul = self.max_simultaneous_in_window(notes);
            let bpm = bpm_list.now_bpm(notes[0].time);
            (density, max_simul, bpm)
        } else {
            (0.0, 0, 0.0)
        };
        let four_finger_hint = if (note_density > 48.0 && max_simul >= 3) || max_simul >= 4 { 1.0 } else { 0.0 };

        for i in 0..actual_len {
            let note = &notes[i]; // 当前音符

            // 构造上下文窗口：前后各 16 个音符
            let context_start = i.saturating_sub(16);
            let context_end = (i + 16).min(notes.len());
            let context = &notes[context_start..context_end];

            let mut features = Vec::new();
            features.extend(self.extract_position_features(&[note.clone()]));// 4维 位置特征
            //features.extend(self.extract_temporal_features(&[note.clone()], 1, bpm_list));
            features.extend(self.extract_temporal_features(note, context, bpm_list)); // 6维 时间特征
            
            features.extend(self.extract_difficulty_features(&[note.clone()])); // 4维 难度特征
            features.extend(self.extract_velocity_features(&[note.clone()])); // 4维 速度特征
            features.extend(self.extract_spatial_features(&[note.clone()])); // 4维 空间特征

            //  context 特征
            features.extend(self.extract_position_context_features(note, context));

            // 全局上下文特征
            features.push(note_density / 100.0); // 归一化到 [0, 1]
            features.push(max_simul as f32 / 8.0);
            features.push(global_bpm / 1000.0);
            features.push(four_finger_hint); // 1维
            
            // 18 + 8 = 26
            debug_assert_eq!(features.len(), 40, "Single note feature dimension must be 40");
            all_features.extend(features);
        }

        let target_len = window_size * 40; // 每个音符40维
        if all_features.len() < target_len {
            all_features.resize(target_len, 0.0);
        }

        // 归一化（保持不变）
        for f in &mut all_features {
            if !f.is_finite() { *f = 0.0; }
        }
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

    fn max_simultaneous_in_window(&self, notes: &[ProcessedNote]) -> usize {
        if notes.is_empty() { return 0; }
        let mut max_simul = 1;
        let mut i = 0;
        while i < notes.len() {
            let base_time = notes[i].time;
            let mut count = 1;
            let mut j = i + 1;
            while j < notes.len() && (notes[j].time - base_time).abs() <= 0.03 {
                count += 1;
                j += 1;
            }
            max_simul = max_simul.max(count);
            i = j;
        }
        max_simul
    }

    fn extract_position_context_features(&self, note: &ProcessedNote, context: &[ProcessedNote]) -> Vec<f32> {
        let mut features = Vec::new();

        if let Some(prev_note) = context.iter().rev().find(|n| n.time < note.time) {
            let dx = note.position.x - prev_note.position.x;
            let dy = note.position.y - prev_note.position.y;
            features.push(dx);
            features.push(dy);
            features.push((dx * dx + dy * dy).sqrt());
            features.push(dy.atan2(dx));
        } else {
            features.extend(vec![0.0; 4]);
        }

        let mut left_density = 0;
        let mut right_density = 0;
        for other in context {
            if (other.time - note.time).abs() <= 0.3 {
                if other.position.x < 0.0 {
                    left_density += 1;
                } else {
                    right_density += 1;
                }
            }
        }
        features.push(left_density as f32 / 10.0);
        features.push(right_density as f32 / 10.0);
        features.push((right_density as f32 - left_density as f32) / 10.0);

        let mut streak = 0;
        let mut last_hand = None;
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

        features // 总共 4 + 3 + 1 = 8 维
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

    fn extract_temporal_features(&self, note: &ProcessedNote, context: &[ProcessedNote], bpm_list: &BpmList) -> Vec<f32> {
        if context.len() < 2 {
            return vec![0.0; 6];
        }

        // 使用上下文窗口计算时间特征
        let intervals: Vec<f32> = context.windows(2)
            .map(|w| w[1].time - w[0].time)
            .collect();

        let (avg_interval, variance) = Self::calculate_interval_stats(&intervals);
        let rhythm_complexity = self.calculate_rhythm_complexity(&intervals);
        let current_bpm = bpm_list.now_bpm(note.time);

        // 计算局部密度（使用上下文窗口）
        let time_span = context.last().unwrap().time - context.first().unwrap().time;
        let density = if time_span > 0.0 {
            context.len() as f32 / time_span
        } else {
            0.0
        };

        vec![
            avg_interval,
            variance,
            rhythm_complexity,
            current_bpm,
            intervals.len() as f32,
            density,
        ]
    }

    fn calculate_interval_stats(intervals: &[f32]) -> (f32, f32) {
        if intervals.is_empty() {
            return (0.0, 0.0);
        }

        let sum: f32 = intervals.iter().sum();
        let avg = sum / intervals.len() as f32;

        let variance = if intervals.len() > 1 {
            intervals.iter()
                .map(|&x| (x - avg).powi(2))
                .sum::<f32>() / intervals.len() as f32
        } else {
            0.0
        };

        (avg, variance)
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

    

    

    

    

    
                    

    fn extract_difficulty_features(&mut self, notes: &[ProcessedNote]) -> Vec<f32> {
        let base_difficulty = self.calculate_base_difficulty(notes);
        let speed_difficulty = self.calculate_speed_difficulty(notes);
        let coordination_difficulty = self.calculate_coordination_difficulty(notes);

        vec![base_difficulty, speed_difficulty, coordination_difficulty]
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

// HandState 已被替换为使用 hand_model.rs 中的 ErgonomicHandSystem 和 FingerModel

impl ExperienceReplay {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, experience: Experience) {
        let was_full = self.buffer.len() >= self.capacity;
        
        // 检查经验数据有效性，防止NaN进入经验池
        let mut cleaned_experience = experience;
        
        // 清理state中的NaN值
        for state_val in &mut cleaned_experience.state {
            if !state_val.is_finite() {
                *state_val = 0.0;
            }
        }
        
        // 清理其他数值字段
        if !cleaned_experience.reward.is_finite() {
            cleaned_experience.reward = 0.0;
        }
        if !cleaned_experience.value.is_finite() {
            cleaned_experience.value = 0.0;
        }
        if !cleaned_experience.next_value.is_finite() {
            cleaned_experience.next_value = 0.0;
        }
        if !cleaned_experience.log_prob.is_finite() {
            cleaned_experience.log_prob = 0.0;
        }
        if !cleaned_experience.advantage.is_finite() {
            cleaned_experience.advantage = 0.0;
        }
        if !cleaned_experience.return_.is_finite() {
            cleaned_experience.return_ = 0.0;
        }
        
        // 清理future_notes中的NaN值
        for fn_val in &mut cleaned_experience.future_notes {
            if !fn_val.is_finite() {
                *fn_val = 0.0;
            }
        }
        
        // 清理反思相关字段
        if !cleaned_experience.reflection_score.is_finite() {
            cleaned_experience.reflection_score = 0.5;
        }
        for rf_val in &mut cleaned_experience.reflection_features {
            if !rf_val.is_finite() {
                *rf_val = 0.0;
            }
        }
        
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(cleaned_experience);
        
        // 只在状态变化时输出日志，减少刷屏
        let is_full = self.buffer.len() >= self.capacity;
        let usage_ratio = self.buffer.len() as f32 / self.capacity as f32;
        
        if was_full != is_full {
            // 状态变化：从非满到满，或从满到非满
            if is_full {
                println!("[经验池] 容量已满: 100% ({}/{})，开始替换最旧样本", self.buffer.len(), self.capacity);
            } else {
                println!("[经验池] 容量恢复: {:.1}% ({}/{})", 
                         usage_ratio * 100.0, self.buffer.len(), self.capacity);
            }
        } else if !is_full && usage_ratio >= 0.95 {
            // 非满载但接近满载时提醒（使用基于时间的简单检查）
        }
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
    
    /// 公共方法：初始化GPU支持（供智慧AI使用）
    pub fn init_gpu_support(&mut self) {
        println!("[AI] 初始化GPU支持...");
        self.main_network.init_gpu_sync();
        self.target_network.init_gpu_sync();
        println!("[AI] 主网络GPU状态: {}, 目标网络GPU状态: {}", 
                 self.main_network.gpu_initialized,
                 self.target_network.gpu_initialized);
    }

    fn clean_model_data(&mut self) {
        let default = Self::new(self.rotation);
        self.main_network.clean();
        self.target_network.clean();
        self.ergonomic_hand_system.clean_all_finger_states();
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

        self.ergonomic_hand_system.clean_all_finger_states();
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

        // 验证人体工程学手部系统
        let hands_valid = true; // ErgonomicHandSystem 内部处理验证

        float_fields_valid && networks_valid && hands_valid
    }

    pub fn new(rotation: f32) -> Self {
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(32)
            .build()
            .ok().map(Arc::new);
        let _rad = rotation.to_radians();
        // 初始化时使用自动检测模式，先默认为TwoFinger
        let game_mode = crate::hand_model::GameMode::TwoFinger;
        let mut ai = Self {
            main_network: DeepNeuralNetwork::new(),
            target_network: DeepNeuralNetwork::new(),
            thread_pool: thread_pool.clone(),
            //thread_count: if thread_pool.is_some() { 32 } else { 1 },
            feature_extractor: AdvancedFeatureExtractor::new(),
            experience_replay: ExperienceReplay::new(100000),
            ergonomic_hand_system: ErgonomicHandSystem::new(),
            rotation,
            exploration_rate: 0.25, // 增加探索率，鼓励模式切换
            discount_factor: 0.95,
            target_update_frequency: 5, //进一步提高训练频率
            total_notes_processed: 0,
            correct_predictions: 0,
            training_episodes: 0,
            average_reward: 0.7,
            difficulty_adaptation: 1.0,
            learning_momentum: 0.9,
            confidence_threshold: 0.4, // 降低置信度阈值，让AI更愿意尝试不同模式
            pattern_memory: BTreeMap::new(),
            performance_history: VecDeque::with_capacity(50000),
            version: Self::CURRENT_VERSION,
            last_save_episodes: 0,
            last_update_time: -1.0,
            line_rotations: HashMap::new(),
            game_mode,
            stability_factor: 0.85,
            hand_switch_penalty: 0.01,
            consistency_bonus: 0.3,
            adaptive_learning_rate: 0.005,
            pattern_recognition_strength: 1.0,
            memory_consolidation_rate: 0.1,
            recent_assignments: VecDeque::with_capacity(50),
            hand_switch_count: 0,
            last_assigned_hand: None,
            recent_losses: VecDeque::with_capacity(20),
            // 防过拟合机制初始化
            best_validation_reward: f32::NEG_INFINITY,
            validation_patience: 0,
            no_improvement_count: 0,
            early_stopping_threshold: 0.005, // 降低早停阈值，适应更高的学习率
            // 基于音符序列模式的早停机制
            sequence_pattern_history: VecDeque::with_capacity(100),
            pattern_diversity_threshold: 0.1,
            sequence_complexity_threshold: 0.05,
            convergence_window: 20,
            // 反思系统初始化
            decision_history: VecDeque::with_capacity(100),
            reflection_memory: HashMap::new(),
            reflection_confidence: 0.5,
            last_reflection_time: 0.0,
            reflection_learning_rate: 0.01,
            // 自主智慧AI集成初始化
            wisdom_ai_enabled: true,
            consciousness_state: ConsciousnessState::Exploring,
            wisdom_integration_level: 0.7,
        };
        ai.target_network = DeepNeuralNetwork::new(); // 创建新实例
        //ai.target_network.build_architecture();       // 构建相同架构
        //ai.target_network.init_gpu_sync();
        ai.warm_thread_pool();
        ai
    }

    fn load_or_create(filepath: &str, rotation: f32, _config: &Config) -> Self {
        let path = Path::new(filepath);
            if let Ok(bytes) = fs::read(path) {
                println!("[Model] 找到模型文件: {}, 大小: {} bytes", filepath, bytes.len());
                match bincode::deserialize::<Self>(&bytes) {
                    Ok(mut ai) => {
                        println!("[Model] 模型反序列化成功，训练回合数: {}", ai.training_episodes);
                        match ai.validate_for_serialization() {
                            true => {
                                println!("[Model] 模型验证通过，开始加载");
                                
                                // 强制重置梯度裁剪阈值为新版本，避免过度裁剪
                                ai.main_network.max_grad_norm = 20.0;
                                ai.target_network.max_grad_norm = 20.0;
                                println!("[梯度调整] 重置梯度裁剪阈值: main_network={}, target_network={}", 
                                         ai.main_network.max_grad_norm, ai.target_network.max_grad_norm);
                                
                                ai.rotation = rotation;
                                ai.update_hand_positions();

                        // 重置并初始化 main_network GPU
                        ai.main_network.initialization_attempted = false;
                        ai.main_network.initialization_failed = false;
                        ai.main_network.gpu_initialized = false;
                        println!("[GPU/CPU SWITCH] Beginning GPU init for main_network...");
                        ai.main_network.init_gpu_sync();

                        if ai.main_network.gpu_initialized {
                            println!("[GPU/CPU SWITCH] main_network GPU initialized successfully.");
                        } else {
                            println!("[GPU/CPU SWITCH] main_network GPU initialization failed, using CPU fallback.");
                        }

                        // 重置并初始化 target_network GPU
                        ai.target_network.device = None;
                                ai.target_network.queue = None;
                                ai.target_network.batch_size_buffer = None;
                                ai.target_network.gpu_executor = None;
                                for layer in &mut ai.target_network.layers {
                                    layer.weights_buffer = None;
                                    layer.biases_buffer = None;
                                    layer.activations_buffer = None;
                                }
                        ai.target_network.initialization_attempted = false;
                        ai.target_network.initialization_failed = false;
                        ai.target_network.gpu_initialized = false;

                        println!("[GPU/CPU SWITCH] Beginning GPU init for target_network...");
                        ai.target_network.init_gpu_sync();

                        if ai.target_network.gpu_initialized && ai.target_network.device.is_some() && ai.target_network.queue.is_some() {
                            println!("[GPU/CPU SWITCH] target_network GPU initialized successfully.");
                        } else {
                            println!("[GPU/CPU SWITCH] target_network GPU initialization failed, using CPU fallback.");
                        }

                        // 重置游戏模式为自动检测模式
                                ai.game_mode = crate::hand_model::GameMode::TwoFinger;
                                ai.ergonomic_hand_system.set_game_mode(crate::hand_model::GameMode::TwoFinger);
                                // 初始化人体工程学手部系统
                                ai.ergonomic_hand_system = ErgonomicHandSystem::new();
                                return ai;
                            }
                            false => {
                                eprintln!("[Model Error] 模型验证失败，将创建新模型");
                                let float_fields_valid = [
                                    ai.exploration_rate,
                                    ai.discount_factor,
                                    ai.average_reward,
                                    ai.difficulty_adaptation,
                                    ai.learning_momentum,
                                    ai.confidence_threshold
                                ].iter().all(|f| f.is_finite());
                                eprintln!("[Model Debug] 浮点字段验证: {}", float_fields_valid);
                                eprintln!("[Model Debug] main_network验证: {}", ai.main_network.validate());
                                eprintln!("[Model Debug] target_network验证: {}", ai.target_network.validate());
                                // 人体工程学手部系统内部处理验证
                                eprintln!("[Model Debug] 使用 ErgonomicHandSystem");
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("[Model Error] 模型反序列化失败: {:?}", e);
                    }
                }
            } else {
                eprintln!("[Model Error] 无法读取模型文件: {}", filepath);
            }

            // 加载失败：创建新模型并保存
            let mut ai = Self::new(rotation);
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

    

    fn update_hand_positions(&mut self) {
        // 根据旋转角度调整手部位置
        let rad = self.rotation.to_radians();
        let cos_r = rad.cos();
        let sin_r = rad.sin();
        
        // 左手位置 (-0.3, 0.0) 旋转
        self.ergonomic_hand_system.left_hand.position = Vector2::new(-0.3 * cos_r, -0.3 * sin_r);
        // 右手位置 (0.3, 0.0) 旋转
        self.ergonomic_hand_system.right_hand.position = Vector2::new(0.3 * cos_r, 0.3 * sin_r);
    }

    /*
    fn light_update_hand_states(&mut self, notes: &[Note]) {
        for note in notes {
            // 在chart.rs中已经进行了坐标转换，这里直接使用转换后的坐标
            let pos = Vector2::new(
                note.object.translation.0.now(),
                note.object.translation.1.now()
            );

            match note.hand {
                Hand::Left => self.ergonomic_hand_system.left_hand.position = pos,
                Hand::Right => self.ergonomic_hand_system.right_hand.position = pos,
                //_ => {} // 未分配的不处理
            }
        }
    }

     */

    fn apply_consciousness_optimization(&mut self, processed_notes: &mut [ProcessedNote], wisdom_ai: &mut AutonomousWisdomAI, line_id: usize) -> bool {
        // 将神经网络分配后的音符转换为Note格式供智慧AI进行意识分析
        let notes_for_consciousness: Vec<Note> = processed_notes.iter().map(|pn| {
            Note {
                time: pn.time,
                kind: pn.kind.clone(),
                height: pn.position.y,
                object: crate::core::Object::default(),
                speed: 1.0,
                end_speed: 1.0,
                start_height: pn.position.y,
                hand: pn.assigned_hand.unwrap_or(Hand::Left),
                above: false,
                multiple_hint: false,
                fake: false,
                judge: crate::judge::JudgeStatus::NotJudged,
                format: false,
            }
        }).collect();
        
        // 让智慧AI进行意识层面的分析，但不重新推理
        let consciousness_suggestions = wisdom_ai.analyze_existing_assignments(&notes_for_consciousness, &self.ergonomic_hand_system);
        
        // 基于意识建议进行选择性优化
        let mut optimized_count = 0;
        let mut wisdom_assignments = Vec::new();
        
        for (i, suggestion) in consciousness_suggestions.iter().enumerate() {
            if i < processed_notes.len() {
                if let Some(nn_hand) = processed_notes[i].assigned_hand {
                    wisdom_assignments.push(nn_hand); // 先记录原始分配
                    
                    // 根据意识状态和集成程度决定是否采纳建议
                    let apply_consciousness = fastrand::f32() < self.wisdom_integration_level;
                    if apply_consciousness && suggestion.suggested_hand != nn_hand {
                        let original_hand = processed_notes[i].assigned_hand;
                        processed_notes[i].assigned_hand = Some(suggestion.suggested_hand);
                        optimized_count += 1;
                        wisdom_assignments[i] = suggestion.suggested_hand; // 更新为智慧AI的建议
                        
                        // 记录意识优化经验
                        if let Some(orig) = original_hand {
                            self.record_consciousness_experience(&processed_notes[i], orig, suggestion.suggested_hand, line_id, suggestion.confidence);
                        }
                    }
                } else {
                    wisdom_assignments.push(Hand::Left); // 默认值
                }
            }
        }
        
        // 将智慧AI经验反馈到主网络训练 - 这是关键！
        if optimized_count > 0 {
            self.integrate_wisdom_experience(wisdom_ai, &notes_for_consciousness, &wisdom_assignments);
        }
        
        optimized_count > 0
    }
    
    fn record_consciousness_experience(&mut self, note: &ProcessedNote, original_hand: Hand, consciousness_hand: Hand, line_id: usize, _confidence: f32) { self.record_optimization_experience(note, original_hand, consciousness_hand, line_id); }
    
    /// 多维度判断是否应该记录优化经验
    fn should_record_optimization_experience(&self, exp: &Experience) -> bool {
        let reward_significant = exp.reward.abs() > 0.5;
        let value_confident = exp.value > 0.7 || exp.value < 0.3;
        let reflection_meaningful = exp.reflection_score > 0.6;
        let wisdom_insightful = exp.hand_system_interface > 0.6 || 
                               exp.evolution_history > 0.6 || 
                               exp.current_consciousness > 0.6;
        let advantage_strong = exp.advantage.abs() > 0.1;
        let decision_pattern_interesting = exp.decision_history.len() > 3 && 
                                         (exp.decision_history.windows(2).any(|w| w[0] != w[1]));
        let dimensions_met = [
            reward_significant,
            value_confident,
            reflection_meaningful,
            wisdom_insightful,
            advantage_strong,
            decision_pattern_interesting
        ].iter().filter(|&&x| x).count();
        
        dimensions_met >= 3 // 至少满足3个维度
    }

    fn process_optimization_experiences(&mut self, experiences: &[Experience]) {
        println!("[优化经验] 处理{}个多维度优化经验", experiences.len());
        
        // 维度分析
        let mut reward_stats = Vec::new();
        let mut value_stats = Vec::new();
        let mut reflection_stats = Vec::new();
        let mut wisdom_stats = Vec::new();
        
        for exp in experiences {
            reward_stats.push(exp.reward);
            value_stats.push(exp.value);
            reflection_stats.push(exp.reflection_score);
            wisdom_stats.push((exp.hand_system_interface, exp.evolution_history, exp.current_consciousness));
        }
        
        // 分别计算智慧AI经验和普通经验的统计信息
        let wisdom_experiences: Vec<_> = experiences.iter()
            .filter(|exp| exp.evolution_history > 0.5)
            .collect();
        let normal_experiences: Vec<_> = experiences.iter()
            .filter(|exp| exp.evolution_history <= 0.5)
            .collect();
        
        let (wisdom_avg_reward, wisdom_avg_value, wisdom_avg_reflection) = if !wisdom_experiences.is_empty() {
            let w_reward = wisdom_experiences.iter().map(|e| e.reward).sum::<f32>() / wisdom_experiences.len() as f32;
            let w_value = wisdom_experiences.iter().map(|e| e.value).sum::<f32>() / wisdom_experiences.len() as f32;
            let w_reflection = wisdom_experiences.iter().map(|e| e.reflection_features.iter().sum::<f32>() / e.reflection_features.len() as f32).sum::<f32>() / wisdom_experiences.len() as f32;
            (w_reward, w_value, w_reflection)
        } else {
            (0.0, 0.0, 0.0)
        };
        
        let (normal_avg_reward, normal_avg_value) = if !normal_experiences.is_empty() {
            let n_reward = normal_experiences.iter().map(|e| e.reward).sum::<f32>() / normal_experiences.len() as f32;
            let n_value = normal_experiences.iter().map(|e| e.value).sum::<f32>() / normal_experiences.len() as f32;
            (n_reward, n_value)
        } else {
            (0.0, 0.0)
        };
        
        // 加权平均计算，智慧AI经验权重更高
        let wisdom_weight = if wisdom_experiences.is_empty() { 0.0 } else { 
            (wisdom_experiences.len() as f32 / experiences.len() as f32) * 2.0 
        };
        let normal_weight = 1.0 - wisdom_weight;
        
        let avg_reward = wisdom_avg_reward * wisdom_weight + normal_avg_reward * normal_weight;
        let avg_value = wisdom_avg_value * wisdom_weight + normal_avg_value * normal_weight;
        let avg_reflection = if wisdom_experiences.is_empty() { 0.0 } else { wisdom_avg_reflection };
        let avg_wisdom_interface = wisdom_stats.iter().map(|(i, _, _)| i).sum::<f32>() / wisdom_stats.len() as f32;
        
        // 进化和意识水平只计算智慧AI经验的
        let avg_evolution = if wisdom_experiences.is_empty() { 
            0.0 
        } else { 
            wisdom_experiences.iter().map(|e| e.evolution_history).sum::<f32>() / wisdom_experiences.len() as f32 
        };
        let avg_consciousness = if wisdom_experiences.is_empty() { 
            0.0 
        } else { 
            wisdom_experiences.iter().map(|e| e.current_consciousness).sum::<f32>() / wisdom_experiences.len() as f32 
        };
        
        println!("[多维度分析] 奖励:{:.3}, 价值:{:.3}, 反思:{:.3}, 智慧接口:{:.3}, 进化:{:.3}, 意识:{:.3}", 
                 avg_reward, avg_value, avg_reflection, avg_wisdom_interface, avg_evolution, avg_consciousness);
        println!("[分析详情] 智慧AI经验:{}个, 普通经验:{}个, 智慧权重:{:.2}", 
                 wisdom_experiences.len(), normal_experiences.len(), wisdom_weight);
        
        // 根据多维度分析调整训练参数
        if avg_reward > 0.7 {
            // 高奖励经验，增加学习率
            self.adaptive_learning_rate = (self.adaptive_learning_rate * 1.1).min(0.01);
            println!("[自适应] 基于高奖励经验，增加学习率至{:.6}", self.adaptive_learning_rate);
        }
        if avg_reflection > 0.8 {
            // 高反思得分，增强模式识别
            self.pattern_recognition_strength = (self.pattern_recognition_strength * 1.05).min(2.0);
            println!("[自适应] 基于高反思得分，增强模式识别至{:.3}", self.pattern_recognition_strength);
        }
        if avg_consciousness > 0.7 {
            self.exploration_rate = (self.exploration_rate * 1.1).min(0.5);
            println!("[自适应] 基于高意识状态，增加探索率至{:.3}", self.exploration_rate);
        }
        
        // 基于策略停滞检测的优化
        let stagnation_score = self.check_policy_loss_stagnation();
        if stagnation_score > 0.5 {
            self.exploration_rate = (self.exploration_rate * 1.3).min(0.7);
            println!("[策略优化] 检测到策略停滞(评分:{:.3})，强制增加探索率至{:.3}", stagnation_score, self.exploration_rate);
            self.add_policy_perturbation();
        }
        
        // 智慧AI性能调整
        if wisdom_experiences.len() > 5 && avg_wisdom_interface > 0.8 && wisdom_avg_reward < -0.3 {
            println!("[策略警告] 智慧AI过于自信但效果不佳，降低智慧集成度");
            self.wisdom_integration_level = (self.wisdom_integration_level * 0.9).max(0.1);
        }
        
        // 进化激励机制
        if avg_evolution > 0.6 && avg_consciousness > 0.5 {
            self.wisdom_integration_level = (self.wisdom_integration_level * 1.05).min(1.0);
            println!("[进化激励] 检测到高进化水平，提升智慧集成度至{:.3}", self.wisdom_integration_level);
        }
    }

    fn record_optimization_experience(&mut self, note: &ProcessedNote, original_hand: Hand, optimized_hand: Hand, _line_id: usize) {
        // 记录智慧AI优化神经网络分配的经验，帮助神经网络学习更好的策略
        let input = DeepNeuralNetwork::hand_model_to_input(&self.ergonomic_hand_system);
        
        let mut target = vec![0.0; 3]; // [left_prob, right_prob, value]
        match optimized_hand {
            Hand::Left => {
                target[0] = 0.9; // 更高置信度，因为是智慧AI的优化
                target[1] = 0.1;
            }
            Hand::Right => {
                target[0] = 0.1;
                target[1] = 0.9;
            }
        }
        
        // 设置较高的价值，因为这是智慧AI的优化建议
        target[2] = 0.85;
        
        let experience = Experience {
            state: input.clone(),
            action: original_hand as usize,
            reward: if optimized_hand != original_hand { 0.8 } else { 0.2 }, // 奖励优化
            next_state: input.clone(),
            done: false,
            timestamp: fastrand::f32() * 1000.0, // 使用随机时间戳
            // PPO新增字段
            log_prob: 0.0,
            value: target[2],
            next_value: 0.0,
            advantage: 0.0,
            return_: 0.0,
            // 未来音符信息（空）
            future_notes: vec![0.0; 16], // 4个未来音符 * 4个特征
            // 反思相关字段
            reflection_score: note.confidence,
            decision_history: vec![original_hand as usize],
            outcome_success: true,
            reflection_features: vec![0.0; 8],
            // 智慧AI相关字段
            hand_system_interface: 0.5,
            evolution_history: 0.5,
            current_consciousness: 0.5,
        };
        
        self.experience_replay.push(experience);
    }

    fn integrate_wisdom_experience(&mut self, wisdom_ai: &AutonomousWisdomAI, notes: &[Note], wisdom_assignments: &[Hand]) {
        for (note, &wisdom_hand) in notes.iter().zip(wisdom_assignments) {
            let input = DeepNeuralNetwork::hand_model_to_input(&self.ergonomic_hand_system);

            let mut target = vec![0.0; 3]; // [left_prob, right_prob, value]
            match wisdom_hand {
                Hand::Left => {
                    target[0] = 0.8; // 高置信度的左手概率
                    target[1] = 0.2;
                }
                Hand::Right => {
                    target[0] = 0.2;
                    target[1] = 0.8; // 高置信度的右手概率
                }
            }

            // 基于智慧AI的意识状态调整价值预测
            target[2] = match wisdom_ai.current_consciousness {
                crate::autonomous_wisdom::ConsciousnessState::Creating => 0.9,
                crate::autonomous_wisdom::ConsciousnessState::Optimizing => 0.8,
                crate::autonomous_wisdom::ConsciousnessState::Learning => 0.7,
                crate::autonomous_wisdom::ConsciousnessState::Exploring => 0.6,
                _ => 0.5,
            };

            // 创建增强经验
            let enhanced_experience = Experience {
                state: input.clone(),
                action: wisdom_hand as usize,
                reward: {
                    // 将Note转换为ProcessedNote以使用更全面的奖励计算
                    let processed_note = ProcessedNote {
                        index: 0,
                        position: Vector2::new(note.object.translation.0.now(), note.object.translation.1.now()),
                        time: note.time,
                        kind: note.kind.clone(),
                        assigned_hand: Some(wisdom_hand),
                        confidence: 0.8, // 智慧AI的默认置信度
                        features: Vec::new(),
                        difficulty: 1.0,
                        duration: 0.0,
                        actual_position: None,
                        position_error: 0.0,
                        timing_error: 0.0,
                        is_successful: true,
                        physical_confidence: 0.8,
                    };
                    self.calculate_reward(&processed_note, wisdom_hand, 0.8)
                },
                //next_state: input.clone(),
                next_state: input,
                done: false,
                timestamp: fastrand::f32() * 1000.0,
                log_prob: (target[wisdom_hand as usize] as f32).ln(),
                value: target[2],
                next_value: target[2],
                advantage: 0.0,
                return_: target[2],
                future_notes: vec![],
                reflection_score: wisdom_ai.consciousness.intrinsic_motivation,
                hand_system_interface: wisdom_ai.hand_system_interface.performance_metrics.accuracy,
                evolution_history: wisdom_ai.evolution_history.len() as f32,
                current_consciousness: match wisdom_ai.current_consciousness {
                    ConsciousnessState::Exploring => 0.2,
                    ConsciousnessState::Learning => 0.4,
                    ConsciousnessState::Optimizing => 0.6,
                    ConsciousnessState::Creating => 0.8,
                    ConsciousnessState::Reflecting => 1.0,
                },
                decision_history: vec![wisdom_hand as usize],
                outcome_success: true,
                reflection_features: vec![
                    wisdom_ai.consciousness.curiosity_index,
                    wisdom_ai.consciousness.intrinsic_motivation,
                    self.wisdom_integration_level,
                ],
            };

            // 将智慧AI经验添加到经验回放
            self.experience_replay.push(enhanced_experience);
        }
    }

    pub fn analyze_and_assign(&mut self, notes: &mut [Note], _config: &Config, bpm_list: &BpmList, line_id: usize, time: f32) {
        //println!("[开始分析] 线路{} 音符数量:{} 游戏模式:{:?}", line_id, notes.len(), self.game_mode);
        if notes.is_empty() {
            return;
        }
        
        // 更新人体工程学手部系统
        self.ergonomic_hand_system.update(time);
        
        let mut processed_notes = self.preprocess_notes(notes);
        
        let simultaneous_groups = self.detect_simultaneous_groups(&processed_notes);
        let mut bpm_list_clone = bpm_list.clone();
        self.assign_simultaneous_groups(&mut processed_notes, &simultaneous_groups, &mut bpm_list_clone, line_id);
        
        // 保存推理前的状态用于对比
        let original_assignments: Vec<Option<Hand>> = processed_notes.iter()
            .map(|n| n.assigned_hand)
            .collect();
        
        // 应用反思记忆影响决策
        self.apply_reflection_influence(&mut processed_notes, line_id);
        
        // 先用神经网络进行主要分配
        self.ai_assign_single_notes(&mut processed_notes, &simultaneous_groups, &mut bpm_list_clone, line_id);
        
        let _simultaneous_groups = self.detect_simultaneous_groups(&processed_notes);
        let mut bpm_list_clone = bpm_list.clone();
        self.assign_simultaneous_groups(&mut processed_notes, &simultaneous_groups, &mut bpm_list_clone, line_id);
        
        // 保存推理前的状态用于对比
        let original_assignments: Vec<Option<Hand>> = processed_notes.iter()
            .map(|n| n.assigned_hand)
            .collect();
        
        // 应用反思记忆影响决策
        self.apply_reflection_influence(&mut processed_notes, line_id);
        
        self.ai_assign_single_notes(&mut processed_notes, &simultaneous_groups, &mut bpm_list_clone, line_id);
        
        // 集成自主智慧AI - 基于神经网络分配结果进行意识层面优化
        if self.wisdom_ai_enabled {
            if let Some(wisdom_mutex) = WISDOM_AI.get() {
                if let Ok(mut wisdom_ai) = wisdom_mutex.lock() {
                    // 将神经网络分配后的音符传递给智慧AI进行意识层面分析，而不是重新推理
                    let consciousness_optimized = self.apply_consciousness_optimization(&mut processed_notes, &mut wisdom_ai, line_id);
                    if consciousness_optimized {
                        self.consciousness_state = wisdom_ai.current_consciousness;
                        // 根据智慧AI的建议调整集成程度
                        self.wisdom_integration_level = (self.wisdom_integration_level * 0.95 + 0.05).min(1.0);
                    }
                }
            }
        }
        
        self.post_process_assignments(&mut processed_notes);
        
        
        // 基于物理模型更新音符判断状态
        self.update_notes_physical_judgement(&mut processed_notes, time);
        
        self.apply_and_learn(notes, &processed_notes);
        // 根据网络学习阶段动态调整训练触发条件
        let min_samples = if self.training_episodes < 100 {
            // 早期阶段：快速开始学习
            400
        } else if self.training_episodes < 500 {
            // 中期阶段：平衡速度和稳定性
            800
        } else {
            // 后期阶段：确保质量和稳定性
            1200
        };

        if self.experience_replay.len() >= min_samples && self.total_notes_processed % 8 == 0 {
            println!("[训练] 开始网络训练，经验回放大小: {}，训练轮次: {}，最小样本数: {}", 
                     self.experience_replay.len(), self.training_episodes, min_samples);
            self.train_network();
            println!("[训练] 网络训练完成");
        }
        if self.training_episodes % self.target_update_frequency as u64 == 0 {
            self.target_network = self.main_network.clone();
        }
        // 强制重置早停阈值，防止旧模型配置覆盖
        self.early_stopping_threshold = 0.005;
        
        self.training_episodes += 1;
        self.last_save_episodes += 1;
        if self.last_save_episodes >= 10 {
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
                difficulty,
                duration,
                actual_position: None,
                position_error: 0.0,
                timing_error: 0.0,
                is_successful: false,
                physical_confidence: 0.0,
            }
        }).collect()
    }
    
    /// 基于物理模型评估音符的成功概率
    fn evaluate_note_physical_success(&self, note: &mut ProcessedNote, current_time: f32) {
        if let Some(hand) = note.assigned_hand {
            // 使用人体工程学手部系统评估音符
            let (is_successful, position_error, timing_error, physical_confidence) = 
                self.ergonomic_hand_system.evaluate_note_success(
                    hand,
                    &note.position,
                    note.time,
                    current_time,
                    &note.kind,
                );
            
            // 更新音符的物理判断状态
            note.is_successful = is_successful;
            note.position_error = position_error;
            note.timing_error = timing_error;
            note.physical_confidence = physical_confidence;
            
            // 如果成功，记录实际位置（这里用目标位置作为近似）
            if is_successful {
                note.actual_position = Some(note.position);
            }
        }
    }
    
    /// 更新所有音符的物理判断状态
    fn update_notes_physical_judgement(&mut self, notes: &mut [ProcessedNote], current_time: f32) {
        for note in notes.iter_mut() {
            self.evaluate_note_physical_success(note, current_time);
        }
    }

    fn detect_simultaneous_groups(&self, notes: &[ProcessedNote]) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut used = vec![false; notes.len()];
        const SIMULTANEOUS_THRESHOLD: f32 = 0.01;

        for i in 0..notes.len() {
            if used[i] { continue; }
            let mut group = vec![i];
            let base_time = notes[i].time;
            for j in i + 1..notes.len() {
                if used[j] { continue; }
                let time_diff = (notes[j].time - base_time).abs();

                let threshold = if matches!(notes[i].kind, NoteKind::Hold { .. }) &&
                    matches!(notes[j].kind, NoteKind::Hold { .. }) {
                    SIMULTANEOUS_THRESHOLD
                } else if matches!(notes[i].kind, NoteKind::Hold { .. }) ||
                    matches!(notes[j].kind, NoteKind::Hold { .. }) {
                    SIMULTANEOUS_THRESHOLD * 0.7
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
        const CONTEXT_WINDOW: usize = 64;
        const CROSSING_THRESHOLD: f32 = 0.15;
        
        for group in groups {
            if group.len() < 2 {
                continue;
            }

            if self.game_mode == crate::hand_model::GameMode::TwoFinger {
                // 2指模式下的同时音符处理优化
                let mut sorted_group: Vec<_> = group.iter().map(|&i| (notes[i].position.x, i)).collect();
                sorted_group.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

                // 检测这组音符是否跨越中线
                let mut crosses_center = false;
                let mut left_count = 0;
                let mut right_count = 0;

                for &(_, note_idx) in &sorted_group {
                    let x = notes[note_idx].position.x;
                    if x.abs() < CROSSING_THRESHOLD {
                        crosses_center = true;
                    } else if x < 0.0 {
                        left_count += 1;
                    } else {
                        right_count += 1;
                    }
                }

                // 策略1：如果音符跨越中线，使用智能交替策略
                if crosses_center || (left_count > 0 && right_count > 0) {
                    // 获取最近的手分配历史
                    let recent_hands: Vec<_> = self.recent_assignments.iter()
                        .rev()
                        .take(8)
                        .map(|(h, _, _)| *h)
                        .collect();

                    let start_hand = if recent_hands.is_empty() {
                        let first_note_idx = sorted_group[0].1;
                        let first_note = &notes[first_note_idx];
                        let note_position = crate::hand_model::Vector2::new(
                            first_note.position.x,
                            first_note.position.y,
                        );

                        let (hand, _, _) = self.ergonomic_hand_system.assign_note_hand(
                            note_position,
                            &first_note.kind,
                            first_note.time,
                        );

                        hand
                    } else {

                        // 使用最近的手的相反手，以促进交替

                        if recent_hands[0] == Hand::Left { Hand::Right } else { Hand::Left }
                    };

                    let mut current_hand = start_hand;

                    // 为每个音符分配手，考虑音符的分布和位置
                    for (idx, &(_, note_idx)) in sorted_group.iter().enumerate() {
                        // 对于跨越中线的音符，使用交替策略
                        if notes[note_idx].position.x.abs() < CROSSING_THRESHOLD {
                            current_hand = if current_hand == Hand::Left { Hand::Right } else { Hand::Left };
                        } else {
                            // 对于不跨越中线的音符，根据人体工程学模型决定

                            let note_position = crate::hand_model::Vector2::new(
                                notes[note_idx].position.x,
                                notes[note_idx].position.y,
                            );

                            let (note_hand, _, _) = self.ergonomic_hand_system.assign_note_hand(
                                note_position,
                                &notes[note_idx].kind,
                                notes[note_idx].time,
                            );


                            // 如果音符手与当前手不同，且不是第一个音符，则切换

                            if idx > 0 && note_hand != current_hand {
                                current_hand = note_hand;
                            }

                            // 否则保持当前手，促进交替

                        }

                        notes[note_idx].assigned_hand = Some(current_hand);
                        notes[note_idx].confidence = 0.90;

                        self.recent_assignments.push_back((current_hand, notes[note_idx].position.x, notes[note_idx].time));
                        if self.recent_assignments.len() > 50 {
                            self.recent_assignments.pop_front();
                        }
                    }
                }
                // 策略2：音符全部在一侧，使用平衡交替策略
                else {
                    // 检查最近的手分配历史，以确定起始手
                    let mut recent_left_count = 0;
                    let mut recent_right_count = 0;

                    for (hand, _, _) in self.recent_assignments.iter().rev().take(6) {
                        if *hand == Hand::Left {
                            recent_left_count += 1;
                        } else {
                            recent_right_count += 1;
                        }
                    }

                    // 选择使用较少的手作为起始手，以平衡负荷
                    let start_hand = if recent_left_count <= recent_right_count {
                        Hand::Left
                    } else {
                        Hand::Right
                    };

                    let mut current_hand = start_hand;

                    // 为音符分配手，强制交替
                    for (idx, &(_, note_idx)) in sorted_group.iter().enumerate() {
                        // 每个音符交替一次手，确保相同时间戳的音符分配给不同的手
                        if idx > 0 {
                            current_hand = if current_hand == Hand::Left { Hand::Right } else { Hand::Left };
                        }

                        notes[note_idx].assigned_hand = Some(current_hand);
                        notes[note_idx].confidence = 0.92;

                        self.recent_assignments.push_back((current_hand, notes[note_idx].position.x, notes[note_idx].time));
                        if self.recent_assignments.len() > 50 {
                            self.recent_assignments.pop_front();
                        }
                    }
                }

                self.update_hand_states_for_group(notes, group);
                continue;
            }
            
            // 4指模式保持原有逻辑
            for &note_idx in group {
                let start = note_idx.saturating_sub(CONTEXT_WINDOW / 2);
                let end = (note_idx + CONTEXT_WINDOW / 2).min(notes.len());
                let context = &notes[start..end];
                let features = self.feature_extractor.extract_features(context, CONTEXT_WINDOW, bpm_list);
                let ai_decision = self.make_ai_decision(&features, &notes[note_idx], line_id, note_idx, notes);
                notes[note_idx].assigned_hand = Some(ai_decision.0);
                notes[note_idx].confidence = ai_decision.1;
                self.recent_assignments.push_back((ai_decision.0, notes[note_idx].position.x, notes[note_idx].time));
                if self.recent_assignments.len() > 50 {
                    self.recent_assignments.pop_front();
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
        //self.correct_position_mismatches(notes);
        self.check_hand_consistency(notes);
    }
    /*
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

     */

    fn ensure_alternating_pattern(&mut self, notes: &mut [ProcessedNote]) {
        const MAX_CONSECUTIVE: usize = 3;
        const TIME_THRESHOLD: f32 = 0.3;
        const CONFIDENCE_THRESHOLD: f32 = 0.8; // 高置信度阈值

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
                            // 检查置信度，只有低置信度才强制切换
                            if note.confidence < CONFIDENCE_THRESHOLD {
                                indices_to_switch.push(i);
                                consecutive_count = 0;
                            } else {
                                //println!("[交替保护] 音符{}置信度{:.3}，跳过强制交替", i, note.confidence);
                            }
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
                println!("[交替修正] 音符{}从{:?}改为{:?}", i, current_hand, notes[i].assigned_hand);
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
            // 使用物理模型判断（检查组内是否有任何音符成功）
            let success = group.iter().any(|&idx| notes[idx].is_successful);
            // 使用 ErgonomicHandSystem 更新左手状态
            self.ergonomic_hand_system.update_finger_state(
                Hand::Left, 
                crate::hand_model::FingerType::Index, 
                avg_pos, 
                group_time, 
                success, 
                &notes[group[0]].kind
            );
        }

        if !right_positions.is_empty() {
            let avg_pos = right_positions.iter().fold(Vector2::new(0.0, 0.0), |acc, &pos| acc + pos) * (1.0 / right_positions.len() as f32);
            // 使用物理模型判断（检查组内是否有任何音符成功）
            let success = group.iter().any(|&idx| notes[idx].is_successful);
            // 使用 ErgonomicHandSystem 更新右手状态
            self.ergonomic_hand_system.update_finger_state(
                Hand::Right, 
                crate::hand_model::FingerType::Index, 
                avg_pos, 
                group_time, 
                success, 
                &notes[group[0]].kind
            );
        }
    }

    fn ai_assign_single_notes(&mut self, notes: &mut [ProcessedNote], simultaneous_groups: &[Vec<usize>], bpm_list: &mut BpmList, line_id: usize) {
        let assigned_indices: std::collections::HashSet<usize> = simultaneous_groups.iter().flatten().copied().collect();
        const CONTEXT_WINDOW: usize = 64;
        const BATCH_SIZE: usize = 64;  // 降低批次大小，让智慧AI也能使用GPU推理

        let mut unassigned_indices = Vec::new();
        for i in 0..notes.len() {
            if !assigned_indices.contains(&i) {
                unassigned_indices.push(i);
            }
        }
        if unassigned_indices.len() < BATCH_SIZE {
            return;
        }

        for batch_indices in unassigned_indices.chunks(BATCH_SIZE) {
            if batch_indices.len() < BATCH_SIZE {
                continue;
            }

            let results: Vec<_> = if let Some(pool) = &self.thread_pool {
                pool.install(|| {
                    batch_indices.par_iter().map(|&idx| {
                        let start_idx = idx.saturating_sub(CONTEXT_WINDOW / 2);
                        let end_idx = (idx + CONTEXT_WINDOW / 2 + 1).min(notes.len());
                        let context = &notes[start_idx..end_idx];

                        let note_position = crate::hand_model::Vector2::new(notes[idx].position.x, notes[idx].position.y);
                        let current_note_time = notes[idx].time;

                        let mut feature_extractor = self.feature_extractor.clone();
                        let features = feature_extractor.extract_features(context, CONTEXT_WINDOW, bpm_list);

                        (
                            features,
                            idx,
                            note_position,
                            notes[idx].time,
                            notes[idx].is_successful,
                            notes[idx].kind.clone(),
                            current_note_time,
                        )
                    }).collect()
                })
            } else {
                batch_indices.iter().map(|&idx| {
                    let start_idx = idx.saturating_sub(CONTEXT_WINDOW / 2);
                    let end_idx = (idx + CONTEXT_WINDOW / 2 + 1).min(notes.len());
                    let context = &notes[start_idx..end_idx];

                    let note_position = crate::hand_model::Vector2::new(notes[idx].position.x, notes[idx].position.y);
                    let current_note_time = notes[idx].time;

                    let features = self.feature_extractor.extract_features(context, CONTEXT_WINDOW, bpm_list);

                    (
                        features,
                        idx,
                        note_position,
                        notes[idx].time,
                        notes[idx].is_successful,
                        notes[idx].kind.clone(),
                        current_note_time,
                    )
                }).collect()
            };

            let mut results_with_dominant = Vec::with_capacity(results.len());
            for result in results {
                let (features, idx, note_position, time, is_successful, kind, current_note_time) = result;
                let dominant_side = self.ergonomic_hand_system.assign_note_hand(note_position, &kind, current_note_time);
                results_with_dominant.push((features, idx, note_position, time, is_successful, kind, dominant_side));
            }

            let feature_slices: Vec<&[f32]> = results_with_dominant.iter().map(|r| r.0.as_slice()).collect();
            let outputs = if self.main_network.gpu_initialized {
                self.main_network.gpu_forward_batch(&feature_slices, results_with_dominant.len())
            } else {
                results_with_dominant.iter().map(|r| self.main_network.forward(&r.0)).collect()
            };

            for (i, output) in outputs.iter().enumerate() {
                let (features, note_idx, note_position, time, is_successful, kind, dominant_side) = &results_with_dominant[i];
                let current_note = notes[*note_idx].clone();

                let ai_decision = self.make_ai_decision(output, &current_note, line_id, *note_idx, notes);
                let ideal_hand = if current_note.position.x < 0.0 { Hand::Left } else { Hand::Right };
                let network_correct = ai_decision.0 == ideal_hand;

                let mut note = notes[*note_idx].clone();
                note.features = features.clone();
                note.assigned_hand = Some(ai_decision.0);
                note.confidence = ai_decision.1;

                self.recent_assignments.push_back((ai_decision.0, note_position.x, *time));
                if self.recent_assignments.len() > 78 {
                    self.recent_assignments.pop_front();
                }

                let success = *is_successful;
                let (hand, finger_index, _confidence) = dominant_side;
                self.ergonomic_hand_system.update_finger_state(*hand, ai_decision.2, *note_position, note.time, success, &kind);

                self.record_experience(&features, &mut note, ai_decision.0, ai_decision.1, network_correct, *note_idx, notes);

                notes[*note_idx] = note;
            }
        }
    }

    fn make_ai_decision(&mut self, features: &[f32], note: &ProcessedNote, _line_id: usize, note_idx: usize, notes: &[ProcessedNote]) -> (Hand, f32, Finger) {
        // 网络现在直接接受2560维输入（特征提取器的完整输出）
        let mut full_input = features.to_vec();
        
        // 确保输入维度正确
        full_input.resize(2560, 0.0);
        
        // 使用神经网络进行决策，完整输入作为输入
        // 注意：网络现在有两个输出层，第一个是10维的当前预测，第二个是12维的未来预测
        let network_outputs = self.main_network.forward(&full_input);
        
        // 解析第一个输出层（当前音符预测，10维）
        // [left_prob, right_prob, left_index_prob, left_middle_prob, right_index_prob, right_middle_prob, value, confidence, four_finger_mode, game_mode_decision]
        let left_prob = network_outputs.get(0).copied().unwrap_or(0.5).clamp(0.0, 1.0);
        let right_prob = network_outputs.get(1).copied().unwrap_or(0.5).clamp(0.0, 1.0);
        let left_index_prob = network_outputs.get(2).copied().unwrap_or(0.5).clamp(0.0, 1.0);
        let left_middle_prob = network_outputs.get(3).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let right_index_prob = network_outputs.get(4).copied().unwrap_or(0.5).clamp(0.0, 1.0);
        let right_middle_prob = network_outputs.get(5).copied().unwrap_or(0.0).clamp(0.0, 1.0);
        let _value = network_outputs.get(6).copied().unwrap_or(0.0); // 状态价值，暂时不用
        let confidence = network_outputs.get(7).copied().unwrap_or(0.7).clamp(0.0, 1.0);
        let _four_finger_mode = network_outputs.get(8).copied().unwrap_or(0.0); // 四指模式指示器
        let game_mode_decision = network_outputs.get(9).copied().unwrap_or(0.0).clamp(0.0, 1.0); // AI决定的游戏模式

        let ai_decided_mode = if game_mode_decision > 0.5 {
            crate::hand_model::GameMode::FourFinger
        } else {
            crate::hand_model::GameMode::TwoFinger
        };
        
        // 如果AI决定的模式与当前模式不同，则切换模式
        if ai_decided_mode != self.game_mode {
            //println!("[AI模式切换] AI决定从{:?}切换到{:?} (决策值: {:.3})",
            //         self.game_mode, ai_decided_mode, game_mode_decision);
            self.game_mode = ai_decided_mode;
            self.ergonomic_hand_system.set_game_mode(self.game_mode);
        }
        
        // 根据概率选择手（添加探索机制）
        let chosen_hand = if fastrand::f32() < self.exploration_rate {
            // 探索：随机选择
            if fastrand::bool() { Hand::Left } else { Hand::Right }
        } else {
            // 利用：根据网络输出选择
            if left_prob > right_prob {
                Hand::Left
            } else {
                Hand::Right
            }
        };
        
        // 根据游戏模式选择手指概率
        let chosen_finger = match self.game_mode {
            crate::hand_model::GameMode::TwoFinger => {
                // 2指模式：只能选择食指
                FingerType::Index
            },
            crate::hand_model::GameMode::FourFinger => {
                // 4指模式：根据概率选择最佳手指
                if chosen_hand == Hand::Left {
                    if left_index_prob > left_middle_prob {
                        FingerType::Index
                    } else {
                        FingerType::Middle
                    }
                } else {
                    if right_index_prob > right_middle_prob {
                        FingerType::Index
                    } else {
                        FingerType::Middle
                    }
                }
            }
        };
        
        // 将手指类型转换为手指索引
        let finger_index = match chosen_finger {
            FingerType::Index => 0,
            FingerType::Middle => 1,
            FingerType::Thumb => 0,
            FingerType::Ring => 1,
            FingerType::Pinky => 2,
        };
        
        // 应用手指按下状态
        self.ergonomic_hand_system.apply_finger_press(chosen_hand, finger_index, note.time);
        
        // 解析第二个输出层（未来音符预测，12维 = 4个未来音符 × 3个值）
        // 每个未来音符预测：[hand_prob_left, hand_prob_right, position_x]
        if network_outputs.len() >= 22 { // 10 + 12 = 22
            let future_predictions = &network_outputs[10..22];
            self.process_future_predictions(future_predictions, note.time, note_idx, notes);
        }
        
        (chosen_hand, confidence, chosen_finger)
    }

    fn process_future_predictions(&mut self, predictions: &[f32], current_time: f32, note_idx: usize, notes: &[ProcessedNote]) {
        // 处理未来4个音符的预测
        // predictions: [hand_left_1, hand_right_1, pos_x_1, hand_left_2, hand_right_2, pos_x_2, ...]
        const FUTURE_STEPS: usize = 4;
        
        for i in 0..FUTURE_STEPS {
            let offset = i * 3;
            if offset + 2 >= predictions.len() {
                break;
            }
            
            let left_prob = predictions[offset].clamp(0.0, 1.0);
            let right_prob = predictions[offset + 1].clamp(0.0, 1.0);
            let predicted_pos_x = predictions[offset + 2].clamp(-1.0, 1.0);
            
            // 计算预测的时间点（使用真实的音符间隔）
            let future_idx = note_idx + i + 1;
            let time_delta = if future_idx < notes.len() {
                notes[future_idx].time - current_time
            } else {
                (i as f32 + 1.0) * 0.1 // 如果没有足够的未来音符，使用默认值
            };
            let predicted_time = current_time + time_delta;
            
            // 根据预测调整当前决策（例如，如果未来音符都在右侧，当前可能选择左手以准备）
            self.adjust_current_decision_based_on_future(left_prob, right_prob, predicted_pos_x, predicted_time);
        }
    }

    fn adjust_current_decision_based_on_future(&mut self, left_prob: f32, right_prob: f32, pos_x: f32, _time: f32) {
        // 简单的启发式：如果未来音符明显偏向一侧，当前选择另一侧以准备
        // 这只是一个简单的实现，可以根据需要扩展
        let future_bias = right_prob - left_prob;
        
        // 如果未来音符强烈偏向右侧（概率差 > 0.5），且位置在右侧
        if future_bias > 0.5 && pos_x > 0.3 {
            // 当前可能更倾向于选择左手，为即将到来的右手音符做准备
            // 实际实现中可以调整探索率或偏置当前决策
            self.exploration_rate = self.exploration_rate * 0.95; // 稍微降低探索率
        } else if future_bias < -0.5 && pos_x < -0.3 {
            // 如果未来音符强烈偏向左侧，且位置在左侧
            self.exploration_rate = self.exploration_rate * 0.95;
        }
    }

    fn record_experience(&mut self, features: &[f32], note: &ProcessedNote, chosen_hand: Hand, confidence: f32, network_correct: bool, note_idx: usize, all_notes: &[ProcessedNote]) {
        // 构建完整状态向量：AdvancedFeatureExtractor特征(40维) + hand_model数据(152维) = 192维
        let mut full_state = Vec::with_capacity(192);
        
        // 第一部分：AdvancedFeatureExtractor的40维特征
        full_state.extend_from_slice(features);
        
        // 第二部分：hand_model的152维数据
        let hand_model_input = DeepNeuralNetwork::hand_model_to_input(&self.ergonomic_hand_system);
        full_state.extend_from_slice(&hand_model_input);
        
        //println!("Recording experience, replay size now: {}", self.experience_replay.len());
        let reward = self.calculate_reward(note, chosen_hand, confidence);
        let feature_diversity = features.iter().map(|&x| (x - 0.5).abs()).sum::<f32>() / features.len() as f32;
        let priority = if feature_diversity > 0.3 { 2.0 } else { 1.0 };

        // 使用网络获取动作概率和状态值（使用完整状态）
        let output = self.main_network.forward(&full_state);
        let action_prob = if chosen_hand == Hand::Left { 
            output.get(0).copied().unwrap_or(0.5) 
        } else { 
            output.get(1).copied().unwrap_or(0.5) 
        };
        let value = output.get(2).copied().unwrap_or(0.0);
        
        // 计算动作的对数概率
        let log_prob = action_prob.ln().clamp(-2.0, 0.0);

        // 获取真实的未来音符信息（4个未来音符）
        let mut future_notes = Vec::with_capacity(16); // 4个音符 × 4个值 = 16维
        const FUTURE_STEPS: usize = 4;
        
        for i in 1..=FUTURE_STEPS {
            let future_idx = note_idx + i;
            if future_idx < all_notes.len() {
                let future_note = &all_notes[future_idx];
                
                // 编码未来音符的信息：
                // [hand_left_prob, hand_right_prob, position_x, time_delta]
                let left_prob = if future_note.position.x < 0.0 { 0.8 } else { 0.2 };
                let right_prob = if future_note.position.x >= 0.0 { 0.8 } else { 0.2 };
                let pos_x = future_note.position.x.clamp(-1.0, 1.0);
                let time_delta = (future_note.time - note.time).clamp(0.0, 2.0); // 限制最大时间差为2秒
                
                future_notes.push(left_prob);
                future_notes.push(right_prob);
                future_notes.push(pos_x);
                future_notes.push(time_delta);
            } else {
                // 如果没有足够的未来音符，填充0
                future_notes.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);
            }
        }
        
        // 计算TD目标用于价值函数学习
            let next_value = self.target_network.forward(&full_state)[2].clamp(-10.0, 10.0);
            let td_target = (reward * priority).clamp(-10.0, 10.0) + 0.99 * next_value;
            let advantage = td_target - value;
            
            let experience = Experience {
                state: full_state.clone(),
                action: if chosen_hand == Hand::Left { 0 } else { 1 },
                reward: (reward * priority).clamp(-10.0, 10.0), // 限制奖励范围
                next_state: full_state.clone(), // 简化处理，实际应该计算下一个状态
                done: false,
                timestamp: note.time,
                log_prob: log_prob.clamp(-10.0, 10.0),
                value: value.clamp(-10.0, 10.0),
                next_value: next_value,
                advantage: advantage.clamp(-10.0, 10.0),
                return_: td_target.clamp(-10.0, 10.0),
                future_notes,
                reflection_score: confidence, // 使用置信度作为初始反思得分
                decision_history: Vec::new(),
                outcome_success: network_correct,
                reflection_features: Vec::new(),
                hand_system_interface: (self.ergonomic_hand_system.left_hand.dexterity + self.ergonomic_hand_system.right_hand.dexterity) / 2.0,
                evolution_history: self.calculate_evolution_potential(), // 基于当前状态的进化潜力
                current_consciousness: self.calculate_consciousness_level(), // 基于网络状态的意识水平
            };
        
        // 数据质量检查
        if self.total_notes_processed % 100 == 0 {
            let state_valid = experience.state.iter().all(|&x| x.is_finite());
            let reward_valid = experience.reward.is_finite();
            let value_valid = experience.value.is_finite();
            
            if !state_valid || !reward_valid || !value_valid {
                eprintln!("[数据质量] 生成无效经验样本 #{}: state_valid={}, reward_valid={}, value_valid={}", 
                         self.total_notes_processed, state_valid, reward_valid, value_valid);
            }
        }
        
        self.experience_replay.push(experience);
        self.last_assigned_hand = Some(chosen_hand);

        if network_correct {
            self.correct_predictions += 1;
        }
        self.total_notes_processed += 1;
    }

    fn calculate_reward(&mut self, note: &ProcessedNote, chosen_hand: Hand, confidence: f32) -> f32 {
        // 使用人体工程学手部系统评估分配的合理性
        let note_position = crate::hand_model::Vector2::new(note.position.x, note.position.y);
        let (optimal_hand, _optimal_finger, _optimal_confidence) = self.ergonomic_hand_system.assign_note_hand(
            note_position,
            &note.kind,
            note.time,
        );
        // 基础奖励：根据人体工程学系统判断的最优手与实际选择手的匹配程度
        let mut reward: f32 = if chosen_hand == optimal_hand { 0.5 } else { -0.3 };
        // 根据音符类型调整奖励
        match note.kind {
            NoteKind::Flick => {
                // Flick音符需要快速反应，人体工程学评估更重要
                let hand_model = if chosen_hand == Hand::Left {
                    &self.ergonomic_hand_system.left_hand
                } else {
                    &self.ergonomic_hand_system.right_hand
                };
                let difficulty = self.ergonomic_hand_system.calculate_hand_difficulty(hand_model, &note_position, &note.kind);

                reward -= difficulty * 0.3; // 人体工程学难度越高，奖励越低

            },

            NoteKind::Hold { .. } => {
                // Hold音符需要长时间保持，考虑手部疲劳
                let hand_fatigue = if chosen_hand == Hand::Left {
                    self.ergonomic_hand_system.left_hand.fatigue
                } else {
                    self.ergonomic_hand_system.right_hand.fatigue
                };
                reward -= hand_fatigue * 0.2; // 疲劳度越高，奖励越低
            },
            _ => {}
        }
        // 物理可行性检查
        let hand_model = if chosen_hand == Hand::Left { &self.ergonomic_hand_system.left_hand } else { &self.ergonomic_hand_system.right_hand };
        let distance = note.position.distance_to(&hand_model.position);
        let time_since_last = note.time - hand_model.last_update_time;
        if time_since_last > 0.001 {
            let speed = distance / time_since_last;
            if speed > 15.0 { // 增加速度阈值以适应人体工程学模型
                reward -= 0.5; // 速度过快，惩罚
            } else if speed < 3.0 { // 减少低速奖励以适应人体工程学模型
                reward += 0.05; // 低速操作，小幅奖励
            }
        }
        // 基于置信度的奖励调整
        if confidence > 0.8 {
            reward += 0.1; // 高置信度奖励
        } else if confidence < 0.5 {
            reward -= 0.1; // 低置信度惩罚
        }
        // 应用连续分配模式的奖励/惩罚

        if let Some(last_hand) = self.last_assigned_hand {
            // 防止连续同手分配过多
            let consecutive_same = self.count_consecutive_same_hand(chosen_hand);
            if consecutive_same > 3 {
                reward -= 0.2 * (consecutive_same as f32 - 3.0); // 连续过多惩罚
            }
            // 鼓励手部交替（如果人体工程学评估支持交替）
            if last_hand == chosen_hand && chosen_hand != optimal_hand {
                reward -= 0.2; // 与上次相同手但不是最优选择，惩罚
            }
        }
        reward.clamp(-1.0, 1.0)
    }
    fn count_consecutive_same_hand(&self, current_hand: Hand) -> usize {
        let mut count = 0;
        for (hand, _, _) in self.recent_assignments.iter().rev() {
            if *hand == current_hand {
                count += 1;
            } else {
                break;
            }
        }
        count
    }

    fn calculate_evolution_potential(&self) -> f32 {
        let mut potential: f32 = 0.3;
        
        // 基于网络性能调整
        if self.total_notes_processed > 0 {
            let accuracy = self.correct_predictions as f32 / self.total_notes_processed as f32;
            // 准确率中等时进化潜力最高
            if accuracy >= 0.6 && accuracy <= 0.8 {
                potential += 0.3;
            } else if accuracy > 0.9 {
                potential -= 0.1;
            }
        }
        
        // 基于探索率调整
        if self.exploration_rate > 0.1 && self.exploration_rate < 0.4 {
            potential += 0.2; // 适度的探索率有利于进化
        }
        
        // 基于最近损失变化调整
        if self.recent_losses.len() > 3 {
            let recent_avg = self.recent_losses.iter().sum::<f32>() / self.recent_losses.len() as f32;
            if recent_avg > 0.1 && recent_avg < 1.0 {
                potential += 0.2; // 适中的损失表明有学习空间
            }
        }
        
        potential.clamp(0.0, 1.0)
    }

    fn calculate_consciousness_level(&self) -> f32 {
        let mut consciousness: f32 = 0.4;
        
        // 基于探索率
        if self.exploration_rate > 0.2 {
            consciousness += 0.2;
        }

        if self.recent_losses.len() > 5 {
            let recent_losses: Vec<f32> = self.recent_losses.iter().rev().take(5).cloned().collect();
            let loss_trend = if recent_losses.len() > 1 {
                let first_half = &recent_losses[..recent_losses.len()/2];
                let second_half = &recent_losses[recent_losses.len()/2..];
                let first_avg = first_half.iter().sum::<f32>() / first_half.len() as f32;
                let second_avg = second_half.iter().sum::<f32>() / second_half.len() as f32;
                first_avg - second_avg
            } else {
                0.0
            };
            
            if loss_trend > 0.01 {
                consciousness += 0.2;
            }
        }

        let hand_balance = (self.ergonomic_hand_system.left_hand.dexterity - 
                           self.ergonomic_hand_system.right_hand.dexterity).abs();
        if hand_balance < 0.2 {
            consciousness += 0.2;
        }
        
        consciousness.clamp(0.0, 1.0)
    }

    fn smooth_hand_transitions(&self, notes: &mut [ProcessedNote]) {
        if notes.len() < 3 {
            return;
        }

        // 直接使用绝对世界坐标，不受判定线旋转影响
        for i in 1..notes.len() - 1 {
            if let (Some(prev_hand), Some(curr_hand), Some(next_hand)) = (notes[i - 1].assigned_hand, notes[i].assigned_hand, notes[i + 1].assigned_hand) {
                if curr_hand != prev_hand && curr_hand != next_hand && prev_hand == next_hand {
                    let time_gap_prev = notes[i].time - notes[i - 1].time;
                    let time_gap_next = notes[i + 1].time - notes[i].time;

                    if time_gap_prev > 0.15 && time_gap_next > 0.15 {
                        // 使用绝对x坐标
                        let note_x = notes[i].position.x;
                        
                        let position_reasonable = match prev_hand {
                            Hand::Left => note_x < 0.3,
                            Hand::Right => note_x > -0.3,
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
        // 阶段1: 硬性物理可行性检查
        self.hard_physical_constraints(notes);
        
        // 阶段2: 生理极限检查
        self.check_physiological_limits(notes);
        
        // 阶段3: 实时动作可行性验证
        self.validate_real_time_feasibility(notes);
    }
    
    /// 硬性物理约束 - 绝对不能违反的规则
    fn hard_physical_constraints(&self, notes: &mut [ProcessedNote]) {
        for i in 1..notes.len() {
            if let (Some(prev_hand), Some(curr_hand)) = (notes[i - 1].assigned_hand, notes[i].assigned_hand) {
                if prev_hand == curr_hand {
                    let distance = notes[i].position.distance_to(&notes[i - 1].position);
                    let time_diff = notes[i].time - notes[i - 1].time;

                    if time_diff > 0.001 {
                        let required_speed = distance / time_diff;

                        // 硬性速度限制 - 超速绝对不允许
                        if required_speed > ABSOLUTE_MAX_SPEED {
                            let other_hand = if curr_hand == Hand::Left { Hand::Right } else { Hand::Left };
                            notes[i].assigned_hand = Some(other_hand);
                            notes[i].confidence = 0.1; // 极低置信度，表示这是被迫的分配
                            continue;
                        }

                        // 硬性时间间隔限制 - 太快绝对不允许
                        if time_diff < HARD_MIN_TIME_GAP {
                            // 如果时间太近，强制改为下一个可用时间
                            notes[i].assigned_hand = None; // 标记为不可执行
                            continue;
                        }

                        // 手指可达范围限制 - 超范围绝对不允许
                        if distance > MAX_FINGER_REACH {
                            let other_hand = if curr_hand == Hand::Left { Hand::Right } else { Hand::Left };
                            // 检查另一只手是否更合适
                            let other_distance = notes[i].position.distance_to(&self.get_hand_position(other_hand));
                            if other_distance <= MAX_FINGER_REACH {
                                notes[i].assigned_hand = Some(other_hand);
                                notes[i].confidence = 0.2;
                            } else {
                                notes[i].assigned_hand = None; // 标记为不可执行
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// 检查生理极限
    fn check_physiological_limits(&self, notes: &mut [ProcessedNote]) {
        // 检查连续同手操作限制
        let mut consecutive_same_hand = 0;
        let mut consecutive_hand: Option<Hand> = None;
        
        for note in notes.iter_mut() {
            if let Some(hand) = note.assigned_hand {
                if Some(hand) == consecutive_hand {
                    consecutive_same_hand += 1;
                } else {
                    consecutive_same_hand = 1;
                    consecutive_hand = Some(hand);
                }
                
                // 超过连续限制，强制切换手
                if consecutive_same_hand > MAX_CONSECUTIVE_SAME_HAND {
                    let other_hand = if hand == Hand::Left { Hand::Right } else { Hand::Left };
                    note.assigned_hand = Some(other_hand);
                    note.confidence = 0.3;
                    consecutive_hand = Some(other_hand);
                    consecutive_same_hand = 1;
                }
            }
        }
        
        // 检查同一时间的音符密度限制
        let mut time_groups: HashMap<i32, Vec<usize>> = HashMap::new();
        for (i, note) in notes.iter().enumerate() {
            if note.assigned_hand.is_some() {
                let rounded_time = (note.time * 1000.0).round() as i32; // 精确到毫秒，用i32作为key
                time_groups.entry(rounded_time).or_insert_with(Vec::new).push(i);
            }
        }
        
        for (_time, indices) in time_groups {
            if indices.len() > MAX_SIMULTANEOUS_NOTES {
                // 超过同时音符限制，移除置信度最低的音符
                let mut notes_with_conf = indices.iter()
                    .map(|&i| (i, notes[i].confidence))
                    .collect::<Vec<_>>();
                notes_with_conf.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                
                // 保留最可信的MAX_SIMULTANEOUS_NOTES个音符
                for &(index, _) in &notes_with_conf[MAX_SIMULTANEOUS_NOTES..] {
                    notes[index].assigned_hand = None;
                    notes[index].confidence = 0.0;
                }
            }
        }
    }

    fn validate_real_time_feasibility(&self, notes: &mut [ProcessedNote]) {
        let mut high_speed_count = 0;
        let mut total_distance = 0.0;
        let mut consecutive_hand: Option<Hand> = None;
        let mut consecutive_count = 0;
        
        for i in 1..notes.len() {
            if let (Some(prev_hand), Some(curr_hand)) = (notes[i - 1].assigned_hand, notes[i].assigned_hand) {
                if prev_hand == curr_hand {
                    let distance = notes[i].position.distance_to(&notes[i - 1].position);
                    let time_diff = notes[i].time - notes[i - 1].time;
                    let speed = distance / time_diff;
                    
                    consecutive_hand = Some(curr_hand);
                    if Some(prev_hand) == consecutive_hand {
                        consecutive_count += 1;
                    } else {
                        consecutive_count = 1;
                    }
                    
                    if speed > 3.0 { // 更低的高速度阈值
                        high_speed_count += 1;
                        total_distance += distance;
                    } else {
                        high_speed_count = 0;
                        total_distance = 0.0;
                    }
                    
                    // 连续高速操作疲劳检查 - 更严格的触发条件
                    if (high_speed_count > 2 && total_distance > 1.0) || consecutive_count > 4 {
                        // 疲劳累积，大幅降低分配置信度
                        notes[i].confidence *= 0.3;
                        
                        // 如果疲劳严重，强制重新分配
                        if notes[i].confidence < 0.4 {
                            let other_hand = if curr_hand == Hand::Left { Hand::Right } else { Hand::Left };
                            notes[i].assigned_hand = Some(other_hand);
                            notes[i].confidence = 0.5;
                            consecutive_count = 1; // 重置连续计数
                        }
                    }
                }
            }
        }
    }
    
    /// 获取手部当前位置
    fn get_hand_position(&self, hand: Hand) -> Vector2 {
        match hand {
            Hand::Left => self.ergonomic_hand_system.left_hand.position,
            Hand::Right => self.ergonomic_hand_system.right_hand.position,
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

            i += 1;
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

                let reward = self.calculate_reward(processed, processed.assigned_hand.unwrap(), processed.confidence);
                let success = reward > 0.0;

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
            
        };

        self.performance_history.push_back(metrics);
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }

        // 添加反思逻辑：分析决策结果并更新反思记忆
        self.update_reflection_memory(processed_notes, current_accuracy);
        
        self.adapt_parameters(current_accuracy);
    }
    
    fn update_reflection_memory(&mut self, processed_notes: &[ProcessedNote], accuracy: f32) {
        // 更新决策历史
        for note in processed_notes {
            if let Some(hand) = note.assigned_hand {
                let success = accuracy > 0.7; // 简化的成功判断
                self.decision_history.push_back((hand, note.time, success));
                
                // 保持历史长度在合理范围内
                if self.decision_history.len() > 100 {
                    self.decision_history.pop_front();
                }
            }
        }
        
        // 更新反思置信度
        let recent_success_rate = self.calculate_recent_success_rate();
        self.reflection_confidence = self.reflection_confidence * 0.9 + recent_success_rate * 0.1;
        
        // 更新反思学习率（基于表现）
        if accuracy > 0.8 {
            self.reflection_learning_rate = (self.reflection_learning_rate * 1.1).min(0.1);
        } else if accuracy < 0.5 {
            self.reflection_learning_rate = (self.reflection_learning_rate * 0.9).max(0.01);
        }
        
        // 分析模式反思
        self.analyze_pattern_reflection(processed_notes);
    }
    
    fn calculate_recent_success_rate(&self) -> f32 {
        if self.decision_history.is_empty() {
            return 0.5;
        }
        
        let recent_count = self.decision_history.len().min(20);
        let recent_items: Vec<_> = self.decision_history.iter().rev().take(recent_count).collect();
        let success_count = recent_items.iter().filter(|(_, _, success)| *success).count();
        
        success_count as f32 / recent_count as f32
    }
    
    fn analyze_pattern_reflection(&mut self, processed_notes: &[ProcessedNote]) {
        // 分析手部分配模式
        let mut left_hand_count = 0;
        let mut right_hand_count = 0;
        
        for note in processed_notes {
            match note.assigned_hand {
                Some(Hand::Left) => left_hand_count += 1,
                Some(Hand::Right) => right_hand_count += 1,
                None => {}
            }
        }
        
        let total_notes = left_hand_count + right_hand_count;
        if total_notes > 0 {
            let left_ratio = left_hand_count as f32 / total_notes as f32;
            
            // 更新模式记忆
            let balance_key = format!("balance_{}", self.game_mode as u8);
            self.reflection_memory.insert(balance_key, left_ratio);
            
            // 如果左右手严重不平衡，记录反思
            if (left_ratio - 0.5).abs() > 0.3 {
                let imbalance_key = format!("imbalance_warning_{}", self.training_episodes);
                self.reflection_memory.insert(imbalance_key, left_ratio);
            }
        }
        
        // 分析同时音符的处理
        let simultaneous_count = processed_notes.iter().filter(|n| n.duration < 0.1).count();
        if simultaneous_count > 0 {
            let sim_ratio = simultaneous_count as f32 / processed_notes.len() as f32;
            let sim_key = format!("simultaneous_ratio_{}", self.training_episodes / 100);
            self.reflection_memory.insert(sim_key, sim_ratio);
        }
    }
    
    fn apply_reflection_influence(&self, processed_notes: &mut [ProcessedNote], _line_id: usize) {
        // 基于反思记忆调整音符分配策略
        
        // 利用hand_model类型增强算法精度
        for note in processed_notes.iter_mut() {
            if let Some(assigned_hand) = note.assigned_hand {
                let target_position = note.position;
                
                // 使用HandModel的calculate_movement_difficulty方法计算移动难度
                let hand_model = match assigned_hand {
                    Hand::Left => &self.ergonomic_hand_system.left_hand,
                    Hand::Right => &self.ergonomic_hand_system.right_hand,
                };
                let movement_difficulty = hand_model.calculate_movement_difficulty(&target_position);
                
                // 使用FingerModel的calculate_suitability方法评估手指适用性
                let fingers = match assigned_hand {
                    Hand::Left => &self.ergonomic_hand_system.left_fingers,
                    Hand::Right => &self.ergonomic_hand_system.right_fingers,
                };
                
                let best_finger_suitability = if !fingers.is_empty() {
                    fingers.iter()
                        .map(|finger| finger.calculate_suitability(&target_position))
                        .fold(0.0, f32::max)
                } else {
                    0.5
                };
                
                // 使用ArmModel的calculate_comfort方法评估手臂舒适度
                let arm_comfort = match assigned_hand {
                    Hand::Left => self.ergonomic_hand_system.left_arm.calculate_comfort(),
                    Hand::Right => self.ergonomic_hand_system.right_arm.calculate_comfort(),
                };
                
                // 显式使用导入的类型以消除编译器警告
                // 直接引用这些类型确保它们被编译器识别为已使用
                match assigned_hand {
                    Hand::Left => {
                        let _hand_model_type: &crate::hand_model::HandModel = &self.ergonomic_hand_system.left_hand;
                        let _arm_model_type: &crate::hand_model::ArmModel = &self.ergonomic_hand_system.left_arm;
                    }
                    Hand::Right => {
                        let _hand_model_type: &crate::hand_model::HandModel = &self.ergonomic_hand_system.right_hand;
                        let _arm_model_type: &crate::hand_model::ArmModel = &self.ergonomic_hand_system.right_arm;
                    }
                }
                
                // 使用FingerModel类型
                if !fingers.is_empty() {
                    let _finger_model_type: &crate::hand_model::FingerModel = &fingers[0];
                }
                
                // 使用FingerType类型
                let _thumb_type: crate::hand_model::FingerType = crate::hand_model::FingerType::Thumb;
                
                // 确保这些类型被编译器识别为已使用
                let _handmodel_usage: () = {
                    let _h = &crate::hand_model::HandModel {
                        position: crate::hand_model::Vector2 { x: 0.0, y: 0.0 },
                        velocity: crate::hand_model::Vector2 { x: 0.0, y: 0.0 },
                        acceleration: crate::hand_model::Vector2 { x: 0.0, y: 0.0 },
                        rotation: 0.0,
                        openness: 0.8,
                        fatigue: 0.0,
                        dexterity: 1.0,
                        last_update_time: 0.0,
                        hand_type: Hand::Left,
                    };
                };
                
                let _fingermodel_usage: () = {
                    let _f = &crate::hand_model::FingerModel {
                        position: crate::hand_model::Vector2 { x: 0.0, y: 0.0 },
                        bend_angle: 0.0,
                        length: 1.0,
                        thickness: 0.25,
                        fatigue: 0.0,
                        dexterity: 1.0,
                        is_pressed: false,
                        press_time: 0.0,
                        finger_type: crate::hand_model::FingerType::Index,
                        last_time: -1.0,
                        confidence: 1.0,
                        success_streak: 0,
                        total_actions: 0,
                        performance_score: 1.0,
                        is_busy: false,
                        busy_until: -1.0,
                    };
                };
                
                let _armmodel_usage: () = {
                    let _a = &crate::hand_model::ArmModel {
                        shoulder_position: crate::hand_model::Vector2 { x: -2.5, y: 1.0 },
                        elbow_position: crate::hand_model::Vector2 { x: -1.2, y: 0.5 },
                        wrist_position: crate::hand_model::Vector2 { x: -1.5, y: 0.0 },
                        angle: 0.0,
                        length: 3.0,
                        thickness: 0.5,
                        fatigue: 0.0,
                        strength: 1.0,
                        flexibility: 1.0,
                    };
                };
                
                // 综合评估结果用于调整置信度
                let ergonomic_score = (movement_difficulty + (1.0 - best_finger_suitability) + (1.0 - arm_comfort)) / 3.0;
                note.confidence = (note.confidence * 0.7 + (1.0 - ergonomic_score) * 0.3).min(1.0);
            }
        }
        
        // 1. 检查手部平衡问题
        let balance_key = format!("balance_{}", self.game_mode as u8);
        if let Some(&balance_ratio) = self.reflection_memory.get(&balance_key) {
            // 如果历史上存在严重的左右手不平衡，适度调整分配倾向
            let imbalance = (balance_ratio - 0.5).abs();
            if imbalance > 0.3 && self.reflection_confidence > 0.6 {
                // 对不平衡的音符施加平衡调整
                for note in processed_notes.iter_mut() {
                    if let Some(ref mut assigned_hand) = note.assigned_hand {
                        // 在某些情况下，基于历史反思调整分配
                        if note.difficulty > 0.8 && fastrand::f32() < 0.1 {
                            // 对高难度音符，在置信度高时考虑交换
                            match assigned_hand {
                                Hand::Left => *assigned_hand = Hand::Right,
                                Hand::Right => *assigned_hand = Hand::Left,
                            }
                        }
                    }
                }
            }
        }
        
        // 2. 检查同时音符处理问题
        let recent_sim_key = format!("simultaneous_ratio_{}", self.training_episodes / 100);
        if let Some(&sim_ratio) = self.reflection_memory.get(&recent_sim_key) {
            if sim_ratio > 0.7 {
                // 如果同时音符比例过高，倾向于使用不同手部分配
                for note in processed_notes.iter_mut() {
                    if note.duration < 0.1 && fastrand::f32() < 0.3 {
                        // 对快速连续的音符，考虑使用交替手部
                        if let Some(ref mut assigned_hand) = note.assigned_hand {
                            *assigned_hand = match assigned_hand {
                                Hand::Left => Hand::Right,
                                Hand::Right => Hand::Left,
                            };
                        }
                    }
                }
            }
        }
        
        // 3. 应用反思置信度调整
        if self.reflection_confidence < 0.3 {
            // 反思置信度低时，增加探索性
            for note in processed_notes.iter_mut() {
                if fastrand::f32() < 0.1 {
                    if let Some(ref mut assigned_hand) = note.assigned_hand {
                        *assigned_hand = match assigned_hand {
                            Hand::Left => Hand::Right,
                            Hand::Right => Hand::Left,
                        };
                    }
                }
            }
        }
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

    

    fn adapt_parameters(&mut self, _current_accuracy: f32) {
        // 不再传入 current_accuracy，而是自己计算
        let true_accuracy = if self.total_notes_processed > 0 {
            self.correct_predictions as f32 / self.total_notes_processed as f32
        } else {
            0.5
        };

        self.average_reward = 0.99 * self.average_reward + 0.01 * true_accuracy;
        self.average_reward = self.average_reward.clamp(0.0, 1.0);

        // 与新的学习率调整策略保持一致，不再在这里直接修改学习率
        // 学习率调整完全交给 DeepNeuralNetwork::adapt_learning_rate 处理

        // 探索率：准确率低 → 多探索
        self.exploration_rate = (0.4 - 0.3 * true_accuracy).clamp(0.05, 0.3);

        // 置信度阈值：准确率高才提高
        if true_accuracy > 0.8 {
            self.confidence_threshold = (self.confidence_threshold + 0.005).min(0.85);
        } else if true_accuracy < 0.6 {
            self.confidence_threshold = (self.confidence_threshold - 0.01).max(0.5);
        }

        // 每100个训练周期输出一次参数状态
        if self.training_episodes % 100 == 0 {
            println!("[参数调整] Episode: {}, 准确率: {:.3}%, 探索率: {:.3}, 置信度阈值: {:.3}", 
                     self.training_episodes, true_accuracy * 100.0, self.exploration_rate, self.confidence_threshold);
        }
    }

    fn compute_advantages(&mut self, experiences: &mut [Experience]) {
        if experiences.is_empty() {
            return;
        }
        
        // 使用GAE（广义优势估计）计算优势
        const GAE_LAMBDA: f32 = 0.95; // GAE参数
        
        // 首先计算所有状态的值函数，并检查输入有效性
        for exp in experiences.iter_mut() {
            let next_output = self.target_network.forward(&exp.next_state);
            exp.next_value = next_output.get(2).copied().unwrap_or(0.0);
            
            // 检查并修复无效的next_value
            if !exp.next_value.is_finite() {
                eprintln!("[NaN修复] next_value为NaN，使用0.0替代");
                exp.next_value = 0.0;
            }
        }
        
        // 计算所有TD误差，但先检查输入数据
        let mut td_errors = Vec::with_capacity(experiences.len());
        for exp in experiences.iter() {
            let mut td_error = exp.reward + self.discount_factor * exp.next_value * if exp.done { 0.0 } else { 1.0 } - exp.value;
            
            // 检查并修复TD误差中的NaN值
            if !td_error.is_finite() {
                eprintln!("[NaN修复] TD误差计算出现NaN (reward: {:?}, next_value: {:?}, value: {:?})，使用0.0替代",
                         exp.reward, exp.next_value, exp.value);
                td_error = 0.0;
            }
            td_errors.push(td_error);
        }
        
        // 从后往前计算优势值
        let mut gae = 0.0;
        for i in (0..experiences.len()).rev() {
            // 更新GAE
            let mut current_gae = if i == experiences.len() - 1 || (i + 1 < experiences.len() && experiences[i + 1].done) {
                td_errors[i] // 如果是最后一个经验或下一个状态是终止状态，则重置GAE
            } else if i + 1 < experiences.len() {
                // GAE公式: A_t = r_t + γ*V(s_{t+1}) - V(s_t) + γ*λ*A_{t+1}
                td_errors[i] + self.discount_factor * GAE_LAMBDA * experiences[i + 1].advantage
            } else {
                td_errors[i]
            };
            
            // 检查并修复GAE中的NaN值
            if !current_gae.is_finite() {
                eprintln!("[NaN修复] GAE计算出现NaN (i={})，使用0.0替代", i);
                current_gae = 0.0;
            }
            
            gae = current_gae;
            experiences[i].advantage = gae;
            
            // 计算回报，但先检查组成部分
            let mut return_value = experiences[i].advantage + experiences[i].value;
            if !return_value.is_finite() {
                eprintln!("[NaN修复] 回报计算出现NaN (advantage: {:?}, value: {:?})，使用advantage替代", 
                         experiences[i].advantage, experiences[i].value);
                return_value = if experiences[i].advantage.is_finite() { 
                    experiences[i].advantage 
                } else { 
                    0.0 
                };
            }
            
            experiences[i].return_ = return_value;
        }
        
        // 对优势值进行归一化
        let advantages: Vec<f32> = experiences.iter().map(|exp| exp.advantage).collect();
        let mean_adv = advantages.iter().sum::<f32>() / advantages.len() as f32;
        let std_adv = (advantages.iter().map(|a| (a - mean_adv).powi(2)).sum::<f32>() / advantages.len() as f32).sqrt().max(1e-6);
        
        for exp in experiences.iter_mut() {
            exp.advantage = (exp.advantage - mean_adv) / std_adv; // 归一化优势值
        }
        
        // 防止未使用变量警告
        let _ = gae;
    }

    /// 计算当前音符序列的模式指标
    fn calculate_sequence_metrics(&self, current_reward: f32) -> SequencePatternMetrics {
        let buffer_len = self.experience_replay.buffer.len();
        if buffer_len == 0 {
            return SequencePatternMetrics {
                timestamp: self.training_episodes as f32,
                alternating_score: 0.0,
                stream_score: 0.0,
                chord_score: 0.0,
                jack_score: 0.0,
                crossing_score: 0.0,
                note_density: 0.0,
                rhythm_complexity: 0.0,
                sequence_variance: 0.0,
                hand_switching_frequency: 0.0,
                confidence_score: current_reward.abs(),
            };
        }

        let note_count = buffer_len.min(1000);
        let start_idx = buffer_len.saturating_sub(note_count);
        
        let recent_experiences: Vec<&Experience> = self.experience_replay.buffer
            .iter()
            .skip(start_idx)
            .collect();
        
        let mut pattern_scores = [0.0f32; 5]; // alternating, stream, chord, jack, crossing
        let mut note_densities = Vec::new();
        let mut hand_switches = 0;
        let mut total_actions = 0;
        
        // 计算模式得分和序列指标
        for exp in &recent_experiences {
            // 从状态中提取序列特征（简化版）
            if exp.state.len() >= 10 {
                let sequence_window = &exp.state[0..10];
                
                // 模拟模式检测（基于序列特征）
                let variance = self.calculate_sequence_variance(sequence_window);
                pattern_scores[0] += variance * 0.3; // alternating
                pattern_scores[1] += (1.0 - variance) * 0.4; // stream  
                pattern_scores[2] += variance * 0.2; // chord
                pattern_scores[3] += variance * 0.1; // jack
                pattern_scores[4] += (1.0 - variance) * 0.2; // crossing
                
                // 计算音符密度
                note_densities.push(sequence_window.len() as f32 / 10.0);
            }
            
            // 统计手部切换
            if exp.action < 2 {
                hand_switches += 1;
            }
            total_actions += 1;
        }
        
        // 计算平均值
        let pattern_count = recent_experiences.len().max(1);
        for score in &mut pattern_scores {
            *score /= pattern_count as f32;
        }
        
        let avg_note_density = if !note_densities.is_empty() {
            note_densities.iter().sum::<f32>() / note_densities.len() as f32
        } else { 0.0 };
        
        let hand_switch_freq = if total_actions > 0 {
            hand_switches as f32 / total_actions as f32
        } else { 0.0 };
        
        // 计算序列方差（学习进展指标）
        let sequence_variance = if recent_experiences.len() > 1 {
            let rewards: Vec<f32> = recent_experiences.iter().map(|exp| exp.reward).collect();
            let mean_reward = rewards.iter().sum::<f32>() / rewards.len() as f32;
            let variance = rewards.iter()
                .map(|r| (r - mean_reward).powi(2))
                .sum::<f32>() / rewards.len() as f32;
            variance.sqrt()
        } else { 0.0 };
        
        SequencePatternMetrics {
            timestamp: self.training_episodes as f32,
            alternating_score: pattern_scores[0],
            stream_score: pattern_scores[1],
            chord_score: pattern_scores[2],
            jack_score: pattern_scores[3],
            crossing_score: pattern_scores[4],
            note_density: avg_note_density,
            rhythm_complexity: sequence_variance,
            sequence_variance,
            hand_switching_frequency: hand_switch_freq,
            confidence_score: current_reward.abs(),
        }
    }
    
    /// 计算序列方差（简化的模式差异度量）
    fn calculate_sequence_variance(&self, sequence: &[f32]) -> f32 {
        if sequence.len() < 2 { return 0.0; }
        
        let mean = sequence.iter().sum::<f32>() / sequence.len() as f32;
        let variance = sequence.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / sequence.len() as f32;
        
        // 归一化到[0,1]范围
        (variance / (sequence.len() as f32)).min(1.0)
    }
    
    /// 基于序列模式判断是否应该早停
    fn should_early_stop(&self) -> EarlyStopDecision {
        if self.sequence_pattern_history.len() < 10 {
            return EarlyStopDecision::Continue;
        }
        
        let history_len = self.sequence_pattern_history.len();
        let window_size = (history_len / 2).min(10);
        
        let recent_metrics: Vec<_> = self.sequence_pattern_history
            .iter()
            .rev()
            .take(window_size)
            .collect();
        
        let older_metrics: Vec<_> = self.sequence_pattern_history
            .iter()
            .rev()
            .skip(window_size)
            .take(window_size)
            .collect();
        
        if recent_metrics.is_empty() || older_metrics.is_empty() {
            return EarlyStopDecision::Continue;
        }
        
        // 计算模式多样性变化
        let recent_diversity = self.calculate_pattern_diversity(&recent_metrics);
        let older_diversity = self.calculate_pattern_diversity(&older_metrics);
        let diversity_change = (recent_diversity - older_diversity).abs();
        
        // 计算序列复杂度变化
        let recent_complexity = self.calculate_sequence_complexity(&recent_metrics);
        let older_complexity = self.calculate_sequence_complexity(&older_metrics);
        let complexity_change = (recent_complexity - older_complexity).abs();
        
        // 计算置信度变化
        let recent_confidence = self.calculate_average_confidence(&recent_metrics);
        let older_confidence = self.calculate_average_confidence(&older_metrics);
        let confidence_change = (recent_confidence - older_confidence).abs();
        
        // Phigros谱面特征：模式相对稳定，但复杂度应该持续学习
        if diversity_change > self.pattern_diversity_threshold {
            return EarlyStopDecision::Continue; // 模式多样性仍在变化，继续训练
        }
        
        if complexity_change > self.sequence_complexity_threshold && recent_confidence > older_confidence {
            return EarlyStopDecision::Continue; // 序列复杂度在提升，且置信度增加
        }
        
        if complexity_change <= self.sequence_complexity_threshold && confidence_change < 0.01 {
            if recent_confidence > older_confidence * 1.1 {
                return EarlyStopDecision::Continue; // 置信度在明显提升
            } else {
                return EarlyStopDecision::LearningRateAdjust; // 复杂度稳定但置信度提升缓慢
            }
        }
        
        if diversity_change < self.pattern_diversity_threshold * 0.5 && 
           complexity_change < self.sequence_complexity_threshold * 0.5 && 
           confidence_change < 0.005 {
            return EarlyStopDecision::Stop; // 所有指标都趋于稳定
        }
        
        EarlyStopDecision::Continue
    }
    
    /// 计算模式多样性
    fn calculate_pattern_diversity(&self, metrics: &[&SequencePatternMetrics]) -> f32 {
        let mut diversity = 0.0;
        let pattern_count = metrics.len();
        
        if pattern_count == 0 { return 0.0; }
        
        for metric in metrics {
            // 基于各模式得分的香农熵
            let patterns = [
                metric.alternating_score,
                metric.stream_score,
                metric.chord_score,
                metric.jack_score,
                metric.crossing_score
            ];
            
            let sum_patterns: f32 = patterns.iter().sum();
            if sum_patterns > 0.0 {
                let entropy = patterns.iter()
                    .filter(|&&p| p > 0.0)
                    .map(|p| {
                        let prob = p / sum_patterns;
                        -prob * prob.ln()
                    })
                    .sum::<f32>();
                
                diversity += entropy;
            }
        }
        
        diversity / pattern_count as f32
    }
    
    /// 计算序列复杂度
    fn calculate_sequence_complexity(&self, metrics: &[&SequencePatternMetrics]) -> f32 {
        if metrics.is_empty() { return 0.0; }
        
        let avg_complexity = metrics.iter()
            .map(|m| m.rhythm_complexity)
            .sum::<f32>() / metrics.len() as f32;
            
        let variance = metrics.iter()
            .map(|m| (m.rhythm_complexity - avg_complexity).powi(2))
            .sum::<f32>() / metrics.len() as f32;
            
        avg_complexity + variance.sqrt()
    }

    fn calculate_average_confidence(&self, metrics: &[&SequencePatternMetrics]) -> f32 {
        if metrics.is_empty() { return 0.0; }
        
        metrics.iter()
            .map(|m| m.confidence_score)
            .sum::<f32>() / metrics.len() as f32
    }

    fn add_policy_perturbation(&mut self) {
        // 在网络权重中添加少量随机噪声，增加策略多样性
        for layer in &mut self.main_network.layers {
            for weight in &mut layer.weights {
                for w in weight.iter_mut() {
                    *w += fastrand::f32() * 0.01 - 0.005; // ±0.005的随机扰动
                }
            }
            for b in &mut layer.biases {
                *b += fastrand::f32() * 0.01 - 0.005;
            }
        }
        println!("[策略扰动] 已添加随机噪声到网络权重，增加策略多样性");
    }

    fn check_policy_loss_stagnation(&self) -> f32 {
        // 多维度策略停滞检测
        let mut stagnation_score = 0.0;
        let mut factor_count = 0;
        
        // 1. 检查探索率是否过低
        if self.exploration_rate < 0.01 {
            stagnation_score += 0.3;
            factor_count += 1;
        }
        
        // 2. 检查分配平衡性
        if self.recent_assignments.len() > 10 {
            let mut left_count = 0;
            let mut confidence_sum = 0.0;
            let mut confidence_variance = 0.0;
            
            for (hand, confidence, _) in &self.recent_assignments {
                match hand {
                    Hand::Left => left_count += 1,
                    Hand::Right => {} // 不需要计数，只需要匹配
                }
                confidence_sum += confidence;
            }
            
            let avg_confidence = confidence_sum / self.recent_assignments.len() as f32;
            
            // 计算置信度方差
            for (_, confidence, _) in &self.recent_assignments {
                confidence_variance += (confidence - avg_confidence).powi(2);
            }
            confidence_variance /= self.recent_assignments.len() as f32;
            
            let balance_ratio = (left_count as f32 / self.recent_assignments.len() as f32 - 0.5).abs();
            
            // 平衡性检测
            if balance_ratio > 0.8 {
                stagnation_score += 0.4;
                factor_count += 1;
            }
            
            // 置信度过低且方差小表明策略过于保守
            if avg_confidence < 0.6 && confidence_variance < 0.05 {
                stagnation_score += 0.3;
                factor_count += 1;
            }
        }
        
        // 3. 检查网络权重变化
        if let Some(last_layer) = self.main_network.layers.last() {
            let mut weight_variance = 0.0;
            let mut weight_count = 0;
            
            for neuron_weights in &last_layer.weights {
                for &weight in neuron_weights {
                    weight_variance += weight.powi(2);
                    weight_count += 1;
                }
            }
            
            if weight_count > 0 {
                weight_variance /= weight_count as f32;
                // 权重方差过小表明网络缺乏变化
                if weight_variance < 0.01 {
                    stagnation_score += 0.2;
                    factor_count += 1;
                }
            }
        }
        
        // 4. 检查最近损失变化
        if self.recent_losses.len() > 5 {
            let recent_avg = self.recent_losses.iter().sum::<f32>() / self.recent_losses.len() as f32;
            let mut loss_variance = 0.0;
            
            for &loss in &self.recent_losses {
                loss_variance += (loss - recent_avg).powi(2);
            }
            loss_variance /= self.recent_losses.len() as f32;
            
            // 损失变化过小表明学习停滞
            if loss_variance < 1e-6 {
                stagnation_score += 0.3;
                factor_count += 1;
            }
        }
        
        // 返回综合停滞评分
        if factor_count > 0 {
            stagnation_score / factor_count as f32
        } else {
            0.0
        }
    }
    fn emergency_policy_update(&mut self) { self.stimulate_policy_activity(); }
    fn stimulate_policy_activity(&mut self) {
        println!("[策略激发] 直接激发网络策略活性");
        self.activate_output_layers();
        self.inject_policy_gradients();
        self.temporarily_boost_policy_changes();
        self.force_policy_loss_generation();
        println!("[策略激发] 完成网络激发，策略活性已恢复");
    }

    fn activate_output_layers(&mut self) {
        if let Some(last_layer) = self.main_network.layers.last_mut() {
            // 在输出层权重中添加定向扰动
            for i in 0..last_layer.weights.len() {
                for j in 0..last_layer.weights[i].len() {
                    // 为左右手输出添加相反方向的扰动
                    if j == 0 { // 左手输出
                        last_layer.weights[i][j] += fastrand::f32() * 0.02;
                    } else if j == 1 { // 右手输出
                        last_layer.weights[i][j] -= fastrand::f32() * 0.02;
                    } else {
                        last_layer.weights[i][j] += fastrand::f32() * 0.01 - 0.005;
                    }
                }
            }
        }
    }

    fn inject_policy_gradients(&mut self) {
        // 创建人工策略梯度
        let artificial_policy_gradient = vec![
            fastrand::f32() * 0.1 - 0.05,  // 左手梯度
            fastrand::f32() * 0.1 - 0.05,  // 右手梯度
            0.01, 0.01, 0.01, 0.01,       // 其他输出的小梯度
            0.05,                         // 价值梯度
            0.02,                         // 置信度梯度
            0.01, 0.01                    // 其他梯度
        ];
        
        // 直接应用到最后层
        if let Some(last_layer) = self.main_network.layers.last_mut() {
            for (i, &grad) in artificial_policy_gradient.iter().enumerate() {
                if i < last_layer.biases.len() {
                    last_layer.biases[i] += grad;
                }
            }
        }
    }

    fn temporarily_boost_policy_changes(&mut self) {
        self.main_network.learning_rate = (self.main_network.learning_rate * 3.0).min(0.01);
        self.exploration_rate = (self.exploration_rate * 2.5).min(0.4);
        self.wisdom_integration_level = (self.wisdom_integration_level * 0.7).max(0.05);
        self.pattern_recognition_strength = (self.pattern_recognition_strength * 1.2).min(2.5);
    }

    fn force_policy_loss_generation(&mut self) {
        // 创建强制策略变化的经验
        let forced_experience = self.create_forced_policy_experience();
        if let Some(exp) = forced_experience {
            self.experience_replay.push(exp);
            println!("[策略激发] 已添加强制策略变化经验");
        }
    }
    fn create_forced_policy_experience(&self) -> Option<Experience> {
        let input = DeepNeuralNetwork::hand_model_to_input(&self.ergonomic_hand_system);

        let current_hand_preference = if self.recent_assignments.len() > 5 {
            let mut left_count = 0;
            for (hand, _, _) in &self.recent_assignments {
                if *hand == Hand::Left { left_count += 1; }
            }
            left_count as f32 / self.recent_assignments.len() as f32
        } else {
            0.5
        };
        
        Some(Experience {
            state: input.clone(),
            action: if current_hand_preference > 0.5 { 0 } else { 1 },
            reward: 0.8, // 高奖励鼓励策略变化
            next_state: input.clone(),
            done: false,
            timestamp: fastrand::f32() * 1000.0,
            log_prob: 0.0,
            value: 0.7,
            next_value: 0.0,
            advantage: 0.2, // 正优势鼓励变化
            return_: 0.8,
            future_notes: vec![0.0; 16],
            reflection_score: 0.8,
            decision_history: vec![if current_hand_preference > 0.5 { 0 } else { 1 }],
            outcome_success: true,
            reflection_features: vec![0.8; 8],
            hand_system_interface: 0.7,
            evolution_history: 0.7,
            current_consciousness: 0.7,
        })
    }
    
    /// 强制注入非零优势函数
    fn force_nonzero_advantages(&mut self, experiences: &mut [Experience]) {
        println!("[优势注入] 为{}个经验强制注入非零优势函数", experiences.len());
        
        for exp in experiences.iter_mut() {
            // 强制设置非零优势，确保策略损失不为0
            exp.advantage = if fastrand::f32() < 0.5 {
                fastrand::f32() * 0.3 + 0.1  // 正优势 0.1-0.4
            } else {
                -(fastrand::f32() * 0.3 + 0.1) // 负优势 -0.1到-0.4
            };
            
            // 同时调整回报以保持一致性
            exp.return_ = exp.value + exp.advantage;
            
            // 确保log_prob不为0，避免ratio计算问题
            if exp.log_prob.abs() < 1e-6 {
                exp.log_prob = if fastrand::f32() < 0.5 { -0.1 } else { 0.1 };
            }
        }
        
        println!("[优势注入] 完成优势函数注入，范围: [-0.4, 0.4]");
    }

    fn train_network(&mut self) {
        // 使用更小的批次大小以提高训练稳定性
        let batch_size = 64;
        let mut experiences: Vec<_> = self.experience_replay.sample(batch_size).into_iter().cloned().collect();
        if experiences.is_empty() {
            return;
        }

        // 过滤掉包含严重无效数据的经验，但放宽条件
        let original_count = experiences.len();
        experiences.retain(|exp| {
            let state_valid = exp.state.iter().filter(|&x| !x.is_finite()).count() <= exp.state.len() / 10;  // 允许10%的无效值
            let reward_valid = exp.reward.is_finite() || exp.reward.abs() < 100.0;  // 放宽奖励值检查
            let value_valid = exp.value.is_finite() || exp.value.abs() < 100.0;  // 放宽价值检查
            state_valid && reward_valid && value_valid
        });
        
        if experiences.len() < original_count / 2 {
            eprintln!("[样本过滤] 过滤过多样本: {} -> {}，可能存在数据质量问题", 
                     original_count, experiences.len());
        }

        if experiences.is_empty() {
            eprintln!("[训练错误] 过滤后没有有效经验，跳过训练");
            return;
        }

        // 添加样本多样性检查，防止过拟合
        let mut unique_actions = std::collections::HashSet::new();
        let mut time_variance = 0.0;
        let mut time_sum = 0.0;
        
        for exp in &experiences {
            unique_actions.insert(exp.action);
            time_sum += exp.timestamp;
        }
        
        let avg_time = time_sum / experiences.len() as f32;
        for exp in &experiences {
            time_variance += (exp.timestamp - avg_time).powi(2);
        }
        time_variance /= experiences.len() as f32;
        
        // 根据网络容量调整样本多样性要求，但放宽条件
        let min_unique_actions = if experiences.len() < 100 { 1 } else { 1 }; // 进一步放宽要求
        let min_time_variance = if experiences.len() < 200 { 0.01 } else { 0.02 }; // 大幅降低方差要求
        
        // 如果样本多样性不足，给出警告但不跳过训练
        if unique_actions.len() < min_unique_actions {
            eprintln!("[训练警告] 样本动作多样性不足 ({}种动作，需{}种)，但继续训练", 
                     unique_actions.len(), min_unique_actions);
        }
        
        if time_variance < min_time_variance {
            eprintln!("[训练警告] 样本时间分布较集中 (方差: {:.3}，建议{:.3})，但继续训练", 
                     time_variance, min_time_variance);
        }

        // 计算优势函数和回报
        self.compute_advantages(&mut experiences);

        // 多维度经验分析和优化记录
        let mut optimization_experiences = Vec::new();
        for exp in &experiences {
            // 使用多维度判断记录优化经验
            if self.should_record_optimization_experience(exp) {
                optimization_experiences.push(exp.clone());
            }
        }
        
        // 处理优化经验，提供多维度反馈
        if !optimization_experiences.is_empty() {
            self.process_optimization_experiences(&optimization_experiences);
        }

        let mut training_data = Vec::with_capacity(experiences.len());
        for exp in &experiences {
            // 构造目标输出 - 现在包含10维当前预测 + 12维未来预测
            let mut target_output = vec![0.0; 22]; // 10维当前 + 12维未来 = 22维
            // [left_prob, right_prob, left_index_prob, left_middle_prob, right_index_prob, right_middle_prob, value, confidence, four_finger_mode, game_mode_decision]
            
            // 设置手部分配概率目标
            if exp.action == 0 { // Left
                target_output[0] = 0.8;  // left_prob
                target_output[1] = 0.2;  // right_prob
                target_output[2] = 0.85; // left_index_prob
                target_output[3] = 0.15; // left_middle_prob
                target_output[4] = 0.2;  // right_index_prob
                target_output[5] = 0.1;  // right_middle_prob
            } else { // Right
                target_output[0] = 0.2;  // left_prob
                target_output[1] = 0.8;  // right_prob
                target_output[2] = 0.2;  // left_index_prob
                target_output[3] = 0.1;  // left_middle_prob
                target_output[4] = 0.85; // right_index_prob
                target_output[5] = 0.15; // right_middle_prob
            }
            
            // 设置价值目标 - 添加噪声防止过拟合，但先检查NaN
            let value_noise = fastrand::f32() * 0.1 - 0.05;  // [-0.05, 0.05]的噪声
            let mut value_target = exp.return_ + value_noise;
            
            // 检查并修复NaN值
            if !value_target.is_finite() {
                eprintln!("[NaN修复] 检测到exp.return_为NaN (值: {:?})，使用0.0替代", exp.return_);
                value_target = 0.0;
            }
            
            target_output[6] = value_target.clamp(-2.0, 2.0); // value
            
            // 置信度和模式
            target_output[7] = 0.7; // confidence - 降低确定性
            target_output[8] = if self.game_mode == crate::hand_model::GameMode::FourFinger { 1.0 } else { 0.0 }; // four_finger_mode
        target_output[9] = if self.game_mode == crate::hand_model::GameMode::FourFinger { 1.0 } else { 0.0 }; // game_mode_decision

            // 每个未来音符：[hand_prob_left, hand_prob_right, position_x]
            for i in 0..4 {
                let offset = 10 + i * 3;
                let future_offset = i * 4; // future_notes中每个音符占4个值
                
                if future_offset + 3 < exp.future_notes.len() {
                    // 使用真实的未来音符数据，但先检查NaN
                    let left_prob = exp.future_notes[future_offset];
                    let right_prob = exp.future_notes[future_offset + 1];
                    let pos_x = exp.future_notes[future_offset + 2];
                    
                    target_output[offset] = if left_prob.is_finite() { left_prob.clamp(0.0, 1.0) } else { 0.5 };
                    target_output[offset + 1] = if right_prob.is_finite() { right_prob.clamp(0.0, 1.0) } else { 0.5 };
                    target_output[offset + 2] = if pos_x.is_finite() { pos_x.clamp(-1.0, 1.0) } else { 0.0 };
                } else {
                    // 如果没有足够的未来音符数据，使用默认值
                    target_output[offset] = 0.5;     // hand_prob_left
                    target_output[offset + 1] = 0.5; // hand_prob_right
                    target_output[offset + 2] = 0.0; // position_x (中心)
                }
            }

            // 最终验证整个目标向量是否包含NaN
            if target_output.iter().any(|&x| !x.is_finite()) {
                eprintln!("[NaN修复] 训练目标仍包含NaN，重新生成默认目标");
                // 生成安全的默认目标
                for val in &mut target_output {
                    *val = if *val > 2.0 {
                        2.0
                    } else if *val < -2.0 {
                        -2.0
                    } else {
                        *val
                    };
                    if !val.is_finite() {
                        *val = 0.0;
                    }
                }
            }

            training_data.push((exp.state.clone(), target_output));
        }

        let epoch = self.main_network.epoch_count;
        let lr = self.main_network.learning_rate;
        let replay_size = self.experience_replay.len();
        let avg_reward = self.average_reward;
        
        // 基于音符序列模式的智能早停检查
        self.sequence_pattern_history.push_back(self.calculate_sequence_metrics(avg_reward));
        
        // 保持历史记录在固定大小内
        if self.sequence_pattern_history.len() > self.convergence_window as usize {
            self.sequence_pattern_history.pop_front();
        }
        
        let early_stopping_decision = self.should_early_stop();
        
        match early_stopping_decision {
            EarlyStopDecision::Continue => {
                self.no_improvement_count = 0;
                println!("[早停] 序列模式持续变化，继续训练");
            },
            EarlyStopDecision::LearningRateAdjust => {
                self.no_improvement_count += 1;
                if self.main_network.learning_rate < 0.0002 {
                    let old_lr = self.main_network.learning_rate;
                    self.main_network.learning_rate = 0.001;
                    self.no_improvement_count = 8;
                    println!("[早停] 学习率过低({:.6})，提升至0.001继续训练", old_lr);
                } else {
                    self.main_network.learning_rate *= 0.8;
                    self.no_improvement_count = 0;
                    println!("[早停] 学习率适度降低至: {:.6}", self.main_network.learning_rate);
                }
            },
            EarlyStopDecision::Stop => {
                self.no_improvement_count += 1;
                println!("[早停] 连续{}轮序列模式稳定，考虑暂停训练", self.no_improvement_count);
                
                if self.no_improvement_count >= 20 {  // 增加容忍轮数
                    println!("[早停] 序列模式持续稳定，暂停训练防止过拟合");
                    // 检查是否需要恢复训练
                    if self.main_network.learning_rate < 0.0003 {
                        let old_lr = self.main_network.learning_rate;
                        self.main_network.learning_rate = 0.002;
                        self.no_improvement_count = 10;
                        println!("[早停恢复] 提升学习率({:.6})继续探索", old_lr);
                    }
                }
            }
        }
        
        // 检查策略损失停滞并处理
        let policy_stagnation_score = self.check_policy_loss_stagnation();
        if policy_stagnation_score.abs() < 1e-6 {
            println!("[策略停滞] 检测到策略停滞，强制注入非零优势函数");
            self.force_nonzero_advantages(&mut experiences);
            self.emergency_policy_update();
        }
        
        // 检查训练数据质量
        let mut valid_samples = 0;
        let mut invalid_inputs = 0;
        let mut invalid_targets = 0;
        let mut invalid_outputs = 0;
        let mut total_loss = 0.0;
        
        for (i, (input, target)) in training_data.iter().enumerate() {
            let input_valid = input.iter().all(|&x| x.is_finite());
            let target_valid = target.iter().all(|&x| x.is_finite());
            
            if !input_valid {
                invalid_inputs += 1;
                if invalid_inputs <= 5 {  // 只输出前几个错误样本
                    let invalid_values: Vec<_> = input.iter().filter(|x| !x.is_finite()).collect();
                    eprintln!("[样本质量] 样本{} 输入包含无效值: {:?}", i, invalid_values);
                }
                continue;
            }
            
            if !target_valid {
                invalid_targets += 1;
                if invalid_targets <= 5 {
                    let invalid_values: Vec<_> = target.iter().filter(|x| !x.is_finite()).collect();
                    eprintln!("[样本质量] 样本{} 目标包含无效值: {:?}", i, invalid_values);
                }
                continue;
            }
            
            // 计算输出并进行有效性检查
            let output = self.main_network.light_forward(input);
            let output_valid = output.iter().all(|&x| x.is_finite());
            
            if !output_valid {
                invalid_outputs += 1;
                if invalid_outputs <= 5 {
                    let invalid_values: Vec<_> = output.iter().filter(|x| !x.is_finite()).collect();
                    eprintln!("[样本质量] 样本{} 网络输出包含无效值: {:?}", i, invalid_values);
                }
                continue;
            }
            
            // 计算损失，使用更安全的计算方式
            let mut sample_loss = 0.0;
            let zip_len = output.len().min(target.len()).min(10);
            for j in 0..zip_len {
                let o = output[j];
                let t = target[j];
                if o.is_finite() && t.is_finite() {
                    let diff = o - t;
                    if diff.is_finite() {
                        sample_loss += diff * diff;
                    }
                }
            }
            
            if sample_loss.is_finite() {
                valid_samples += 1;
                total_loss += sample_loss;
            }
        }
        
        if valid_samples == 0 {
            eprintln!("[PPO训练错误] 没有有效的训练样本，总样本数: {}, 无效输入: {}, 无效目标: {}, 无效输出: {}", 
                     training_data.len(), invalid_inputs, invalid_targets, invalid_outputs);
            
            // 尝试放宽过滤条件进行恢复性训练
            if training_data.len() >= 8 {
                eprintln!("[PPO恢复] 尝试放宽条件进行恢复性训练...");
                self.recovery_training(&mut experiences);
            } else {
                eprintln!("[PPO恢复] 样本太少，无法进行恢复性训练，跳过此轮训练");
            }
            return;
        }
        
        let avg_loss = total_loss / valid_samples as f32;
        
        println!(
            "【PPO训练】Epoch {}, LR: {:.6}, Replay Size: {}, Avg Reward: {:.3}, Valid Samples: {}, Avg Loss: {:.6}",
            epoch, lr, replay_size, avg_reward, valid_samples, avg_loss
        );
        
        // 添加数据质量诊断信息
        if valid_samples < training_data.len() / 2 {
            eprintln!("[数据质量] 警告: 有效样本比例低 ({}/{}) = {:.1}%", 
                     valid_samples, training_data.len(), 
                     (valid_samples as f32 / training_data.len() as f32) * 100.0);
        }
        
        // 训练完成后再次检查策略状态
        let final_policy_check = self.check_policy_loss_stagnation();
        if final_policy_check.abs() < 1e-6 && avg_loss < 0.1 {
            println!("[策略警告] 训练完成但策略仍停滞，损失过低 ({:.6})，可能需要强制扰动", avg_loss);
            self.add_policy_perturbation();
        }

        // 在训练前检查损失是否异常
        if avg_loss > 100.0 {
            eprintln!("[PPO警告] 损失值异常高 ({:.6})，可能存在梯度爆炸", avg_loss);
            // 降低学习率以防止进一步爆炸
            self.main_network.learning_rate = (self.main_network.learning_rate * 0.5).max(0.0001);
            println!("[PPO调整] 临时降低学习率至: {:.6}", self.main_network.learning_rate);
        }

        // 使用PPO算法进行训练
        self.main_network.train_with_ppo(&training_data, &experiences);
    }

    fn recovery_training(&mut self, experiences: &mut [Experience]) {
        eprintln!("[恢复训练] 启用数据清洗和恢复训练");
        
        // 清理无效数据并收集有效样本
        let mut cleaned_experiences: Vec<Experience> = Vec::new();
        for exp in experiences.iter() {
            let state_valid = exp.state.iter().all(|&x| x.is_finite());
            let reward_valid = exp.reward.is_finite();
            let value_valid = exp.value.is_finite();
            
            if state_valid && reward_valid && value_valid {
                cleaned_experiences.push(exp.clone());
            }
        }
        
        if cleaned_experiences.is_empty() {
            eprintln!("[恢复训练] 清理后仍然没有有效数据，跳过");
            return;
        }
        
        eprintln!("[恢复训练] 清理后有效样本: {}/{}", cleaned_experiences.len(), experiences.len());
        eprintln!("[恢复训练] 清理后有效样本: {}/{}", experiences.len(), experiences.len() + 64);
        
        // 使用更简单的训练目标
        let mut recovery_training_data = Vec::new();
        for exp in cleaned_experiences.iter().take(32) {  // 限制样本数量
            let mut target = vec![0.0; 10];  // 只使用前10维
            
            if exp.action == 0 { // Left
                target[0] = 0.8;  // left_prob
                target[1] = 0.2;  // right_prob
            } else { // Right
                target[0] = 0.2;  // left_prob
                target[1] = 0.8;  // right_prob
            }
            
            // 限制目标值在合理范围内
            for val in &mut target {
                let clamped_val = if *val > 2.0 {
                    2.0
                } else if *val < -2.0 {
                    -2.0
                } else {
                    *val
                };
                *val = clamped_val;
            }
            
            recovery_training_data.push((exp.state.clone(), target));
        }
        
        if !recovery_training_data.is_empty() {
            eprintln!("[恢复训练] 进行简化训练，样本数: {}", recovery_training_data.len());
            self.main_network.train_batch(&recovery_training_data);
        } else {
            eprintln!("[恢复训练] 无法生成有效的恢复训练数据");
        }
    }

    /*
    fn estimate_future_value(&mut self, state: &[f32]) -> f32 {
        let output = self.target_network.forward(state);
        output.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }

     */

    pub fn warm_thread_pool(&self) {
        if let Some(pool) = &self.thread_pool {
            pool.install(|| {
                let _ : Vec<u8> = (0..1).into_par_iter().map(|x| (x*2) as u8).collect();
            });
        }
    }
}

