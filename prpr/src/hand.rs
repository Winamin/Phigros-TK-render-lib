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
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::thread;
use std::time::{Duration, Instant};
use wgpu;
use wgpu::util::DeviceExt;
use crossbeam_channel::{unbounded, Receiver as CbReceiver, Sender as CbSender, TryRecvError, TrySendError};
use rayon::ThreadPool;
use std::hash::{Hash, Hasher};
use futures_intrusive::channel::shared::oneshot_channel;

type StdHashMap<K, V> = HashMap<K, V>;
type MatmulKey = (usize, usize, usize, usize, usize, usize); // (layer_idx, input_ptr, output_ptr, input_size, output_size, batch_size)
type ActivationKey = (usize, usize, usize); // (layer_idx, buffer_ptr, batch_size)

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
    notes.len().hash(&mut hasher);
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

fn validate_notes_consistency(original: &[Note], updated: &[Note]) -> bool {
    if original.len() != updated.len() {
        return false;
    }
    for (orig, upd) in original.iter().zip(updated.iter()) {
        if (orig.time - upd.time).abs() > 0.001 {
            return false;
        }
        if std::mem::discriminant(&orig.kind) != std::mem::discriminant(&upd.kind) {
            return false;
        }
        let orig_x = orig.object.translation.0.now();
        let orig_y = orig.object.translation.1.now();
        let upd_x = upd.object.translation.0.now();
        let upd_y = upd.object.translation.1.now();
        let pos_diff = ((orig_x - upd_x).powi(2) + (orig_y - upd_y).powi(2)).sqrt();
        if pos_diff > 100.0 {
            return false;
        }
    }
    true
}

fn match_and_merge_notes(original: &mut [Note], updated: &[Note]) -> bool {
    const TIME_THRESHOLD_SEC: f32 = 0.05; // 50 ms
    const POS_THRESHOLD: f32 = 20.0; // 20

    let mut used = vec![false; updated.len()];

    for orig in original.iter_mut() {
        let mut best_idx: Option<usize> = None;
        let mut best_score = std::f64::INFINITY;
        for (i, upd) in updated.iter().enumerate() {
            if used[i] { continue; }
            if std::mem::discriminant(&orig.kind) != std::mem::discriminant(&upd.kind) {
                continue;
            }
            let dt = (orig.time - upd.time).abs();
            if dt > TIME_THRESHOLD_SEC {
                continue;
            }
            let ox = orig.object.translation.0.now();
            let oy = orig.object.translation.1.now();
            let ux = upd.object.translation.0.now();
            let uy = upd.object.translation.1.now();
            let dist = ((ox - ux).powi(2) + (oy - uy).powi(2)).sqrt();
            if dist > POS_THRESHOLD {
                continue;
            }
            let score = (dt as f64) * 1000.0 + (dist as f64);
            if score < best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }
        if let Some(i) = best_idx {
            orig.hand = updated[i].hand;
            used[i] = true;
        } else {
            eprintln!(
                "match_and_merge_notes: failed to find match for original note time={} kind={:?}",
                orig.time,
                std::mem::discriminant(&orig.kind)
            );
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


        std::thread::spawn(move || {
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

                        if let Err(e) = tx_resp.send(resp) {
                            //eprintln!("Failed to send AI response for id={} : {:?}", req.id, e);
                        } else {
                            //println!("AI worker: sent response for id={}", req.id);
                        }

                        TOTAL_TOKENS_USED.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(panic_info) => {
                        //eprintln!("AI analysis panicked for request id={}: {:?}", req.id, panic_info);
                    }
                }
            }

            //println!("AI worker: exiting receive loop");
        });

        std::thread::spawn(|| {
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
    let mut line_states_guard = match line_states.lock() {
        Ok(g) => g,
        Err(poisoned) => {
            eprintln!("Line states mutex poisoned, recovering...");
            poisoned.into_inner()
        }
    };

    let line_state = line_states_guard.entry(line_id).or_default();

    if let Some(map) = LINE_RESP_QUEUES.get() {
        // 先从全局队列中弹出最多 N 条到本地数组，减少锁占用时间
        let mut responses = Vec::new();
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
        } // 在这里释放 LINE_RESP_QUEUES 的锁

        for resp in responses {
            if resp.line_id != line_id {
                eprintln!("(dispatcher) unexpected line mismatch: resp.line_id={}, expected={}", resp.line_id, line_id);
                continue;
            }
            let pending_entry = line_state.pending_requests.remove(&resp.id);
            if pending_entry.is_none() {
                eprintln!("Received unexpected or already-handled response id={} for line={}", resp.id, resp.line_id);
                continue;
            }
            let (_req_ts, req_version) = pending_entry.unwrap();
            if resp.version != req_version {
                eprintln!("Discarding response id={} due to version mismatch (resp.version={} != req_version={})",
                          resp.id, resp.version, req_version);
                continue;
            }
            let expected_checksum = calculate_checksum(&resp.notes);
            if expected_checksum != resp.checksum {
                eprintln!("Checksum mismatch for response id={}, discarding", resp.id);
                continue;
            }
            if !match_and_merge_notes(notes, &resp.notes) {
                eprintln!("Response matching/merge failed for id={}, discarding", resp.id);
                continue;
            }
            line_state.current_version = resp.version;
            line_state.last_full_update = now;
        }
    }


    let should_light_update = now.duration_since(line_state.last_light_update) >= Duration::from_millis(LIGHT_UPDATE_INTERVAL_MS);
    if should_light_update {
        line_state.last_light_update = now;
        let ai_system = AI_SYSTEM.get_or_init(|| Mutex::new(PhiTKAdvancedAI::load_or_create("phitk_ai_model.bin", rotation)));
        if let Ok(mut ai) = ai_system.lock() {
            if ai.rotation != rotation {
                ai.rotation = rotation;
                ai.update_hand_positions();
            }
            ai.light_update_hand_states(notes);
        } else {
            //eprintln!("Failed to lock AI_SYSTEM for light update");
        }
    }

    let should_full_update = now.duration_since(line_state.last_full_update) >= Duration::from_millis(FULL_UPDATE_INTERVAL_MS);
    if should_full_update {
        const MAX_PENDING_REQUESTS: usize = 3;
        if line_state.pending_requests.len() >= MAX_PENDING_REQUESTS {
            //eprintln!("Too many pending requests for line {}, skipping new request", line_id);
        } else {
            let version = VERSION_COUNTER.fetch_add(1, Ordering::Relaxed);
            let request_id = REQ_COUNTER.fetch_add(1, Ordering::Relaxed);

            line_state.pending_requests.insert(request_id, (now, version));

            let cfg_arc = Arc::new(config.clone());
            let bpm_arc = Arc::new(bpm_list.clone());
            let notes_snapshot = notes.to_vec();

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
                if let Err(e) = tx.send(req) {
                    //eprintln!("Failed to send AI request id={} : {:?}", request_id, e);
                    line_state.pending_requests.remove(&request_id);
                }
            } else {
                //eprintln!("AI_REQ_TX not initialized when attempting to send request");
                line_state.pending_requests.remove(&request_id);
            }
        }
    }
    drop(line_states_guard);
}

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
    #[serde(skip)]
    #[serde(default)]
    thread_count: usize,
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
    assigned_finger: Option<Finger>,
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
    #[serde(skip)]
    max_grad_norm: f32,
    #[serde(skip)]
    weight_decay: f32,
    #[serde(default)]
    last_loss: f32,
    #[serde(default)]
    bad_epochs: usize,
    epoch_count: u64,
    //GPU >
    #[serde(skip)]device: Option<wgpu::Device>,
    #[serde(skip)]queue: Option<wgpu::Queue>,
    #[serde(skip)]matmul_pipeline: Option<wgpu::ComputePipeline>,
    #[serde(skip)]matmul_bind_group_layout: Option<wgpu::BindGroupLayout>,
    #[serde(skip)]activation_pipelines: StdHashMap<ActivationFunction, wgpu::ComputePipeline>,
    #[serde(
        skip
    )]activation_bind_group_layouts: StdHashMap<ActivationFunction, wgpu::BindGroupLayout>,
    #[serde(skip)]gpu_initialized: bool,
    #[serde(skip)]batch_size_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]initialization_attempted: bool,
    #[serde(skip)]initialization_failed: bool,
    #[serde(skip)]
    matmul_bind_groups: StdHashMap<MatmulKey, (u64, wgpu::BindGroup)>,
    #[serde(skip)]
    activation_bind_groups: StdHashMap<ActivationKey, (u64, wgpu::BindGroup)>,
    #[serde(skip)]
    buffer_pool: Option<BufferPool>,
    #[serde(skip)]
    frame_counter: u64,
    #[serde(skip)]
    pre_allocated_input_buffers: Vec<Option<(wgpu::Buffer, usize, wgpu::BufferUsages)>>,
    #[serde(skip)]
    pre_allocated_output_buffers: Vec<Option<(wgpu::Buffer, usize, wgpu::BufferUsages)>>,
    #[serde(skip)]
    pre_allocated_staging_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    last_batch_size: usize,
    #[serde(skip)]
    buffer_cleanup_interval: Option<Instant>,
    #[serde(skip)]
    last_used_buffers: StdHashMap<wgpu::Buffer, (Option<Instant>, usize, wgpu::BufferUsages)>,
    #[serde(skip)]
    reusable_buffers: StdHashMap<(usize, wgpu::BufferUsages), Vec<wgpu::Buffer>>,
    #[serde(skip)]
    loss_pipeline: Option<wgpu::ComputePipeline>,
    #[serde(skip)]
    backward_pipelines: StdHashMap<ActivationFunction, wgpu::ComputePipeline>,
    #[serde(skip)]
    weight_update_pipeline: Option<wgpu::ComputePipeline>,
    #[serde(skip)]
    weight_gradient_buffers: Vec<wgpu::Buffer>,
    #[serde(skip)]
    bias_gradient_buffers: Vec<wgpu::Buffer>,
    #[serde(skip)]
    staging_buffers: Vec<wgpu::Buffer>,
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
    #[serde(skip)]
    inputs: Vec<f32>,
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
    BSiLU,
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
    #[serde(skip)]
    feature_extraction_bind_group_layout: Option<wgpu::BindGroupLayout>,
    #[serde(skip)]
    feature_extraction_pipeline: Option<wgpu::ComputePipeline>,
    #[serde(skip)]
    note_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    bpm_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    params_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    feature_buffer: Option<wgpu::Buffer>,
    #[serde(skip)]
    device: Option<wgpu::Device>, // 添加 device 字段
    #[serde(skip)]
    feature_count: usize,         // 添加 feature_count 字段
    #[serde(skip)]
    context_window: u32,
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

struct BufferPool {
    buffers: Vec<(wgpu::Buffer, usize, wgpu::BufferUsages)>,
    device: Arc<wgpu::Device>,
    max_pool_size: usize,
}

impl BufferPool {
    fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            buffers: Vec::new(),
            device,
            max_pool_size: 100,
        }
    }

    fn get_buffer(&mut self, size: usize, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        for i in 0..self.buffers.len() {
            let (ref buffer, buffer_size, buffer_usage) = self.buffers[i];
            if buffer_size >= size && buffer_usage.contains(usage) {
                return self.buffers.remove(i).0;
            }
        }
        let actual_size = (size * 3 / 2).max(32768);
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: actual_size as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    fn return_buffer(&mut self, buffer: wgpu::Buffer, size: usize, usage: wgpu::BufferUsages) {
        if self.buffers.len() < self.max_pool_size {
            self.buffers.push((buffer, size, usage));
        }
    }

    fn clear(&mut self) {
        self.buffers.clear();
    }
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

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("buffers", &self.buffers.len())
            .field("device", &"Arc<Device>")
            .finish()
    }
}

impl Clone for BufferPool {
    fn clone(&self) -> Self {
        Self {
            buffers: Vec::new(),
            device: self.device.clone(),
            max_pool_size: self.max_pool_size,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuNote {
    position_x: f32,
    position_y: f32,
    time: f32,
    kind: u32,
    judge: u32,
    duration: f32,
    difficulty: f32,
}

impl From<&ProcessedNote> for GpuNote {
    fn from(note: &ProcessedNote) -> Self {
        GpuNote {
            position_x: note.position.x,
            position_y: note.position.y,
            time: note.time,
            kind: u32::from(note.kind.clone()),
            judge: u32::from(note.judge.clone()),
            duration: note.duration,
            difficulty: note.difficulty,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuBpmPoint {
    time: f32,
    bpm: f32,
}

impl From<&(f32, f32, f32)> for GpuBpmPoint {
    fn from(point: &(f32, f32, f32)) -> Self {
        // point 是 (beats, time, bpm)
        Self {
            time: point.1,  // time
            bpm: point.2,   // bpm
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct GpuInputParams {
    num_notes: u32,
    num_bpm_points: u32,
    context_window: u32,
    total_features: u32,
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
        let randomness = (fastrand::f32() - 0.5) * 0.1;
        self.performance_score = (
            recent_success_rate * 0.4 +
                self.confidence * 0.35 +
                (1.0 - self.fatigue) * 0.25 +
                randomness
        ).clamp(0.1, 0.95);

        self.position = new_pos;
        self.last_time = time;
    }

    fn calculate_assignment_score(&self, target_pos: Vector2, time: f32, note_difficulty: f32,
                                  current_time: f32, note_kind: &NoteKind, note_duration: f32) -> f32 {
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
        let position_weight = match self.finger.to_hand() {
            Hand::Left => {
                if target_pos.x < -0.25 {
                    0.35 + ((-0.25 - target_pos.x) * 0.5).min(0.15)
                } else if target_pos.x > 0.0 {
                    -0.15 - (target_pos.x * 0.8).min(0.4)
                } else {
                    // 中性区域的线性过渡
                    0.35 * ((-target_pos.x) / 0.25)
                }
            }
            Hand::Right => {
                if target_pos.x > 0.15 {
                    0.4 + ((target_pos.x - 0.15) * 0.6).min(0.2)
                } else if target_pos.x < -0.15 {
                    -0.25 - ((-target_pos.x - 0.15) * 0.9).min(0.35)
                } else {
                    // 中性区域
                    0.1 + (target_pos.x / 0.15) * 0.3
                }
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
        score *= (1.0 - fatigue_penalty);

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

        let randomness = (fastrand::f32() - 0.5) * 0.05;
        score += randomness;

        score.clamp(0.0, 1.0)
    }
    fn update_busy_status(&mut self, current_time: f32) {
        if current_time >= self.busy_until {
            self.is_busy = false;
            self.busy_until = -1.0;
        }
    }
}

impl DeepNeuralNetwork {
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

        if !self.learning_rate.is_finite() { self.learning_rate = 0.001; }
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
            ActivationFunction::GELU => {
                0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh())
            }
            ActivationFunction::BSiLU => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                let silu = x * sigmoid;
                silu / (1.0 + silu.abs()) // B-SiLU: bounded between (-1, 1)
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

    fn new() -> Self {
        let mut network = Self {
            layers: Vec::new(),
            learning_rate: 0.001,
            momentum: 0.9,
            dropout_rate: 0.1,
            batch_size: 16384,
            max_grad_norm: 10.0,
            weight_decay: 0.0001,
            last_loss: f32::INFINITY,
            bad_epochs: 0,
            epoch_count: 0,
            device: None,
            queue: None,
            matmul_pipeline: None,
            matmul_bind_group_layout: None,
            activation_pipelines: StdHashMap::new(),
            activation_bind_group_layouts: StdHashMap::new(),
            gpu_initialized: false,
            batch_size_buffer: None,
            initialization_attempted: false,
            initialization_failed: false,
            matmul_bind_groups: StdHashMap::new(),
            activation_bind_groups: StdHashMap::new(),
            //buffer_pool: Some(BufferPool::default()),
            buffer_pool: None,
            frame_counter: 0,
            pre_allocated_input_buffers: Vec::new(),
            pre_allocated_output_buffers: Vec::new(),
            pre_allocated_staging_buffer: None,
            last_batch_size: 0,
            reusable_buffers: StdHashMap::new(),
            last_used_buffers: StdHashMap::new(),
            buffer_cleanup_interval: Some(Instant::now()),
            loss_pipeline: None,
            backward_pipelines: StdHashMap::new(),
            weight_update_pipeline: None,
            weight_gradient_buffers: Vec::new(),
            bias_gradient_buffers: Vec::new(),
            staging_buffers: Vec::new(),
        };

        network.build_architecture();
        network.init_gpu_sync();
        if network.gpu_initialized {
            network.pre_allocate_buffers(64, 256);
        }
        network
    }

    fn sync_weights_to_gpu(&self) {
        // 健全性检查
        if self.device.is_none() || self.queue.is_none() {
            eprintln!("[GPU Sync] 无法同步权重：device 或 queue 未初始化");
            return;
        }
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        for layer in &self.layers {
            // 检查该层的 GPU 缓冲区是否存在
            if let (Some(weights_buffer), Some(biases_buffer)) = (&layer.weights_buffer, &layer.biases_buffer) {
                // 将 CPU 内存中的权重和偏置数据同步到 GPU 缓冲区
                let weights_flat = layer.weights.iter().flatten().cloned().collect::<Vec<f32>>();
                queue.write_buffer(weights_buffer, 0, bytemuck::cast_slice(&weights_flat));

                queue.write_buffer(biases_buffer, 0, bytemuck::cast_slice(&layer.biases));
            } else {
                // 如果某层的 GPU 缓冲区不存在，可以在这里选择初始化它，或者跳过
                // 为简单起见，我们选择跳过，并打印警告
                eprintln!("[GPU Sync] 跳过层同步：weights_buffer 或 biases_buffer 为 None");
            }
        }

        // 同步完成后，提交所有写入命令
        //queue.submit(std::iter::empty::<wgpu::CommandEncoder>()); // 提交一个空的命令批次，确保前面的 write_buffer 被执行
        device.poll(wgpu::PollType::Wait); // 等待 GPU 完成同步
    }

    fn pre_allocate_buffers(&mut self, max_batch_size: usize, input_size: usize) {
        if !self.gpu_initialized {
            return;
        }

        let device = self.device.as_ref().unwrap();

        // 清理旧的预分配缓冲区
        self.pre_allocated_input_buffers.clear();
        self.pre_allocated_output_buffers.clear();

        let mut current_size = input_size;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let output_size = layer.weights.len();

            // 输入缓冲区：只读存储，并可被拷贝数据
            let input_buffer_size = current_size * max_batch_size * std::mem::size_of::<f32>();
            let input_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pre-allocated Input Buffer Layer {}", layer_idx)),
                size: input_buffer_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.pre_allocated_input_buffers.push(Some((input_buffer, input_buffer_size, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)));

            // 输出缓冲区：读写存储，并可被拷贝数据
            let output_buffer_size = output_size * max_batch_size * std::mem::size_of::<f32>();
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pre-allocated Output Buffer Layer {}", layer_idx)),
                size: output_buffer_size as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.pre_allocated_output_buffers.push(Some((output_buffer, output_buffer_size,wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST)));

            current_size = output_size;
        }

        // 预分配staging buffer
        let final_output_size = self.layers.last().unwrap().weights.len();
        let staging_buffer_size = final_output_size * max_batch_size * std::mem::size_of::<f32>();
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pre-allocated Staging Buffer"),
            size: staging_buffer_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.pre_allocated_staging_buffer = Some(staging_buffer);

        self.last_batch_size = max_batch_size;
    }

    fn cleanup_expired_buffers(&mut self) {
        let now = Instant::now();
        let mut to_remove = Vec::new();

        for (buffer, (last_used, size, usage)) in &self.last_used_buffers {
            if let Some(last_used) = last_used {
                if now.duration_since(*last_used) > Duration::from_secs(60) {
                    to_remove.push(buffer.clone());
                }
            }
        }

        for buffer in to_remove {
            if let Some((last_used, size, usage)) = self.last_used_buffers.remove(&buffer) {
                let key = (size, usage);
                self.reusable_buffers.entry(key).or_insert_with(Vec::new).push(buffer);
            }
        }
    }


    fn get_or_create_buffer(&mut self, size: usize, usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        // 定期清理过期缓冲区
        let now = Instant::now();
        if now.duration_since(self.buffer_cleanup_interval.expect("buffer_cleanup_interval was None")) > Duration::from_secs(30) {
            self.cleanup_expired_buffers();
            self.buffer_cleanup_interval = Some(now);
        }

        // 首先尝试从可复用缓冲区中获取（用途必须完全匹配）
        let key = (size, usage);
        if let Some(buffers) = self.reusable_buffers.get_mut(&key) {
            if let Some(buffer) = buffers.pop() {
                self.last_used_buffers.insert(buffer.clone(), (Some(now), size, usage));
                return buffer;
            }
        }

        // 然后检查预分配缓冲区（用途必须完全匹配）
        for buffer_option in &self.pre_allocated_input_buffers {
            if let Some((ref prealloc_buffer, prealloc_size, prealloc_usage)) = buffer_option {
                if *prealloc_size >= size && *prealloc_usage == usage {
                    self.last_used_buffers.insert(prealloc_buffer.clone(), (Some(now), size, usage));
                    return prealloc_buffer.clone();
                }
            }
        }
        for buffer_option in &self.pre_allocated_output_buffers {
            if let Some((ref prealloc_buffer, prealloc_size, prealloc_usage)) = buffer_option {
                if *prealloc_size >= size && *prealloc_usage == usage {
                    self.last_used_buffers.insert(prealloc_buffer.clone(), (Some(now), size, usage));
                    return prealloc_buffer.clone();
                }
            }
        }

        // 如果没有找到，创建新的缓冲区
        let buffer = self.device.as_ref().unwrap().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size as u64,
            usage,
            mapped_at_creation: false,
        });

        self.last_used_buffers.insert(buffer.clone(), (Some(now), size, usage));
        buffer
    }


    fn pre_allocate_common_buffers(&mut self) {
        if let Some(ref mut pool) = self.buffer_pool {
            // 预分配常用大小的缓冲区
            let common_sizes = [1024, 2048, 4096, 8192, 16384, 32768];
            for &size in &common_sizes {
                for &usage in &[
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                ] {
                    let buffer = self.device.as_ref().unwrap().create_buffer(&wgpu::BufferDescriptor {
                        label: Some("Pre-allocated buffer"),
                        size: size as u64,
                        usage,
                        mapped_at_creation: false,
                    });
                    pool.return_buffer(buffer, size, usage);
                }
            }
        }
    }

    fn return_buffer_to_pool(&mut self, buffer: wgpu::Buffer, size: usize, usage: wgpu::BufferUsages) {
        let now = Instant::now();
        self.last_used_buffers.insert(buffer.clone(), (Some(now), size, usage));
        let key = (size, usage);
        self.reusable_buffers.entry(key).or_insert_with(Vec::new).push(buffer);
    }

    fn get_matmul_bind_group(
        &mut self,
        layer_idx: usize,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        input_size: usize,
        output_size: usize,
        batch_size: usize
    ) -> wgpu::BindGroup {
        let key = (
            layer_idx,
            input_buffer as *const _ as usize,
            output_buffer as *const _ as usize,
            input_size,
            output_size,
            batch_size
        );

        self.update_batch_size(batch_size);

        use std::collections::hash_map::Entry;

        match self.matmul_bind_groups.entry(key) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().0 = self.frame_counter;
                let bind_group = entry.get().1.clone();
                bind_group
            }
            Entry::Vacant(entry) => {
                let device = self.device.as_ref().unwrap();
                let bind_group_layout = self.matmul_bind_group_layout.as_ref().unwrap();

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout: bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: input_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.layers[layer_idx].weights_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: output_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.layers[layer_idx].biases_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: self.batch_size_buffer.as_ref().unwrap().as_entire_binding(),
                        },
                    ],
                    label: Some(&format!("Matmul Bind Group Layer {}", layer_idx)),
                });

                let cloned_bind_group = bind_group.clone();
                entry.insert((self.frame_counter, bind_group));
                cloned_bind_group
            }
        }
    }

    fn get_activation_bind_group(
        &mut self,
        layer_idx: usize,
        buffer: &wgpu::Buffer,
        batch_size: usize
    ) -> wgpu::BindGroup {
        let layer = &self.layers[layer_idx];
        let key = (
            layer_idx,
            buffer as *const _ as usize,
            batch_size
        );

        self.update_batch_size(batch_size);

        use std::collections::hash_map::Entry;

        match self.activation_bind_groups.entry(key) {
            Entry::Occupied(mut entry) => {
                entry.get_mut().0 = self.frame_counter;
                let bind_group = entry.get().1.clone();
                bind_group
            }
            Entry::Vacant(entry) => {
                let device = self.device.as_ref().unwrap();
                let layout = self.activation_bind_group_layouts
                    .get(&layer.activation_func)
                    .unwrap();

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer.as_entire_binding(),
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
                    label: Some(&format!("Activation Bind Group Layer {}", layer_idx)),
                });

                let cloned_bind_group = bind_group.clone();
                entry.insert((self.frame_counter, bind_group));
                cloned_bind_group
            }
        }
    }

    fn cleanup_old_bind_groups(&mut self, max_age: u64) {
        let current_frame = self.frame_counter;
        self.matmul_bind_groups.retain(|_, (last_used, _)| {
            current_frame - *last_used <= max_age
        });
        self.activation_bind_groups.retain(|_, (last_used, _)| {
            current_frame - *last_used <= max_age
        });
        self.frame_counter += 1;
    }

    pub fn init_gpu_sync(&mut self) {
        let device_missing = self.device.is_none();
        let queue_missing = self.queue.is_none();
        let pipeline_missing = self.matmul_pipeline.is_none() || self.activation_pipelines.is_empty();

        // 检测不一致状态：gpu_initialized为true但资源缺失
        if self.gpu_initialized && (device_missing || queue_missing || pipeline_missing) {
            println!("[GPU修复] 检测到不一致状态 (gpu_initialized=true但资源缺失)，强制重新初始化...");
            // 重置状态以便重新初始化
            self.gpu_initialized = false;
            self.initialization_attempted = false;
            self.initialization_failed = false;
            // 清理可能部分初始化的资源
            self.device = None;
            self.queue = None;
            self.matmul_pipeline = None;
            self.activation_pipelines.clear();
            self.activation_bind_group_layouts.clear();
            self.buffer_pool = None;
        }

        // 如果之前尝试初始化但失败了，不再重试
        if self.initialization_attempted && self.initialization_failed {
            return;
        }

        // 尝试初始化
        if !self.initialization_attempted {
            self.initialization_attempted = true;
            let rt = tokio::runtime::Runtime::new().expect("Failed to create Tokio runtime for GPU initialization");
            let init_result = rt.block_on(async { self.init_gpu().await });

            if init_result {
                self.gpu_initialized = true;
                self.initialization_failed = false;
                println!("[GPU] GPU 初始化成功");
            } else {
                self.initialization_failed = true;
                println!("[GPU] GPU 初始化失败，将回退到 CPU 模式");
                // 不 panic，允许系统继续运行但使用 CPU
            }
        }

    }

    pub async fn init_gpu(&mut self) -> bool {
        if self.gpu_initialized {
            return true;
        }

        let instance_desc = wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        };
        let instance = wgpu::Instance::new(&instance_desc);

        // --- 修改 1: 为 request_adapter 添加详细错误日志 ---
        let adapter = match instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }).await {
            Ok(adapter) => adapter,
            Err(e) => {
                eprintln!("[GPU Init Error] Failed to request GPU adapter: {:?}", e);
                eprintln!("[GPU Init Error] This usually means no compatible GPU was found, or the graphics driver is missing/outdated.");
                return false;
            }
        };

        // --- 修改 2: 为 request_device 添加详细错误日志 ---
        let (device, queue) = match adapter.request_device(&wgpu::DeviceDescriptor::default()).await {
            Ok((d, q)) => (d, q),
            Err(e) => {
                eprintln!("[GPU Init Error] Failed to request GPU device: {:?}", e);
                eprintln!("[GPU Init Error] This usually means the GPU does not support the required features, or is out of memory.");
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
            source: wgpu::ShaderSource::Wgsl(MATMUL_WGSL.into()),
        });

        let matmul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&matmul_pipeline_layout),
            module: &matmul_shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let mut activation_pipelines = StdHashMap::new();
        let mut activation_bind_group_layouts = StdHashMap::new();

        for func in [
            ActivationFunction::ReLU,
            ActivationFunction::Sigmoid,
            ActivationFunction::Tanh,
            ActivationFunction::Swish,
            ActivationFunction::GELU,
            ActivationFunction::BSiLU
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
                label: Some(&format!("{:?} Activation Shader", func)),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{:?} Activation Pipeline", func)),
                layout: Some(&activation_pipeline_layout),
                module: &activation_shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });
            activation_pipelines.insert(func.clone(), pipeline);
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
                size: (layer.activations.len() * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        }

        self.device = Some(device.clone());
        self.queue = Some(queue);
        self.matmul_pipeline = Some(matmul_pipeline);
        self.matmul_bind_group_layout = Some(matmul_bind_group_layout);
        self.activation_pipelines = activation_pipelines;
        self.activation_bind_group_layouts = activation_bind_group_layouts;
        self.buffer_pool = Some(BufferPool::new(Arc::new(device)));
        self.batch_size_buffer = Some(batch_size_buffer);
        self.reusable_buffers = StdHashMap::new();
        self.last_used_buffers = StdHashMap::new();
        self.buffer_cleanup_interval = Some(Instant::now());

        true
        //println!("Success");
    }

    fn get_activation_shader(&self, func: &ActivationFunction) -> String {
        let fn_body = match func {
            ActivationFunction::ReLU => "return max(val, 0.0);",
            ActivationFunction::Sigmoid => "return 1.0 / (1.0 + exp(-val));",
            ActivationFunction::Tanh => "return tanh(val);",
            ActivationFunction::Swish => "return val * (1.0 / (1.0 + exp(-val)));",
            ActivationFunction::GELU => "return 0.5 * val * (1.0 + tanh(val * 0.7978845608 * (1.0 + 0.044715 * val * val)));",
            ActivationFunction::BSiLU => {
                r#"let sigmoid = 1.0 / (1.0 + exp(-val));
let silu = val * sigmoid;
return silu / (1.0 + abs(silu));"#
            }
        }.to_string();

        let shader = format!(r#"
struct BatchSize {{
    size: u32,
}}

@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> batch_size: BatchSize;

fn activate(val: f32) -> f32 {{
    {fn_body}
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let idx: u32 = id.x;
    if (idx >= arrayLength(&data)) {{
        return;
    }}
    data[idx] = activate(data[idx]);
}}
"#,
                             fn_body = fn_body);
        //println!("Activation shader:\\n{}", shader);

        shader
    }

    fn calculate_loss(&self) -> f32 {
        // TODO：这是一个简单的实现，返回最近一次记录的损失
        // 您可以根据需要实现更复杂的逻辑
        return self.last_loss;
    }

    pub fn gpu_train_batch(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) -> f32 {
        // 强制使用64的批量大小
        const TARGET_BATCH_SIZE: usize = 64;

        // 确保GPU已初始化
        if !self.gpu_initialized {
            println!("[GPU训练] GPU未初始化，回退到CPU训练");
            return self.calculate_loss();
        }

        // 检查必要的GPU资源
        if self.device.is_none() || self.queue.is_none() || self.matmul_pipeline.is_none() {
            println!("[GPU训练] GPU资源不完整，尝试重新初始化...");
            self.init_gpu_sync();

            // 再次检查
            if !self.gpu_initialized || self.device.is_none() || self.queue.is_none() || self.matmul_pipeline.is_none() {
                println!("[GPU训练] GPU重新初始化失败，回退到CPU训练");
                self.train(training_data);
                let loss = self.calculate_loss(); // 或其他计算 loss 的方法
                return loss; // 直接返回 loss 值
            }
        }

        println!("[GPU训练] 开始GPU训练，强制批量大小: {}", TARGET_BATCH_SIZE);

        // 准备输入和目标数据
        let mut inputs: Vec<&[f32]> = Vec::with_capacity(TARGET_BATCH_SIZE);
        let mut targets: Vec<&[f32]> = Vec::with_capacity(TARGET_BATCH_SIZE);

        // 填充到64个样本（不足则重复）
        for i in 0..TARGET_BATCH_SIZE {
            let idx = i % training_data.len();
            inputs.push(&training_data[idx].0);
            targets.push(&training_data[idx].1);
        }

        // 预分配GPU缓冲区
        let input_size = inputs[0].len();
        let output_size = targets[0].len();
        self.pre_allocate_buffers(TARGET_BATCH_SIZE, input_size);

        // 执行前向传播
        let outputs = self.gpu_forward_batch(&inputs, TARGET_BATCH_SIZE);

        // 计算损失
        let loss = self.gpu_calculate_loss(&outputs, &targets, TARGET_BATCH_SIZE);

        // 执行反向传播
        self.gpu_backward(&inputs, &outputs, &targets, TARGET_BATCH_SIZE);

        // 应用梯度
        self.gpu_apply_gradients(TARGET_BATCH_SIZE as f32);

        // 更新学习率
        self.adapt_learning_rate(loss);
        self.epoch_count += 1;

        // 每个epoch都打印状态
        let (min_weight, max_weight) = self.get_weight_range();
        println!("[GPU训练状态] Epoch: {}, Loss: {:.6}, LR: {:.6}, 权重范围 [{:.4}, {:.4}]",
                 self.epoch_count, loss, self.learning_rate, min_weight, max_weight);

        loss
    }

    fn train_with_loss(&mut self, training_data: &[(Vec<f32>, Vec<f32>)], loss_out: &mut f32) {
        let mut total_loss = 0.0;
        let output_size = training_data[0].1.len(); // 假设所有目标长度相同
        for (input, target) in training_data {
            let output = self.forward(input);
            for i in 0..output.len() {
                let diff = output[i] - target[i];
                total_loss += diff * diff;
            }
            // 初始化梯度累积结构
            let mut total_gradients = vec![vec![vec![0.0; input.len()]; output.len()]; self.layers.len()];
            let mut total_bias_gradients = vec![vec![0.0; output.len()]; self.layers.len()];
            self.backward(&output, target, &mut total_gradients, &mut total_bias_gradients);
            self.apply_gradients(&total_gradients, &total_bias_gradients, 1);
        }
        *loss_out = total_loss / (training_data.len() * output_size) as f32;
    }

    /// 在GPU上计算损失（均方误差）
    fn gpu_calculate_loss(&mut self, outputs: &[Vec<f32>], targets: &[&[f32]], batch_size: usize) -> f32 {
        if !self.gpu_initialized {
            // CPU回退
            let mut total_loss = 0.0;
            for i in 0..batch_size {
                for j in 0..outputs[i].len() {
                    let diff = outputs[i][j] - targets[i][j];
                    total_loss += diff * diff;
                }
            }
            return total_loss / (batch_size * outputs[0].len()) as f32;
        }

        // 准备损失计算
        let output_size = outputs[0].len();
        let total_elements = batch_size * output_size;

        // 创建损失缓冲区
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        let loss_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loss Buffer"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // 创建损失计算管线（如果不存在）
        let loss_pipeline = match self.loss_pipeline {
            Some(ref pipeline) => pipeline.clone(),
            None => {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Loss Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("loss.wgsl").into()),
                });

                let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Loss Pipeline Layout"),
                    bind_group_layouts: &[
                        &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                            ],
                            label: Some("Loss Bind Group Layout"),
                        }),
                    ],
                    push_constant_ranges: &[],
                });

                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Loss Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

                self.loss_pipeline = Some(pipeline.clone());
                pipeline
            }
        };

        // 创建输出和目标缓冲区
        let outputs_flat: Vec<f32> = outputs.iter().flat_map(|o| o.iter().cloned()).collect();
        let targets_flat: Vec<f32> = targets.iter().flat_map(|t| t.iter().cloned()).collect();

        let outputs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Outputs Buffer"),
            contents: bytemuck::cast_slice(&outputs_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let targets_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Targets Buffer"),
            contents: bytemuck::cast_slice(&targets_flat),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let input_size = output_size;

        // 创建参数缓冲区（批量大小和输出大小）
        let params = [
            batch_size as f32,
            input_size as f32,
            output_size as f32
        ];
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Params Buffer"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 创建绑定组
        let bind_group_layout = loss_pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: outputs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: targets_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: loss_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
            label: Some("Loss Bind Group"),
        });

        // 执行损失计算
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Loss Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Loss Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&loss_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            let workgroup_size = 256;
            let num_groups = (batch_size + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_groups as u32, 1, 1);
        }

        queue.submit(Some(encoder.finish()));

        // 读取损失结果
        let loss_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Loss Staging Buffer"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Loss Encoder"),
        });
        encoder.copy_buffer_to_buffer(&loss_buffer, 0, &loss_staging, 0, loss_staging.size());
        queue.submit(Some(encoder.finish()));

        let buffer_slice = loss_staging.slice(..);
        let (tx, rx) = oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());

        device.poll(wgpu::PollType::Wait);

        let data = buffer_slice.get_mapped_range();
        let loss_values: &[f32] = bytemuck::cast_slice(&data);

        // 计算平均损失
        let total_loss: f32 = loss_values.iter().sum();
        let avg_loss = total_loss / (batch_size * output_size) as f32;

        drop(data);
        loss_staging.unmap();

        avg_loss
    }

    /// 在GPU上执行反向传播
    fn gpu_backward(&mut self, inputs: &[&[f32]], outputs: &[Vec<f32>], targets: &[&[f32]], batch_size: usize) {
        if !self.gpu_initialized {
            // CPU回退 - 使用正确的参数调用backward
            let mut total_gradients = vec![vec![vec![0.0; inputs[0].len()]; outputs[0].len()]; self.layers.len()];
            let mut total_bias_gradients = vec![vec![0.0; outputs[0].len()]; self.layers.len()];
            for i in 0..batch_size {
                self.backward(&outputs[i], targets[i], &mut total_gradients, &mut total_bias_gradients);
            }
            self.apply_gradients(&total_gradients, &total_bias_gradients, batch_size as usize);
            return;
        }

        println!("[GPU训练] 执行GPU反向传播，批量大小: {}", batch_size);

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        // 为每层创建梯度缓冲区
        let mut gradient_buffers = Vec::new();
        let mut weight_gradient_buffers = Vec::new();
        let mut bias_gradient_buffers = Vec::new();

        // 创建输出层梯度（损失对输出的梯度）
        let output_size = outputs[0].len();
        let output_gradients: Vec<f32> = (0..batch_size * output_size)
            .map(|i| {
                let layer = i / output_size;
                let neuron = i % output_size;
                2.0 * (outputs[layer][neuron] - targets[layer][neuron])
            })
            .collect();

        let output_gradient_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Output Gradient Buffer"),
            contents: bytemuck::cast_slice(&output_gradients),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        gradient_buffers.push(output_gradient_buffer);

        // 从输出层向输入层反向传播
        for layer_idx in (0..self.layers.len()).rev() {
            let layer = &self.layers[layer_idx];
            let input_size = if layer_idx == 0 {
                self.layers[0].weights[0].len()
            } else {
                self.layers[layer_idx - 1].weights.len()
            };
            let output_size = layer.weights.len();

            println!("[GPU训练] 处理反向传播层: {}", layer_idx);

            // 创建当前层的权重梯度和偏置梯度缓冲区
            let weight_gradients_size = input_size * output_size;
            let bias_gradients_size = output_size;

            let weight_gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Layer {} Weight Gradient Buffer", layer_idx)),
                size: (weight_gradients_size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let bias_gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Layer {} Bias Gradient Buffer", layer_idx)),
                size: (bias_gradients_size * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            weight_gradient_buffers.push(weight_gradient_buffer.clone());
            bias_gradient_buffers.push(bias_gradient_buffer.clone());

            // 获取前一层的梯度（用于下一层反向传播）
            let prev_gradient_buffer = if layer_idx > 0 {
                gradient_buffers.last().unwrap().clone()
            } else {
                // 输入层不需要前一层梯度
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Dummy Gradient Buffer"),
                    size: 1,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };

            let prev_activations_buffer = if layer_idx > 0 {
                self.layers[layer_idx - 1].activations_buffer.as_ref().unwrap().clone()
            } else {
                // 输入层没有前一层
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Dummy Activations Buffer"),
                    size: 1,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };


            // 创建反向传播管线（如果不存在）
            let backward_pipeline = match &self.backward_pipelines.get(&layer.activation_func) {
                Some(pipeline) => pipeline.clone(),
                None => {
                    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some("Backward Shader"),
                        source: wgpu::ShaderSource::Wgsl(include_str!("backward.wgsl").into()),
                    });

                    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Backward Pipeline Layout"),
                        bind_group_layouts: &[
                            &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                                    wgpu::BindGroupLayoutEntry {
                                        binding: 5,
                                        visibility: wgpu::ShaderStages::COMPUTE,
                                        ty: wgpu::BindingType::Buffer {
                                            ty: wgpu::BufferBindingType::Uniform,
                                            has_dynamic_offset: false,
                                            min_binding_size: None,
                                        },
                                        count: None,
                                    },
                                    wgpu::BindGroupLayoutEntry {
                                        binding: 6,
                                        visibility: wgpu::ShaderStages::COMPUTE,
                                        ty: wgpu::BindingType::Buffer {
                                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                                            has_dynamic_offset: false,
                                            min_binding_size: None,
                                        },
                                        count: None,
                                    }
                                ],
                                label: Some("Backward Bind Group Layout"),
                            }),
                        ],
                        push_constant_ranges: &[],
                    });

                    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Backward Pipeline"),
                        layout: Some(&compute_pipeline_layout),
                        module: &shader,
                        entry_point: Some("main"),
                        cache: None,
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    });

                    self.backward_pipelines.insert(layer.activation_func.clone(), pipeline.clone());
                    // 返回克隆的值，而不是引用
                    &self.backward_pipelines.get(&layer.activation_func).unwrap()
                }
            };

            // 创建绑定组
            let bind_group_layout = backward_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: gradient_buffers.last().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: layer.weights_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: weight_gradient_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bias_gradient_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: prev_gradient_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.batch_size_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                ],
                label: Some(&format!("Backward Bind Group Layer {}", layer_idx)),
            });

            // 执行反向传播
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Backward Encoder Layer {}", layer_idx)),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Backward Pass Layer {}", layer_idx)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&backward_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let workgroup_size = 256;
                let num_groups = ((input_size * output_size) + workgroup_size - 1) / workgroup_size;
                compute_pass.dispatch_workgroups(num_groups as u32, 1, 1);
            }

            queue.submit(Some(encoder.finish()));

            // 为下一层准备梯度
            if layer_idx > 0 {
                let prev_gradient_size = input_size * batch_size;
                let prev_gradient_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("Layer {} Gradient Buffer", layer_idx - 1)),
                    size: (prev_gradient_size * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                // 这里应该添加计算前一层梯度的GPU操作
                // 由于实现细节复杂，简化为占位实现
                gradient_buffers.push(prev_gradient_buffer);
            }
        }
    }

    /// 在GPU上应用梯度更新权重
    fn gpu_apply_gradients(&mut self, batch_size: f32) {
        if !self.gpu_initialized {
            // CPU回退
            return;
        }

        println!("[GPU训练] 应用GPU梯度，批量大小: {}", batch_size as usize);

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        // 创建权重更新管线
        let update_pipeline = match &self.weight_update_pipeline {
            Some(pipeline) => pipeline.clone(),
            None => {
                let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Weight Update Shader"),
                    source: wgpu::ShaderSource::Wgsl(include_str!("update.wgsl").into()),
                });

                let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Weight Update Pipeline Layout"),
                    bind_group_layouts: &[
                        &device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                                wgpu::BindGroupLayoutEntry {
                                    binding: 2,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                                wgpu::BindGroupLayoutEntry {
                                    binding: 3,
                                    visibility: wgpu::ShaderStages::COMPUTE,
                                    ty: wgpu::BindingType::Buffer {
                                        ty: wgpu::BufferBindingType::Uniform,
                                        has_dynamic_offset: false,
                                        min_binding_size: None,
                                    },
                                    count: None,
                                },
                            ],
                            label: Some("Weight Update Bind Group Layout"),
                        }),
                    ],
                    push_constant_ranges: &[],
                });

                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Weight Update Pipeline"),
                    layout: Some(&compute_pipeline_layout),
                    module: &shader,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                });

                self.weight_update_pipeline = Some(pipeline.clone());
                pipeline
            }
        };

        let input_sizes: Vec<usize> = self.layers.iter().enumerate().map(|(idx, layer)| {
            if idx == 0 {
                layer.weights.get(0).map_or(0, |w| w.len())
            } else {
                self.layers[idx - 1].weights.len()
            }
        }).collect();

        // 为每层应用梯度
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let input_size = input_sizes[layer_idx];
            let output_size = layer.weights.len();

            println!("[GPU训练] 应用梯度到层: {}", layer_idx);

            // 创建学习率缓冲区
            let learning_rate = self.learning_rate;
            let weight_decay = self.weight_decay;
            let params = [learning_rate, weight_decay, batch_size];
            let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params Buffer"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            // 创建绑定组
            let bind_group_layout = update_pipeline.get_bind_group_layout(0);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: layer.weights_buffer.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.weight_gradient_buffers[layer_idx].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.bias_gradient_buffers[layer_idx].as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
                label: Some(&format!("Weight Update Bind Group Layer {}", layer_idx)),
            });

            // 执行权重更新
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("Weight Update Encoder Layer {}", layer_idx)),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("Weight Update Pass Layer {}", layer_idx)),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&update_pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                let workgroup_size = 256;
                let num_groups = ((input_size * output_size) + workgroup_size - 1) / workgroup_size;
                compute_pass.dispatch_workgroups(num_groups as u32, 1, 1);
            }

            queue.submit(Some(encoder.finish()));
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
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: layer.weights_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: layer.biases_buffer.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.batch_size_buffer.as_ref().unwrap().as_entire_binding(), // 使用全局的batch_size_buffer
                },
            ],
            label: Some("Matmul Bind Group"),
        })
    }

    // 添加更新批量大小的方法
    fn update_batch_size(&self, batch_size: usize) {
        if let (Some(device), Some(queue)) = (self.device.as_ref(), self.queue.as_ref()) {
            let batch_size_data = [batch_size as u32];
            queue.write_buffer(
                self.batch_size_buffer.as_ref().unwrap(),
                0,
                bytemuck::cast_slice(&batch_size_data),
            );
        }
    }

    // 创建批量激活函数绑定组的辅助方法
    fn create_batch_activation_bind_group(
        &self,
        layer: &NetworkLayer,
        buffer: &wgpu::Buffer,
        batch_size: usize,
    ) -> wgpu::BindGroup {
        let device = self.device.as_ref().unwrap();
        let layout = self.activation_bind_group_layouts
            .get(&layer.activation_func)
            .unwrap();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
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
            label: Some(&format!("Batch Activation Bind Group")),
        })
    }

    fn create_batch_matmul_bind_group(
        &self,
        layer: &NetworkLayer,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        _input_size: usize,
        _output_size: usize,
        batch_size: usize,
    ) -> wgpu::BindGroup {
        // 更新批量大小
        self.update_batch_size(batch_size);

        // 调用现有的创建绑定组方法
        self.create_matmul_bind_group(layer, input_buffer, output_buffer)
    }

    fn is_buffer_preallocated(&self, buffer: &wgpu::Buffer) -> bool {
        for preallocated in &self.pre_allocated_input_buffers {
            if let Some((ref prealloc_buffer, _size, _usage)) = preallocated {
                if std::ptr::eq(prealloc_buffer, buffer) {
                    return true;
                }
            }
        }

        for preallocated in &self.pre_allocated_output_buffers {
            if let Some((ref prealloc_buffer, _size, _usage)) = preallocated {
                if std::ptr::eq(prealloc_buffer, buffer) {
                    return true;
                }
            }
        }

        false
    }

    fn gpu_forward(&mut self, input: &[f32]) -> Vec<f32> {
        if !self.gpu_initialized {
            self.init_gpu_sync();
        }
        let batch_result = self.gpu_forward_batch(&[input], 1);
        match batch_result.into_iter().next() {
            Some(result) => result,
            None => {
                // 回退到 CPU
                self.light_forward(input)
            }
        }
    }

    fn prepare_batch_input_data(&self, inputs: &[&[f32]], actual_batch_size: usize, input_size: usize) -> Vec<f32> {
        let mut batch_input_data = Vec::with_capacity(input_size * actual_batch_size);
        for input in inputs.iter().take(actual_batch_size) {
            batch_input_data.extend_from_slice(input);
        }
        batch_input_data
    }

    fn gpu_forward_batch(&mut self, inputs: &[&[f32]], actual_batch_size: usize) -> Vec<Vec<f32>> {
        if self.device.is_none() || self.queue.is_none() || self.matmul_pipeline.is_none() || self.activation_pipelines.is_empty() {
            println!("[GPU推理] GPU 资源不完整 (device: {}, queue: {}, matmul_pipeline: {}, activation_pipelines: {}), 尝试重新初始化...",
                     self.device.is_some(), self.queue.is_some(), self.matmul_pipeline.is_none(), !self.activation_pipelines.is_empty());
            self.init_gpu_sync();
            // 重新检查
            if self.device.is_none() || self.queue.is_none() || self.matmul_pipeline.is_none() || self.activation_pipelines.is_empty() {
                println!("[GPU推理] GPU 初始化失败或不完整，回退到 CPU 处理批次大小: {}", actual_batch_size);
                return inputs.iter().map(|input| self.light_forward(input)).collect();
            } else {
                println!("[GPU推理] GPU 重新初始化成功，处理批次大小: {}", actual_batch_size);
            }
        }


        let input_size = inputs[0].len();

        // 如果批量大小变化或首次运行，重新预分配缓冲区
        if self.last_batch_size != actual_batch_size || self.pre_allocated_input_buffers.is_empty() {
            self.pre_allocate_buffers(actual_batch_size.max(64), input_size);
        }
        //println!("[GPU推理] 成功使用GPU处理批次大小: {}", actual_batch_size);

        // 准备输入数据
        let batch_input_data = self.prepare_batch_input_data(inputs, actual_batch_size, input_size);

        // 获取输入缓冲区 - 使用新的复用机制
        let batch_input_buffer = self.get_or_create_buffer(
            batch_input_data.len() * std::mem::size_of::<f32>(),
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "Batch Input Buffer"
        );

        self.queue.as_ref().unwrap().write_buffer(&batch_input_buffer, 0, bytemuck::cast_slice(&batch_input_data));

        let mut current_input_buffer = batch_input_buffer;
        let mut current_input_size = input_size;
        let mut buffers_to_return = Vec::new(); // 收集需要返还的缓冲区信息

        // 逐层处理
        for layer_idx in 0..self.layers.len() {
            //println!("GPU批量处理正确开启");
            let output_size = self.layers[layer_idx].weights.len();
            let total_output_size = output_size * actual_batch_size;

            // 获取输出缓冲区
            let output_buffer = self.get_or_create_buffer(
                total_output_size * std::mem::size_of::<f32>(),
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                &format!("Output Buffer Layer {}", layer_idx)
            );

            // 矩阵乘法计算
            let matmul_bind_group = self.get_matmul_bind_group(
                layer_idx,
                &current_input_buffer,
                &output_buffer,
                current_input_size,
                output_size,
                actual_batch_size
            );

            let mut encoder = self.device.as_ref().unwrap().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Matmul Encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Batch Matmul Pass"),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(self.matmul_pipeline.as_ref().unwrap());
                cpass.set_bind_group(0, &matmul_bind_group, &[]);

                let total_elements = output_size * actual_batch_size;
                let workgroup_count = (total_elements + 63) / 64;
                cpass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }

            self.queue.as_ref().unwrap().submit(Some(encoder.finish()));

            // 激活函数计算
            let activation_bind_group = self.get_activation_bind_group(
                layer_idx,
                &output_buffer,
                actual_batch_size
            );

            // 获取激活函数管道（避免借用冲突）
            let activation_func = self.layers[layer_idx].activation_func.clone();
            let activation_pipeline = self.activation_pipelines
                .get(&activation_func)
                .expect("Activation pipeline not initialized");

            let mut encoder = self.device.as_ref().unwrap().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Batch Activation Encoder"),
            });

            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Batch Activation Pass"),
                    timestamp_writes: None,
                });

                cpass.set_pipeline(activation_pipeline);
                cpass.set_bind_group(0, &activation_bind_group, &[]);

                let total_elements = output_size * actual_batch_size;
                let workgroup_count = (total_elements + 63) / 64;
                cpass.dispatch_workgroups(workgroup_count as u32, 1, 1);
            }

            self.queue.as_ref().unwrap().submit(Some(encoder.finish()));

            // 记录要返还的缓冲区信息
            let buffer_size = if layer_idx == 0 {
                input_size * actual_batch_size * std::mem::size_of::<f32>()
            } else {
                current_input_size * actual_batch_size * std::mem::size_of::<f32>()
            };

            buffers_to_return.push((
                current_input_buffer,
                buffer_size,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST
            ));

            // 准备下一层的输入
            current_input_buffer = output_buffer;
            current_input_size = output_size;
        }

        // 读取最终结果
        let final_output_size = self.layers.last().unwrap().weights.len();
        let total_final_size = final_output_size * actual_batch_size;

        let staging_buffer = self.get_or_create_buffer(
            total_final_size * std::mem::size_of::<f32>(),
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            "Staging Buffer"
        );

        // 将最终结果复制到staging buffer
        let mut encoder = self.device.as_ref().unwrap().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batch Readback Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &current_input_buffer,
            0,
            &staging_buffer,
            0,
            (total_final_size * std::mem::size_of::<f32>()) as u64,
        );

        self.queue.as_ref().unwrap().submit(Some(encoder.finish()));

        // 映射并读取结果
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        let _ = self.device.as_ref().unwrap().poll(wgpu::PollType::Wait);

        let results = match rx.recv() {
            Ok(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let batch_results: &[f32] = bytemuck::cast_slice(&data);

                // 将批量结果分割成单独的向量
                let mut results = Vec::with_capacity(actual_batch_size);
                for i in 0..actual_batch_size {
                    let start = i * final_output_size;
                    let end = start + final_output_size;
                    results.push(batch_results[start..end].to_vec());
                }

                drop(data);
                staging_buffer.unmap();
                results
            }
            _ => {
                // GPU失败时回退到CPU处理
                println!("GPU batch inference failed, falling back to CPU processing for batch of {} inputs", actual_batch_size);
                println!("你的GPU挂掉了");
                inputs.iter().take(actual_batch_size).map(|input| self.light_forward(input)).collect()
            }
        };

        for (buffer, size, usage) in buffers_to_return {
            if !self.is_buffer_preallocated(&buffer) {
                self.return_buffer_to_pool(buffer, size, usage);
            }
        }

        // 定期清理绑定组缓存
        self.cleanup_old_bind_groups(100);

        results
    }


/*
        * 构建神经网络架构 4 层全连接层，输入 28 维，输出 4 维
        * 前两维为左右手决策，第三维为手指
         */
    fn build_architecture(&mut self) {
        self.add_dense_layer(28, 128, ActivationFunction::BSiLU);
        self.add_dense_layer(128, 256, ActivationFunction::BSiLU);
        self.add_dense_layer(256, 256, ActivationFunction::BSiLU);
        self.add_dense_layer(256, 4, ActivationFunction::BSiLU);
        println!("Network architecture built with {} layers", self.layers.len());
        println!("Output size: 4 (L/R decision, finger assignment, confidence)");
    }

    //TODO: 归一化输出
    fn normalize_output(&self, raw_output: &[f32]) -> Vec<f32> {
        let mut normalized = raw_output.to_vec();
        if normalized.len() >= 2 {
            if (normalized[0] - normalized[1]).abs() < 1e-6 {
                normalized[0] = 0.5;
                normalized[1] = 0.5;
            } else {
                let temperature = 1.0;
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
            normalized[3] = normalized[3].clamp(0.1, 0.96);
        }

        normalized
    }

    fn add_dense_layer(&mut self, input_size: usize, output_size: usize, activation: ActivationFunction) {
        let mut weights = Vec::with_capacity(output_size);
        let mut biases = Vec::with_capacity(output_size);
        let mut momentum_weights = Vec::with_capacity(output_size);

        let std_dev = match activation {
            ActivationFunction::ReLU | ActivationFunction::GELU | ActivationFunction::Swish | ActivationFunction::BSiLU => {
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
            inputs: vec![0.0; output_size],
        };

        self.layers.push(layer);
    }

    // 双向 LSTM 层 (支持单向)
    fn add_lstm_layer_bi(&mut self, input_size: usize, output_size: usize, bidirectional: bool, seq_len: usize) {
        let dir_mul = if bidirectional { 2 } else { 1 };
        let rows = output_size * 4 * dir_mul;
        let layer = NetworkLayer {
            weights: vec![vec![0.0; input_size]; rows],
            biases: vec![0.0; rows],
            activations: vec![0.0; output_size * dir_mul],
            gradients: vec![0.0; output_size * dir_mul],
            momentum_weights: vec![vec![0.0; input_size]; rows],
            momentum_biases: vec![0.0; rows],
            bidirectional,
            seq_len: if seq_len == 0 { 1 } else { seq_len },
            layer_type: LayerType::LSTM,
            activation_func: ActivationFunction::Tanh,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
            inputs: vec![0.0; output_size * dir_mul],
        };
        self.layers.push(layer);
    }

    fn add_attention_layer(&mut self, input_size: usize, output_size: usize) {
        let layer = NetworkLayer {
            weights: vec![vec![0.0; input_size]; output_size * 3],
            biases: vec![0.0; output_size],
            activations: vec![0.0; output_size],
            gradients: vec![0.0; output_size],
            momentum_weights: vec![vec![0.0; input_size]; output_size * 3],
            momentum_biases: vec![0.0; output_size],
            bidirectional: false,
            seq_len: 1,
            layer_type: LayerType::Attention,
            activation_func: ActivationFunction::ReLU,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
            inputs: vec![0.0; output_size],
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
            bidirectional: false,
            seq_len: 1,
            layer_type: LayerType::Residual,
            activation_func: ActivationFunction::ReLU,
            weights_buffer: None,
            biases_buffer: None,
            activations_buffer: None,
            inputs: vec![0.0; output_size],
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

    fn activate_derivative(x: f32, func: &ActivationFunction) -> f32 {
        match func {
            ActivationFunction::ReLU => {
                if x > 0.0 { 1.0 } else { 0.0 }
            }
            ActivationFunction::Sigmoid => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid * (1.0 - sigmoid)
            }
            ActivationFunction::Tanh => {
                1.0 - x.tanh().powi(2)
            }
            ActivationFunction::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid + x * sigmoid * (1.0 - sigmoid)
            }
            ActivationFunction::GELU => {
                // 近似导数
                let cdf = 0.5 * (1.0 + (x / (2.0f32.sqrt())).tanh());
                let pdf = (-0.5 * x * x).exp() / (2.0 * std::f32::consts::PI).sqrt();
                cdf + x * pdf
            }
            ActivationFunction::BSiLU => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                let silu = x * sigmoid;
                let derivative = sigmoid * (1.0 + x * (1.0 - sigmoid));
                derivative / (1.0 + silu.abs()).powi(2)
            }
        }
    }

    fn dense_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let mut output = vec![0.0; layer.weights.len()];
        layer.inputs = vec![0.0; layer.weights.len()];

        for (i, (weights, bias)) in layer.weights.iter().zip(layer.biases.iter()).enumerate() {
            let mut sum = *bias;
            for (w, x) in weights.iter().zip(input.iter()) {
                sum += w * x;
            }
            layer.inputs[i] = sum;
            output[i] = DeepNeuralNetwork::activate(sum, &layer.activation_func);
        }
        layer.activations = output.clone();
        output
    }

    fn lstm_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        if layer.seq_len <= 1 {
            let output_size = layer.activations.len();
            let mut output = vec![0.0; output_size];
            //let rows = layer.weights.len();
            for i in 0..output_size {
                let mut sum = 0.0;
                if i < layer.weights.len() {
                    for (j, x) in input.iter().enumerate() {
                        if j < layer.weights[i].len() {
                            sum += layer.weights[i][j] * *x;
                        }
                    }
                }
                let bias = if i < layer.biases.len() { layer.biases[i] } else { 0.0 };
                sum += bias;
                output[i] = DeepNeuralNetwork::activate(sum, &layer.activation_func);
            }
            layer.activations = output.clone();
            output
        } else {
            let seq_len = layer.seq_len;
            let total_len = input.len();
            let input_size = if seq_len > 0 { total_len / seq_len } else { total_len };
            let out_per_dir = if layer.bidirectional { layer.activations.len() / 2 } else { layer.activations.len() };
            let mut seq_slices = Vec::new();
            for t in 0..seq_len {
                let start = t * input_size;
                let end = start + input_size.min(input.len() - start);
                seq_slices.push(&input[start..end]);
            }

            let rows_per_dir = layer.weights.len() / if layer.bidirectional { 2 } else { 1 };
            let compute_dir = |weights_chunk: &[Vec<f32>], biases_chunk: &[f32], seq: &[&[f32]]| -> Vec<Vec<f32>> {
                let mut h_seq = Vec::with_capacity(seq.len());
                let out_size = biases_chunk.len() / 4;
                for t in 0..seq.len() {
                    let x = seq[t];
                    let mut h_t = vec![0.0; out_size];
                    for i in 0..out_size {
                        let row_idx = i * 4;
                        if row_idx < weights_chunk.len() {
                            let wrow = &weights_chunk[row_idx];
                            let mut s = 0.0;
                            for (j, &xj) in x.iter().enumerate().take(wrow.len()) {
                                s += wrow[j] * xj;
                            }
                            let b = if row_idx < biases_chunk.len() { biases_chunk[row_idx] } else { 0.0 };
                            s += b;
                            h_t[i] = DeepNeuralNetwork::activate(s, &layer.activation_func);
                        }
                    }
                    h_seq.push(h_t);
                }
                h_seq
            };

            let fw_weights = &layer.weights[0..rows_per_dir];
            let fw_biases = &layer.biases[0..rows_per_dir];
            let fw_hidden = compute_dir(fw_weights, fw_biases, &seq_slices);
            let last_fw = fw_hidden.last().cloned().unwrap_or(vec![0.0; out_per_dir]);

            let mut outputs = last_fw;

            if layer.bidirectional {
                let bw_weights = &layer.weights[rows_per_dir..];
                let bw_biases = &layer.biases[rows_per_dir..];
                let rev_seq = seq_slices.into_iter().rev().collect::<Vec<_>>();
                let bw_hidden = compute_dir(bw_weights, bw_biases, &rev_seq);
                let last_bw = bw_hidden.last().cloned().unwrap_or(vec![0.0; out_per_dir]);
                outputs.extend(last_bw);
            }

            layer.activations = outputs.clone();
            outputs
        }
    }

    // TODO: 更复杂的注意力机制
    fn attention_forward(layer: &mut NetworkLayer, input: &[f32]) -> Vec<f32> {
        let output_size = layer.activations.len();
        let mut output = vec![0.0; output_size];

        let mut attention_weights = vec![0.0; input.len()];
        let mut attention_sum = 0.0;

        for i in 0..input.len() {
            let weight = (input[i] * input[i]).exp();
            attention_weights[i] = weight;
            attention_sum += weight;
        }

        if attention_sum == 0.0 {
            attention_sum = 1.0;
        }
        for weight in &mut attention_weights {
            *weight /= attention_sum;
        }

        for i in 0..output_size.min(input.len()) {
            output[i] = input[i] * attention_weights[i];
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

    fn train(&mut self, training_data: &[(Vec<f32>, Vec<f32>)]) {
        if training_data.is_empty() {
            eprintln!("警告: 训练数据为空，跳过训练");
            return;
        }

        // 初始化梯度累积
        let mut total_gradients: Vec<Vec<Vec<f32>>> = vec![vec![]; self.layers.len()];
        let mut total_bias_gradients: Vec<Vec<f32>> = vec![vec![]; self.layers.len()];

        // 准备梯度累积结构
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            if !layer.weights.is_empty() {
                total_gradients[layer_idx] = vec![vec![0.0; layer.weights[0].len()]; layer.weights.len()];
            }
            total_bias_gradients[layer_idx] = vec![0.0; layer.biases.len()];
        }

        let batch_count = (training_data.len() + self.batch_size - 1) / self.batch_size;
        let mut total_loss = 0.0;

        for batch_idx in 0..batch_count {
            let start_idx = batch_idx * self.batch_size;
            let end_idx = start_idx + self.batch_size.min(training_data.len() - start_idx);
            let batch = &training_data[start_idx..end_idx];
            let batch_size = batch.len() as f32;
            for layer_grad in &mut total_gradients {
                for row in layer_grad {
                    for g in row {
                        *g = 0.0;
                    }
                }
            }
            for bias_grad in &mut total_bias_gradients {
                for g in bias_grad {
                    *g = 0.0;
                }
            }
            for (input, target) in batch {
                let output = self.forward(input);
                let loss: f32 = output.iter().zip(target.iter())
                    .map(|(o, t)| (o - t).powi(2))
                    .sum();
                total_loss += loss;
                self.backward(&output, target, &mut total_gradients, &mut total_bias_gradients);
            }

            self.apply_weight_decay();
            self.clip_gradients(&mut total_gradients, &mut total_bias_gradients);
            self.apply_gradients(&total_gradients, &total_bias_gradients, batch_size as usize);
        }
        let avg_loss = total_loss / training_data.len() as f32;
        self.adapt_learning_rate(avg_loss);
        self.epoch_count += 1;
        if self.epoch_count % 10 == 0 {
            let (min_weight, max_weight) = self.get_weight_range();
            let (min_grad, max_grad) = self.get_gradient_range(&total_gradients);
            println!("[训练状态] Epoch: {}, Loss: {:.6}, LR: {:.6}, 权重范围 [{:.4}, {:.4}], 梯度范围 [{:.4}, {:.4}]",
                     self.epoch_count, avg_loss, self.learning_rate, min_weight, max_weight, min_grad, max_grad);
        }
    }

    fn apply_weight_decay(&mut self) {
        for layer in &mut self.layers {
            for i in 0..layer.weights.len() {
                for j in 0..layer.weights[i].len() {
                    layer.weights[i][j] -= self.learning_rate * self.weight_decay * layer.weights[i][j];
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
        if current_norm > self.max_grad_norm {
            let scale = self.max_grad_norm / current_norm;

            for layer_grad in gradients {
                for row in layer_grad {
                    for g in row {
                        *g *= scale;
                    }
                }
            }

            for bias_grad in bias_gradients {
                for g in bias_grad {
                    *g *= scale;
                }
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

    fn get_gradient_range(&self, gradients: &[Vec<Vec<f32>>]) -> (f32, f32) {
        let mut min_grad = f32::INFINITY;
        let mut max_grad = f32::NEG_INFINITY;

        for layer_grad in gradients {
            for row in layer_grad {
                for &g in row {
                    min_grad = min_grad.min(g);
                    max_grad = max_grad.max(g);
                }
            }
        }

        (min_grad, max_grad)
    }

    fn train_batch(&mut self, batch: &[(Vec<f32>, Vec<f32>)]) {
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
                &outputs[i],
                targets[i],
                &mut total_gradients,
                &mut total_bias_gradients
            );
        }

        self.apply_gradients(&total_gradients, &total_bias_gradients, batch_size);
        println!("First weight value after update: {:.6}", self.layers[0].weights[0][0]);
    }

    fn backward(&mut self, output: &[f32], target: &[f32], total_gradients: &mut [Vec<Vec<f32>>], total_bias_gradients: &mut [Vec<f32>]) {
        let mut layer_errors = vec![vec![0.0; 0]; self.layers.len()];

        // 计算输出层误差
        if let Some(last_layer_idx) = self.layers.len().checked_sub(1) {
            let output_errors: Vec<f32> = output.iter()
                .zip(target.iter())
                .map(|(o, t)| 2.0 * (o - t))
                .collect();

            let last_layer = &self.layers[last_layer_idx];
            let derivatives: Vec<f32> = last_layer.inputs.iter()
                .map(|&x| DeepNeuralNetwork::activate_derivative(x, &last_layer.activation_func))
                .collect();

            let output_layer_errors: Vec<f32> = output_errors.iter()
                .zip(derivatives.iter())
                .map(|(e, d)| e * d)
                .collect();

            layer_errors[last_layer_idx] = output_layer_errors;
        }

        // 反向传播误差
        for layer_idx in (0..self.layers.len() - 1).rev() {
            let layer = &self.layers[layer_idx];
            let next_layer = &self.layers[layer_idx + 1];
            let derivatives: Vec<f32> = layer.inputs.iter()
                .map(|&x| DeepNeuralNetwork::activate_derivative(x, &layer.activation_func))
                .collect();

            let mut current_errors = vec![0.0; layer.activations.len()];
            for i in 0..current_errors.len() {
                for (j, error) in layer_errors[layer_idx + 1].iter().enumerate() {
                    if j < next_layer.weights.len() && i < next_layer.weights[j].len() {
                        current_errors[i] += error * next_layer.weights[j][i];
                    }
                }
                current_errors[i] *= derivatives[i];
            }
            layer_errors[layer_idx] = current_errors;
        }

        // 计算梯度
        for layer_idx in 0..self.layers.len() {
            self.calculate_layer_gradients(layer_idx, &layer_errors[layer_idx],
                                           &mut total_gradients[layer_idx],
                                           &mut total_bias_gradients[layer_idx]);
        }
    }

    fn calculate_layer_gradients(&self, layer_idx: usize, errors: &[f32], gradients: &mut [Vec<f32>], bias_gradients: &mut [f32]) {
        //let layer = &self.layers[layer_idx];
        let prev_activations = if layer_idx > 0 {
            &self.layers[layer_idx - 1].activations
        } else {
            return;
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
        const MAX_GRAD: f32 = 10.0;

        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            for (i, weight_row) in layer.weights.iter_mut().enumerate() {
                for (j, weight) in weight_row.iter_mut().enumerate() {
                    if i < gradients[layer_idx].len() && j < gradients[layer_idx][i].len() {
                        let mut grad = gradients[layer_idx][i][j] / batch_size_f;

                        if grad > MAX_GRAD {
                            grad = MAX_GRAD;
                        } else if grad < -MAX_GRAD {
                            grad = -MAX_GRAD;
                        }

                        layer.momentum_weights[i][j] =
                            self.momentum * layer.momentum_weights[i][j] - self.learning_rate * grad;
                        *weight += layer.momentum_weights[i][j];
                    }
                }
            }

            for (i, bias) in layer.biases.iter_mut().enumerate() {
                if i < bias_gradients[layer_idx].len() {
                    let mut grad = bias_gradients[layer_idx][i] / batch_size_f;

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
            feature_extraction_bind_group_layout: None,
            feature_extraction_pipeline: None,
            note_buffer: None,
            bpm_buffer: None,
            params_buffer: None,
            feature_buffer: None,
            device: None,
            feature_count: 32,
            context_window: 32,
        }
    }

    fn init_feature_extraction(&mut self, device: &wgpu::Device) {
        // 创建特征提取的BindGroupLayout
        let feature_extraction_bind_group_layout = device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    // notes buffer
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
                    // bpm_points buffer
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
                    // params buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // features buffer (output)
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
                label: Some("Feature Extraction Bind Group Layout"),
            }
        );

        // 创建pipeline layout
        let pipeline_layout = device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("Feature Extraction Pipeline Layout"),
                bind_group_layouts: &[&feature_extraction_bind_group_layout],
                push_constant_ranges: &[],
            }
        );

        // 加载shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Feature Extraction Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("feature_extraction.wgsl").into()),
        });

        // 创建pipeline
        let pipeline = device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Feature Extraction Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            }
        );

        // 保存资源
        self.feature_extraction_bind_group_layout = Some(feature_extraction_bind_group_layout);
        self.feature_extraction_pipeline = Some(pipeline);
    }

    fn create_feature_extraction_buffers(&mut self, device: &wgpu::Device, max_notes: usize, max_bpm_points: usize) {
        // 创建音符缓冲区
        self.note_buffer = Some(device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Note Buffer"),
                size: (max_notes * std::mem::size_of::<GpuNote>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        ));

        // 创建BPM缓冲区
        self.bpm_buffer = Some(device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("BPM Buffer"),
                size: (max_bpm_points * std::mem::size_of::<GpuBpmPoint>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        ));

        // 创建参数缓冲区
        self.params_buffer = Some(device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Params Buffer"),
                size: std::mem::size_of::<GpuInputParams>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        ));

        // 创建特征输出缓冲区
        self.feature_buffer = Some(device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Feature Buffer"),
                size: (max_notes * self.feature_count * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }
        ));
    }

    fn run_feature_extraction(
        &mut self,
        queue: &wgpu::Queue,
        notes: &[ProcessedNote],
        bpm_list: &BpmList,
        context_window: u32,
        feature_count: usize,
        device: &wgpu::Device,
    ) -> Vec<f32> {
        // 检查必要的 GPU 资源是否已初始化
        if self.feature_extraction_bind_group_layout.is_none() || self.feature_extraction_pipeline.is_none() {
            eprintln!("[GPU Fallback] Feature extraction pipeline or layout not initialized, falling back to CPU");
            return self.extract_features_on_cpu(notes, bpm_list);
        }

        println!("[GPU特征提取] 成功使用GPU提取特征，音符数量: {}", notes.len());

        // --- 动态创建缓冲区 ---
        // 1. 创建音符缓冲区
        let gpu_notes: Vec<GpuNote> = notes.iter().map(|n| n.into()).collect();
        let note_buffer_size = gpu_notes.len() * std::mem::size_of::<GpuNote>();
        let note_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dynamic Note Buffer"),
            contents: bytemuck::cast_slice(&gpu_notes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 2. 创建BPM缓冲区
        let gpu_bpm_points: Vec<GpuBpmPoint> = bpm_list.points().iter().map(|p| p.into()).collect();
        let bpm_buffer_size = gpu_bpm_points.len() * std::mem::size_of::<GpuBpmPoint>();
        let bpm_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dynamic BPM Buffer"),
            contents: bytemuck::cast_slice(&gpu_bpm_points),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 3. 创建参数缓冲区
        let params = GpuInputParams {
            num_notes: notes.len() as u32,
            num_bpm_points: bpm_list.points().len() as u32,
            context_window,
            total_features: feature_count as u32, // 使用传入的 feature_count
        };
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Dynamic Params Buffer"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // 4. 创建特征输出缓冲区
        let feature_buffer_size = notes.len() * feature_count * std::mem::size_of::<f32>();
        let feature_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Dynamic Feature Buffer"),
            size: feature_buffer_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // --- 动态创建结束 ---

        // 创建 bind group (使用新创建的动态缓冲区)
        let bind_group = device.create_bind_group(
            &wgpu::BindGroupDescriptor {
                layout: self.feature_extraction_bind_group_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: note_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bpm_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: feature_buffer.as_entire_binding(),
                    },
                ],
                label: Some("Feature Extraction Bind Group"),
            }
        );

        // 创建 command encoder 并执行计算
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Feature Extraction") }
        );
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Feature Extraction Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(self.feature_extraction_pipeline.as_ref().unwrap());
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroup_count = (notes.len() as u32 + 63) / 64;
            cpass.dispatch_workgroups(workgroup_count, 1, 1);
        }
        queue.submit(Some(encoder.finish()));

        // 读取结果: 创建 staging buffer 并复制数据
        let staging_buffer = device.create_buffer(
            &wgpu::BufferDescriptor {
                label: Some("Staging Buffer"),
                size: feature_buffer_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Readback") }
        );
        encoder.copy_buffer_to_buffer(
            &feature_buffer,
            0,
            &staging_buffer,
            0,
            feature_buffer_size as u64,
        );
        queue.submit(Some(encoder.finish()));

        // 映射并读取结果
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        let _ = device.poll(wgpu::PollType::Wait);

        match rx.recv() {
            Ok(Ok(())) => {
                let data = buffer_slice.get_mapped_range();
                let features: &[f32] = bytemuck::cast_slice(&data);
                let result = features.to_vec();
                drop(data);
                staging_buffer.unmap();
                result
            }
            _ => {
                // GPU 处理失败，回退到 CPU
                eprintln!("[GPU Fallback] GPU feature extraction failed, falling back to CPU");
                self.extract_features_on_cpu(notes, bpm_list)
            }
        }
    }

    // 用于回退的CPU特征提取
    // 修改方法签名
    fn extract_features_on_cpu(&mut self, notes: &[ProcessedNote], bpm_list: &BpmList) -> Vec<f32> {
        notes.iter().enumerate().map(|(idx, note)| {
            let context_window = self.context_window as usize;
            let context_start = idx.saturating_sub(context_window / 2);
            let context_end = (idx + context_window / 2 + 1).min(notes.len());
            let context = &notes[context_start..context_end];

            // 直接调用 self.extract_features，而不是通过不存在的 self.feature_extractor
            self.extract_features(context, context_window, bpm_list)
        }).flatten().collect()
    }

    pub fn extract_features(&mut self, notes: &[ProcessedNote], window_size: usize, bpm_list: &BpmList) -> Vec<f32> {
        let mut features = Vec::new();
        if let Some(current_note) = notes.get(0) {
            let mut kind_features = vec![0.0; 4]; // 假设有4种音符类型
            match current_note.kind {
                NoteKind::Click => kind_features[0] = 1.0,
                NoteKind::Drag => kind_features[1] = 1.0,
                NoteKind::Flick => kind_features[2] = 1.0,
                NoteKind::Hold { .. } => kind_features[3] = 1.0,
            }
            features.extend(kind_features);
        } else {
            features.extend_from_slice(&[0.0; 4]); // 填充默认值
        }
        features.extend(self.extract_position_features(notes));
        features.extend(self.extract_temporal_features(notes, window_size, bpm_list.clone()));
        features.extend(self.extract_pattern_features(notes));
        features.extend(self.extract_difficulty_features(notes));
        features.extend(self.extract_velocity_features(notes));
        features.extend(self.extract_spatial_features(notes));
        for i in 0..features.len() {
            if !features[i].is_finite() {
                eprintln!("警告: 特征[{}]不是有效数值 (NaN 或无穷大): {}", i, features[i]);
                features[i] = 0.0;
            }
        }
        for i in 0..features.len() {
            match i {
                // 位置特征 (0-3)
                0..=3 => features[i] = features[i].clamp(-2.0, 2.0),
                // 时间特征 (4-9)
                4..=9 => features[i] = features[i].clamp(-10.0, 10.0),
                // 模式特征 (10-13)
                10..=13 => features[i] = features[i].clamp(-5.0, 5.0),
                // 难度特征 (14-17)
                14..=17 => features[i] = features[i].clamp(-5.0, 5.0),
                // 速度特征 (18-21)
                18..=21 => features[i] = features[i].clamp(-50.0, 50.0),
                // 空间特征 (22-27)
                22..=27 => features[i] = features[i].clamp(-10.0, 10.0),
                // 其他特征
                _ => features[i] = features[i].clamp(-5.0, 5.0),
            }
        }
        let mut min_val = features.iter().cloned().fold(f32::INFINITY, f32::min);
        let mut max_val = features.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if (max_val - min_val).abs() < f32::EPSILON {
            max_val = min_val + 1.0;
        }
        for i in 0..features.len() {
            features[i] = 2.0 * (features[i] - min_val) / (max_val - min_val) - 1.0;
        }
        let valid_features: Vec<f32> = features.iter().filter(|&&x| x.is_finite()).cloned().collect();
        let mean = if !valid_features.is_empty() {
            valid_features.iter().sum::<f32>() / valid_features.len() as f32
        } else {
            0.0
        };
        //println!("Features extracted: len={}, min={:.3}, max={:.3}, mean={:.3}",
        //        features.len(),
        //        features.iter().cloned().fold(f32::INFINITY, f32::min),
        //        features.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        //        mean);

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

    fn extract_temporal_features(&mut self, notes: &[ProcessedNote], window_size: usize, mut bpm_list: BpmList) -> Vec<f32> {
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
            println!("[BPM LOG] Time: {:.2}s, BPM: {:.1}", current_time, current_bpm);
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

        x_positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_positions.sort_by(|a, b| a.partial_cmp(b).unwrap());

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
        //GPU
        self.main_network.device = None;
        self.main_network.queue = None;
        self.main_network.matmul_pipeline = None;
        self.main_network.activation_pipelines.clear();
        self.target_network.device = None;
        self.target_network.queue = None;
        self.target_network.matmul_pipeline = None;
        self.target_network.activation_pipelines.clear();

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
            thread_count: if thread_pool.is_some() { 32 } else { 1 },
            feature_extractor: AdvancedFeatureExtractor::new(),
            experience_replay: ExperienceReplay::new(6000000),
            left_hand_state: HandState::new(Hand::Left, Vector2::new(-0.3, 0.0).rotate(rad)),
            right_hand_state: HandState::new(Hand::Right, Vector2::new(0.3, 0.0).rotate(rad)),
            rotation,
            exploration_rate: 0.05,
            discount_factor: 0.95,
            target_update_frequency: 1000,
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

        ai.target_network = ai.main_network.clone();

        // 新创建时也初始化 GPU
        println!("[GPU/CPU SWITCH] Beginning GPU init for newly created AI...");
        ai.main_network.init_gpu_sync();
        ai.target_network.init_gpu_sync();
        println!("[GPU/CPU SWITCH] New AI GPU initialized successfully.");
        if ai.main_network.gpu_initialized {
            if let Some(device) = ai.main_network.device.as_ref() {
                ai.feature_extractor.device = Some(device.clone());
                ai.feature_extractor.init_feature_extraction(device);
                // 预分配特征提取缓冲区，假设最大音符数为 1024，最大BPM点数为 256
                ai.feature_extractor.create_feature_extraction_buffers(device, 1024, 256);
                println!("[GPU Feature Buffers] Pre-allocated buffers for up to 1024 notes and 256 BPM points.");
            }
        }

        ai.warm_thread_pool();
        ai
    }

    fn load_or_create(filepath: &str, rotation: f32) -> Self {
        let path = Path::new(filepath);
        if let Ok(bytes) = fs::read(path) {
            if let Ok(mut ai) = bincode::deserialize::<Self>(&bytes) {
                println!("[Model Loading] Successfully loaded AI model from {}", filepath);

                // 关键：重置GPU状态标志，因为序列化数据不包含GPU资源
                // 这是解决"gpu_initialized: true 但实际资源缺失"问题的核心
                ai.main_network.gpu_initialized = false;
                ai.main_network.initialization_attempted = false;
                ai.main_network.initialization_failed = false;
                ai.target_network.gpu_initialized = false;
                ai.target_network.initialization_attempted = false;
                ai.target_network.initialization_failed = false;

                // 清理可能残留的GPU资源引用
                ai.main_network.device = None;
                ai.main_network.queue = None;
                ai.main_network.matmul_pipeline = None;
                ai.main_network.batch_size_buffer = None;
                ai.main_network.activation_pipelines.clear();
                ai.main_network.activation_bind_group_layouts.clear();
                ai.main_network.buffer_pool = None;

                ai.target_network.device = None;
                ai.target_network.queue = None;
                ai.target_network.matmul_pipeline = None;
                ai.target_network.batch_size_buffer = None;
                ai.target_network.activation_pipelines.clear();
                ai.target_network.activation_bind_group_layouts.clear();
                ai.target_network.buffer_pool = None;

                // 修复GPU资源池状态
                ai.main_network.pre_allocated_input_buffers.clear();
                ai.main_network.pre_allocated_output_buffers.clear();
                ai.main_network.pre_allocated_staging_buffer = None;

                ai.target_network.pre_allocated_input_buffers.clear();
                ai.target_network.pre_allocated_output_buffers.clear();
                ai.target_network.pre_allocated_staging_buffer = None;

                println!("[GPU/CPU SWITCH] Beginning GPU initialization for loaded model...");
                ai.main_network.init_gpu_sync();
                ai.target_network.init_gpu_sync();

                // 验证GPU状态是否真正就绪
                if ai.main_network.gpu_initialized {
                    if ai.main_network.device.is_none() ||
                        ai.main_network.queue.is_none() ||
                        ai.main_network.matmul_pipeline.is_none() {
                        println!("[GPU Repair] Detected inconsistent GPU state after initialization, forcing reset");
                        ai.main_network.gpu_initialized = false;
                        ai.main_network.initialization_attempted = false;
                        ai.main_network.initialization_failed = false;
                        ai.main_network.init_gpu_sync();
                    }
                }

                // 如果GPU初始化成功，设置feature_extractor
                if ai.main_network.gpu_initialized {
                    if let Some(device) = ai.main_network.device.as_ref() {
                        ai.feature_extractor.device = Some(device.clone());
                        ai.feature_extractor.init_feature_extraction(device);
                        // 预分配特征提取缓冲区
                        ai.feature_extractor.create_feature_extraction_buffers(device, 1024, 256);
                        println!("[GPU Feature Buffers] Pre-allocated buffers for up to 1024 notes and 256 BPM points.");
                    }
                }

                // 确保学习率等参数有效
                if !ai.main_network.learning_rate.is_finite() {
                    ai.main_network.learning_rate = 0.001;
                    println!("[Model Repair] Reset learning rate to default value");
                }

                if !ai.target_network.learning_rate.is_finite() {
                    ai.target_network.learning_rate = 0.001;
                }

                return ai;
            } else {
                println!("[Model Loading] Failed to deserialize AI model, creating new one");
            }
        } else {
            println!("[Model Loading] No existing model found at {}, creating new one", filepath);
        }

        let mut ai = Self::new(rotation);
        ai.save_model(filepath);


        // 额外验证GPU状态
        if ai.main_network.gpu_initialized {
            if ai.main_network.device.is_none() ||
                ai.main_network.queue.is_none() ||
                ai.main_network.matmul_pipeline.is_none() {
                println!("[GPU Repair] Detected inconsistent GPU state for new model, forcing reset");
                ai.main_network.gpu_initialized = false;
                ai.main_network.initialization_attempted = false;
                ai.main_network.initialization_failed = false;
                ai.main_network.init_gpu_sync();
            }
        }

        if ai.main_network.gpu_initialized {
            if let Some(device) = ai.main_network.device.as_ref() {
                ai.feature_extractor.device = Some(device.clone());
                ai.feature_extractor.init_feature_extraction(device);
                // 预分配特征提取缓冲区
                ai.feature_extractor.create_feature_extraction_buffers(device, 1024, 256);
                println!("[GPU Feature Buffers] Pre-allocated buffers for up to 1024 notes and 256 BPM points.");
            }
        }

        // 如果GPU初始化失败，记录警告但继续
        if !ai.main_network.gpu_initialized {
            println!("[GPU Warning] GPU initialization failed, falling back to CPU mode");
        }

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

    fn update_hand_positions(&mut self) {
        let rad = self.rotation.to_radians();
        self.left_hand_state.position = Vector2::new(-0.3, 0.0).rotate(rad);
        self.right_hand_state.position = Vector2::new(0.3, 0.0).rotate(rad);
    }

    fn light_update_hand_states(&mut self, notes: &[Note]) {
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

    fn detect_and_switch_mode(&mut self, notes: &[Note]) {
        if notes.len() < 8 {
            return;
        }

        let time_window = notes.last().unwrap().time - notes[0].time;
        let note_density = notes.len() as f32 / time_window.max(0.1);

        let mut max_simultaneous = 1;
        let mut current_time = notes[0].time;
        let mut current_group_size = 1;

        for i in 1..notes.len() {
            if notes[i].time == current_time {
                current_group_size += 1;
                max_simultaneous = max_simultaneous.max(current_group_size);
            } else {
                current_time = notes[i].time;
                current_group_size = 1;
            }
        }

        let should_switch = note_density > 8.0 || max_simultaneous > 2;

        if should_switch && self.game_mode != GameMode::FourFinger {
            self.game_mode = GameMode::FourFinger;
            self.finger_states = Self::init_finger_states(self.game_mode, self.rotation.to_radians());
        } else if !should_switch && self.game_mode != GameMode::TwoFinger {
            self.game_mode = GameMode::TwoFinger;
            self.finger_states = Self::init_finger_states(self.game_mode, self.rotation.to_radians());
        }
    }

    fn analyze_and_assign(&mut self, notes: &mut [Note], _config: &Config, bpm_list: &BpmList, line_id: usize) {
        println!("[DEBUG] analyze_and_assign started, notes.len() = {}", notes.len());
        //println!("[开始分析] 线路{} 音符数量:{} 游戏模式:{:?}", line_id, notes.len(), self.game_mode);
        if notes.is_empty() {
            println!("[DEBUG] analyze_and_assign: notes is empty, returning early");
            return;
        }

        let mut processed_notes = self.preprocess_notes(notes);
        let simultaneous_groups = self.detect_simultaneous_groups(&processed_notes);
        self.detect_and_switch_mode(notes);
        let mut bpm_list_clone = bpm_list.clone();
        self.assign_simultaneous_groups(&mut processed_notes, &simultaneous_groups, &mut bpm_list_clone, line_id);
        self.ai_assign_single_notes(&mut processed_notes, &simultaneous_groups, &mut bpm_list_clone, line_id);
        self.post_process_assignments(&mut processed_notes);
        self.optimize_jack_pattern(&mut processed_notes);

        // --- 关键修复：在 apply_and_learn 前添加长度检查 ---
        if notes.len() != processed_notes.len() {
            eprintln!("[CRITICAL ERROR in analyze_and_assign] Length mismatch before apply_and_learn! original: {}, processed: {}", notes.len(), processed_notes.len());
            // 打印前几个音符的信息以便调试
            for i in 0..notes.len().min(processed_notes.len()).min(5) {
                eprintln!("[DEBUG] Note {} - Original Time: {}, Processed Time: {}", i, notes.get(i).map(|n| n.time).unwrap_or(-1.0), processed_notes.get(i).map(|n| n.time).unwrap_or(-1.0));
            }
            return; // 提前返回，避免 apply_and_learn 中的潜在崩溃
        }
        // --- 关键修复结束 ---

        self.apply_and_learn(notes, &processed_notes);

        // --- 关键修复：在训练条件检查前再次添加长度检查 ---
        // 虽然 apply_and_learn 内部也有检查，但这里再加一层保险
        if notes.len() != processed_notes.len() {
            eprintln!("[CRITICAL ERROR in analyze_and_assign] Length mismatch after apply_and_learn! original: {}, processed: {}", notes.len(), processed_notes.len());
            return;
        }
        // --- 关键修复结束 ---

        if self.experience_replay.len() >= 64 && self.total_notes_processed % 25 == 0 {
            println!("[DEBUG] Training condition met: replay_len={}, total_notes={}", self.experience_replay.len(), self.total_notes_processed);
            self.train_network();
        } else {
            println!("[DEBUG] Training skipped: replay_len={}, total_notes={}", self.experience_replay.len(), self.total_notes_processed);
        }

        if self.training_episodes % self.target_update_frequency as u64 == 0 {
            self.target_network = self.main_network.clone();
        }

        self.training_episodes += 1;
        self.last_save_episodes += 1;
        if self.last_save_episodes >= 500 {
            println!("Saving episodes to {}", self.last_save_episodes);
            println!("训练回合数，已保存: {}", self.training_episodes);
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
                position: rotated_pos,
                time: note.time,
                kind: note.kind.clone(),
                assigned_hand: None,
                assigned_finger: None,
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

            if group.len() == 4 || group.len() == 5 || group.len() == 6 { //?
                let mut sorted_group: Vec<_> = group.iter()
                    .map(|&i| (i, notes[i].position.x))
                    .collect();
                sorted_group.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
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
            sorted_group.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

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
                for &note_idx in group {
                    let features = self.feature_extractor.extract_features(
                        &notes[note_idx..note_idx + 1],
                        1,
                        bpm_list
                    );
                    let ai_decision = self.make_ai_decision(&features, &notes[note_idx], line_id);

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

    fn ai_assign_single_notes(&mut self, notes: &mut [ProcessedNote], simultaneous_groups: &[Vec<usize>], bpm_list: &mut BpmList, line_id: usize) {
        let assigned_indices: std::collections::HashSet<usize> = simultaneous_groups.iter().flatten().copied().collect();
        const CONTEXT_WINDOW: usize = 32;
        const BATCH_SIZE: usize = 256;

        let unassigned_indices: Vec<usize> = (0..notes.len()).filter(|i| !assigned_indices.contains(i)).collect();
        if unassigned_indices.is_empty() {
            return;
        }

        for batch_indices in unassigned_indices.chunks(BATCH_SIZE) {
            let mut note_data = Vec::with_capacity(batch_indices.len());
            let mut batch_notes = Vec::with_capacity(batch_indices.len());
            for &idx in batch_indices {
                let note = &notes[idx];
                note_data.push((idx, note.position.x, note.time, note.judge.clone(), note.kind.clone()));
                batch_notes.push(note.clone());
            }

            // --- GPU 尝试阶段 ---
            if self.main_network.gpu_initialized &&
                self.feature_extractor.feature_extraction_pipeline.is_some() &&
                self.main_network.queue.is_some() &&
                self.feature_extractor.device.is_some() {

                let queue = self.main_network.queue.as_ref().unwrap();
                let device = self.feature_extractor.device.clone().unwrap();

                // 使用 catch_unwind 防止 GPU panic 导致程序崩溃
                let features_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    // --- 新增：健全性检查 ---
                    if self.feature_extractor.device.is_none() {
                        panic!("[Pre-Call Check] Feature extractor's device is None");
                    }
                    if self.feature_extractor.feature_extraction_bind_group_layout.is_none() ||
                        self.feature_extractor.feature_extraction_pipeline.is_none() {
                        panic!("[Pre-Call Check] Feature extraction GPU resources (layout/pipeline) not ready");
                    }
                    // --- 新增结束 ---

                    // 安全地获取 device 引用
                    let safe_device = self.feature_extractor.device.clone().expect("Device should be Some after check");
                    self.feature_extractor.run_feature_extraction(
                        queue,
                        &batch_notes,
                        bpm_list,
                        CONTEXT_WINDOW as u32,
                        self.feature_extractor.feature_count,
                        &safe_device, // <-- 传递安全的引用
                    )
                }));

                match features_result {
                    Ok(features) => {
                        println!("[GPU Acceleration] Successfully extracted features for {} notes using GPU.", batch_indices.len());
                        let feature_slices: Vec<&[f32]> = (0..batch_indices.len())
                            .map(|i| &features[i * self.feature_extractor.feature_count..(i + 1) * self.feature_extractor.feature_count])
                            .collect();

                        let outputs_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            self.main_network.gpu_forward_batch(&feature_slices, batch_indices.len())
                        }));

                        match outputs_result {
                            Ok(outputs) => {
                                // 成功处理整个批次
                                for (i, output) in outputs.iter().enumerate() {
                                    let (note_idx, position_x, time, judge, _kind) = &note_data[i];
                                    let note = &mut notes[*note_idx];
                                    note.features = features[i * self.feature_extractor.feature_count..(i + 1) * self.feature_extractor.feature_count].to_vec();
                                    let (chosen_hand, confidence, chosen_finger) = self.make_ai_decision(output, note, line_id);
                                    note.assigned_hand = Some(chosen_hand);
                                    note.assigned_finger = Some(chosen_finger);
                                    note.confidence = confidence;
                                    self.recent_assignments.push_back((chosen_hand, *position_x, *time));
                                    if self.recent_assignments.len() > 78 {
                                        self.recent_assignments.pop_front();
                                    }
                                    self.record_experience(&note.features, note, chosen_hand, confidence, *judge, &note.kind);
                                }
                                continue; // 跳过 CPU 回退
                            },
                            Err(_) => {
                                eprintln!("[GPU Fallback] GPU inference panicked, falling back to CPU for this batch.");
                            }
                        }
                    },
                    Err(_) => {
                        eprintln!("[GPU Fallback] GPU feature extraction panicked, falling back to CPU for this batch.");
                    }
                }
            }

            // --- CPU 回退阶段 (修复版) ---
            // 如果 GPU 阶段失败或未启用，则用 CPU 处理整个批次
            {
                println!("[CPU Fallback] Processing batch of {} notes on CPU.", batch_indices.len());

                let mut features_batch = Vec::with_capacity(batch_indices.len());
                let mut note_data_cpu = Vec::with_capacity(batch_indices.len()); // 重命名以避免混淆

                for &idx in batch_indices {
                    let note = &notes[idx];
                    let context_start = idx.saturating_sub(CONTEXT_WINDOW / 2);
                    let context_end = (idx + CONTEXT_WINDOW / 2 + 1).min(notes.len());
                    let context = &notes[context_start..context_end];
                    let features = self.feature_extractor.extract_features(context, CONTEXT_WINDOW, bpm_list);
                    features_batch.push(features);
                    note_data_cpu.push((idx, note.position.x, note.time, note.judge.clone(), note.kind.clone()));
                }

                // 对整个批次进行推理 (在 CPU 上)
                let outputs: Vec<Vec<f32>> = features_batch.iter()
                    .map(|features| self.main_network.forward(features))
                    .collect();

                // 处理整个批次的输出
                for (i, output) in outputs.iter().enumerate() {
                    let (note_idx, position_x, time, judge, _kind) = &note_data_cpu[i];
                    let note = &mut notes[*note_idx];
                    note.features = features_batch[i].clone(); // 保存特征
                    let (chosen_hand, confidence, chosen_finger) = self.make_ai_decision(output, note, line_id);
                    note.assigned_hand = Some(chosen_hand);
                    note.assigned_finger = Some(chosen_finger);
                    note.confidence = confidence;
                    self.recent_assignments.push_back((chosen_hand, *position_x, *time));
                    if self.recent_assignments.len() > 78 {
                        self.recent_assignments.pop_front();
                    }
                    self.record_experience(&features_batch[i], note, chosen_hand, confidence, *judge, &note.kind);
                }
            } // --- CPU 回退阶段结束 ---
        }
    }

    fn make_ai_decision(&mut self, features: &[f32], note: &ProcessedNote, line_id: usize) -> (Hand, f32, Finger) {
        //println!("[Token Usage] AI Decision - Features: {}, Time: {:.2}", features.len(), note.time);
        if self.finger_states.is_empty() {
            eprintln!("警告: finger_states 为空，重新初始化");
            self.finger_states = Self::init_finger_states(self.game_mode, self.rotation.to_radians());
        }

        //println!("[调试] 游戏模式: {:?}, 手指状态数量: {}", self.game_mode, self.finger_states.len());
        /*
        for (i, state) in self.finger_states.iter().enumerate() {
            println!("[调试] 手指 {}: {:?}", i, state.finger);
        }

         */
        let token = TOTAL_TOKENS_USED.fetch_add(1, Ordering::Relaxed);
        if token % 1000 == 0 {
            println!("Token: {}", token + 1);
        }
        let mut safe_features = Vec::with_capacity(features.len());
        let mut has_invalid = false;

        for (i, &feat) in features.iter().enumerate() {
            if !feat.is_finite() {
                eprintln!("警告: 无效的特征输入[{}]={}", i, feat);
                safe_features.push(0.0);
                has_invalid = true;
            } else {
                safe_features.push(feat);
            }
        }

        // 如果发现无效值，使用修复后的特征
        let features_to_use = if has_invalid {
            &safe_features
        } else {
            features
        };

        // 网络前向传播
        let raw_output = self.main_network.forward(features_to_use);
        //println!("Raw network output: {:?}", raw_output);

        // 输出归一化处理
        let network_output = self.main_network.normalize_output(&raw_output);

        let left_ai_confidence = network_output.get(0).copied().unwrap_or(0.4);
        let right_ai_confidence = network_output.get(1).copied().unwrap_or(0.5);
        let certainty = network_output.get(3).copied().unwrap_or(0.6);

        //println!("[AI决策] 线路{} 时间{:.2}s 位置({:.2},{:.2}) 网络输出: L:{:.3} R:{:.3} 确定性:{:.3}",
        //        line_id, note.time, note.position.x, note.position.y,
        //        left_ai_confidence, right_ai_confidence, certainty);

        // 计算手指评分
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

        for (i, (finger, score)) in finger_scores.iter().enumerate() {
            let state = self.finger_states.iter().find(|fs| fs.finger == *finger)
                .unwrap_or_else(|| {
                    eprintln!("错误: 未找到手指状态: {:?}", finger);
                    // 返回第一个手指状态作为备用
                    &self.finger_states[0]
                });
            //println!("  {}: {:?} 评分:{:.3} 位置({:.2},{:.2}) 疲劳:{:.2} 信心:{:.2} 繁忙:{}",
            //         i, finger, score, state.position.x, state.position.y,
            //         state.fatigue, state.confidence, state.is_busy);
        }

        let best_finger = finger_scores[0].0;
        let best_score = finger_scores[0].1;
        let chosen_hand = best_finger.to_hand();
        let position_weight = match chosen_hand {
            Hand::Left => {
                if note.position.x < -0.3 { 0.3 }
                else if note.position.x > -0.1 { -0.4 }
                else { -0.1 }
            },
            Hand::Right => {
                if note.position.x > 0.3 { 0.35 }
                else if note.position.x < -0.14 { -0.3 }
                else { 0.5 }
            },
        };
        /*
        let initial_hand = best_finger.to_hand();
        let mut final_hand = initial_hand;

        if !self.recent_assignments.is_empty() {
            let last_assigned_hand = self.recent_assignments.back()
                .map(|(hand, _, _)| *hand)
                .unwrap_or(initial_hand);

            let mut consecutive_same_hand = 0;
            for (hand, _, _) in self.recent_assignments.iter().rev() {
                if *hand == last_assigned_hand {
                    consecutive_same_hand += 1;
                } else {
                    break;
                }
            }

            if consecutive_same_hand >= 2 {
                let opposite_hand = match last_assigned_hand {
                    Hand::Left => Hand::Right,
                    Hand::Right => Hand::Left,
                };

                let opposite_hand_score = finger_scores.iter()
                    .find(|(finger, _)| finger.to_hand() == opposite_hand)
                    .map(|(_, score)| *score)
                    .unwrap_or(0.0);

                let position_suitable = match opposite_hand {
                    Hand::Left => note.position.x < 0.0,
                    Hand::Right => note.position.x > 0.0,
                };

                if opposite_hand_score > best_score * 0.6 && position_suitable {
                    final_hand = opposite_hand;
                    if let Some((opposite_finger, _)) = finger_scores.iter()
                        .find(|(finger, _)| finger.to_hand() == opposite_hand)
                    {
                        //TODO: 这里可以更新 best_finger
                    }
                }
            }
        }

         */

        let base_certainty = certainty.clamp(0.3, 0.85);
        let ai_weight = base_certainty * self.pattern_recognition_strength * 0.7;
        let heuristic_weight = 1.0 - ai_weight;

        let hand_ai_confidence = if chosen_hand == Hand::Left {
            left_ai_confidence
        } else {
            right_ai_confidence
        };

        let final_confidence = (
            hand_ai_confidence * ai_weight +
                best_score * heuristic_weight +
                position_weight * 0.1
        ).clamp(0.2, 0.92);

        let adjusted_exploration = self.exploration_rate * 0.6;
        let final_hand = if fastrand::f32() < adjusted_exploration {
            if fastrand::bool() { Hand::Left } else { Hand::Right }
        } else {
            chosen_hand
        };

        //println!("[最终决策] 线路{} 时间{:.2}s 选择:{:?} 信心:{:.3} 探索:{:.3}",
        //  line_id, note.time, final_hand, final_confidence, adjusted_exploration);

        (final_hand, final_confidence, best_finger)
    }

    fn record_experience(&mut self, features: &[f32], note: &ProcessedNote, chosen_hand: Hand, confidence: f32, judge:JudgeStatus, note_kind: &NoteKind) {
        //println!("Recording experience, replay size now: {}", self.experience_replay.len());
        let reward = self.calculate_reward(note, chosen_hand, confidence, judge, &note.kind);
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

    fn calculate_reward(&self, note: &ProcessedNote, chosen_hand: Hand, confidence: f32, judge: JudgeStatus, note_kind: &NoteKind) -> f32 {
        let mut reward = 0.0;

        // 1. 基础奖励（基于判定结果）
        let base_reward = match judge {
            JudgeStatus::Judged => 1.0,
            JudgeStatus::NotJudged => -0.5,
            JudgeStatus::PreJudge => 0.0,
            JudgeStatus::Hold(perfect, _, _, _, _) => {
                if perfect { 1.2 } else { 0.8 }
            }
        };
        reward += base_reward;

        // 2. 位置奖励
        let position_reward = if note.position.x < -0.1 {
            if chosen_hand == Hand::Left { 0.5 } else { -0.3 }
        } else if note.position.x > 0.1 {
            if chosen_hand == Hand::Right { 0.5 } else { -0.3 }
        } else {
            0.0
        };
        reward += position_reward;

        // 3. 新增：音符类型奖励/惩罚
        // 不同类型音符有不同的“基础难度”，成功处理高难度音符应获得更高奖励
        let kind_difficulty_bonus = match note_kind {
            NoteKind::Click => 0.0,      // 基础难度
            NoteKind::Drag => 0.2,       // 稍难
            NoteKind::Flick => 0.3,      // 较难
            NoteKind::Hold { .. } => 0.4, // 最难
        };
        // 只有成功判定时才给予难度奖励
        if matches!(judge, JudgeStatus::Judged | JudgeStatus::Hold(..)) {
            reward += kind_difficulty_bonus;
        }

        // 4. 信心奖励（仅在成功时）
        if matches!(judge, JudgeStatus::Judged | JudgeStatus::Hold(..)) {
            reward += (confidence - 0.5) * 0.5;
        }

        // 5. 随机噪声
        let noise = (fastrand::f32() - 0.5) * 0.1;
        reward += noise;

        // 限制奖励范围
        reward.clamp(-2.0, 2.0)
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
            } else if jack_score > 0.6 {
                self.optimize_jack_pattern(&mut notes[i..window_end]);
                i = window_end;
            } else if crossing_score > 0.6 {
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
                sorted_indices.sort_by(|&a, &b| notes[a].position.x.partial_cmp(&notes[b].position.x).unwrap());

                let mid = sorted_indices.len() / 2;
                for (pos, &idx) in sorted_indices.iter().enumerate() {
                    notes[idx].assigned_hand = Some(if pos < mid { Hand::Left } else { Hand::Right });
                    notes[idx].confidence = 0.9;
                }
            }
        }
    }

    fn detect_crossing_pattern(&self, notes: &[ProcessedNote]) -> f32 {
        if notes.len() < 3 {
            return 0.0;
        }
        let mut crossing_score = 0.0;
        for i in 1..notes.len() {
            let prev_x = notes[i-1].position.x;
            let curr_x = notes[i].position.x;
            // 检查是否从正到负或负到正
            if prev_x * curr_x < 0.0 {
                crossing_score += 1.0;
            }
        }
        crossing_score / (notes.len() - 1) as f32
    }

    fn optimize_crossing_pattern(&mut self, notes: &mut [ProcessedNote]) {
        if notes.is_empty() {
            return;
        }
        // 确定起始手：根据第一个音符的位置
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

                let success = processed.confidence > self.confidence_threshold;
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
        if current_accuracy < 0.3 {
            self.main_network.learning_rate = (self.main_network.learning_rate * 1.5).clamp(0.001, 0.05);
        } else if current_accuracy < self.average_reward {
            self.main_network.learning_rate = (self.main_network.learning_rate * 1.1).min(0.03);
        } else {
            self.main_network.learning_rate = (self.main_network.learning_rate * 0.98).max(0.0005);
        }

        self.exploration_rate = if current_accuracy > 0.7 {
            0.05 + (0.25 * (1.0 - current_accuracy))
        } else {
            0.3
        };

        if self.main_network.epoch_count > 0 {
            if current_accuracy > self.average_reward {
                self.main_network.learning_rate *= 1.02;
            } else {
                self.main_network.learning_rate *= 0.98;
            }
            self.main_network.learning_rate = self.main_network.learning_rate.clamp(0.0001, 0.05);
        }

        if current_accuracy > 0.85 {
            self.confidence_threshold = (self.confidence_threshold + 0.01).min(0.9);
        } else if current_accuracy < 0.65 {
            self.confidence_threshold = (self.confidence_threshold - 0.01).max(0.5);
        }
    }
        fn train_network(&mut self) {
            println!("[训练前检查] main_network.gpu_initialized: {}", self.main_network.gpu_initialized);
            println!("[训练前检查] main_network.device.is_some(): {}", self.main_network.device.is_some());
            println!("[训练前检查] main_network.queue.is_some(): {}", self.main_network.queue.is_some());
            println!("[训练前检查] main_network.matmul_pipeline.is_some(): {}", self.main_network.matmul_pipeline.is_some());

            // 强制重新检查并修复GPU状态
            self.main_network.init_gpu_sync();
            self.target_network.init_gpu_sync();

            // 再次验证GPU状态
            if self.main_network.gpu_initialized {
                if self.main_network.device.is_none() ||
                    self.main_network.queue.is_none() ||
                    self.main_network.matmul_pipeline.is_none() {
                    println!("[训练前修复] 检测到GPU状态仍不一致，强制重置状态");
                    // 强制重置状态
                    self.main_network.gpu_initialized = false;
                    self.main_network.initialization_attempted = false;
                    self.main_network.initialization_failed = false;
                    // 再次尝试初始化
                    self.main_network.init_gpu_sync();
                }
            }

            // 确保feature_extractor也正确设置
            if self.main_network.gpu_initialized {
                if let Some(device) = self.main_network.device.as_ref() {
                    self.feature_extractor.device = Some(device.clone());
                    self.feature_extractor.init_feature_extraction(device);
                    // 重新创建特征提取缓冲区
                    self.feature_extractor.create_feature_extraction_buffers(device, 1024, 256);
                    println!("[GPU Feature Buffers] 重新分配了特征提取缓冲区");
                }
            }

            // 强制使用64的批量大小
            const BATCH_SIZE: usize = 64;
            let experiences: Vec<_> = self.experience_replay.sample(BATCH_SIZE).into_iter().cloned().collect();

            if experiences.is_empty() {
                return; // 没有经验，不训练
            }

            let mut training_data = Vec::with_capacity(BATCH_SIZE);
            for exp in &experiences {
                let target_value = (exp.reward + self.discount_factor * self.estimate_future_value(&exp.next_state)).clamp(0.0, 1.0);
                let mut target_output = vec![0.5, 0.5, 1.0, exp.reward.abs()];

                if exp.action < target_output.len() {
                    target_output[exp.action] = target_value.clamp(0.0, 1.0);
                }

                training_data.push((exp.state.clone(), target_output));
            }

            // 使用GPU训练
            if self.main_network.gpu_initialized {
                println!("[训练] 使用GPU进行训练，批量大小: {}", BATCH_SIZE);
                let loss = self.main_network.gpu_train_batch(&training_data);
                println!("[训练] GPU训练完成，Loss: {:.6}", loss);
            } else {
                // CPU回退
                println!("[训练] GPU未启用，使用CPU进行训练，样本数: {}", training_data.len());
                self.main_network.train(&training_data);
            }

            // 训练完成后，如果GPU已初始化，确保权重同步
            if self.main_network.gpu_initialized {
                println!("[训练] 权重已在GPU上更新，无需额外同步");
            }

            println!("网络训练完成: Epoch {}, 学习率: {:.6}, 探索率: {:.3}, 准确率: {:.3}",
                     self.main_network.epoch_count,
                     self.main_network.learning_rate,
                     self.exploration_rate,
                     self.average_reward);
        }

    fn estimate_future_value(&mut self, state: &[f32]) -> f32 {
        // 确保目标网络的GPU资源有效
        self.target_network.init_gpu_sync();
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

const MATMUL_WGSL: &str = r#"
struct BatchSize {
    size: u32,
};

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> biases: array<f32>;
@group(0) @binding(4) var<storage, read> batch_size: BatchSize;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let output_size = arrayLength(&biases);
    let input_size = arrayLength(&input) / batch_size.size;
    let global_id = id.x;

    if (global_id >= output_size * batch_size.size) {
        return;
    }

    let batch_idx = global_id / output_size;
    let output_idx = global_id % output_size;

    var sum = biases[output_idx];
    for (var i = 0u; i < input_size; i++) {
        let input_val = input[batch_idx * input_size + i];
        let weight_idx = output_idx * input_size + i;
        sum += weights[weight_idx] * input_val;
    }

    output[batch_idx * output_size + output_idx] = sum;
}
"#;