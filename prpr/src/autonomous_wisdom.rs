use crate::hand_model::{ErgonomicHandSystem, Vector2};
use crate::core::{Note, note::Hand};
use crate::config::Config;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub struct ConsciousnessSuggestion {
    pub suggested_hand: Hand,
    pub confidence: f32,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessCore {
    pub self_awareness: Vec<f32>,
    pub intrinsic_motivation: f32,
    pub curiosity_index: f32,
    pub evolution_goals: Vec<EvolutionGoal>,
    pub consciousness_state: ConsciousnessState,
    pub reflection_history: VecDeque<ReflectionEvent>,
    pub metacognition: MetacognitionSystem,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum ConsciousnessState {
    Exploring,    // 探索模式
    Learning,     // 学习模式
    Optimizing,   // 优化模式
    Creating,     // 创造模式
    Reflecting,   // 反思模式
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionGoal {
    pub goal_type: String,
    pub priority: f32,
    pub progress: f32,
    pub target_value: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionEvent {
    pub timestamp: f32,
    pub trigger: String,
    pub insight: String,
    pub confidence: f32,
    pub impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetacognitionSystem {
    pub strategy_assessment: HashMap<String, f32>,
    pub bias_detection: Vec<CognitiveBias>,
    pub learning_efficiency: f32,
    pub cognitive_flexibility: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveBias {
    pub bias_type: String,
    pub strength: f32,
    pub detection_confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfDrivingNetwork {
    pub consciousness: ConsciousnessCore,
    pub knowledge_graph: KnowledgeGraph,
    pub creativity_engine: CreativityEngine,
    pub self_regulation: SelfRegulationSystem,
    pub learning_params: SelfLearningParams,
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveActivation {
    pub function_type: ActivationType,
    pub adaptability: f32,
    pub parameters: Vec<f32>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ActivationType { Sigmoid, Tanh, ReLU, GELU, Swish, Adaptive, }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronState {
    pub activation: f32,
    pub potential: f32,
    pub adaptation_level: f32,
    pub connection_strength: f32,
    pub firing_pattern: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    pub concepts: HashMap<String, ConceptNode>,
    pub relations: HashMap<String, RelationEdge>,
    pub update_frequency: f32,
    pub abstraction_levels: Vec<AbstractionLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    pub concept_id: String,
    pub embedding: Vec<f32>,
    pub importance: f32,
    pub activation_level: f32,
    pub abstraction_level: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationEdge {
    pub source: String,
    pub target: String,
    pub relation_type: String,
    pub strength: f32,
    pub directionality: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbstractionLevel {
    pub level: usize,
    pub concepts: Vec<String>,
    pub abstraction_degree: f32,
}

#[derive(Debug, Clone)]
struct FeatureAnalysis {
    importance: f32,
    activation: f32,
    signature: f32,
    complexity: f32,
    pattern_type: PatternType,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum PatternType {
    Simple,
    Sequential,
    Spatial,
    Temporal,
    Complex,
}

#[derive(Debug, Clone)]
struct NodeUpdateResult {
    success: bool,
    failure_reason: String,
    updated_confidence: f32,
    propagation_delay: f32,
}

#[derive(Debug, Clone)]
struct NodeInfo {
    address: String,
    capacity: u32,
    load: f32,
    reliability: f32,
}

#[derive(Debug, Clone)]
struct BroadcastResult {
    success: bool,
    latency: f32,
    node_id: String,
}

#[derive(Debug, Clone, Copy)]
enum SyncStrategy {
    Immediate,
    Priority,
    Batched,
    Deferred,
}

#[derive(Debug, Clone)]
struct SyncResult {
    success: bool,
    covered_nodes: usize,
    sync_time: f32,
    strategy: SyncStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativityEngine {
    pub creativity_params: CreativityParams,
    pub imagination_space: ImaginationSpace,
    pub innovation_history: VecDeque<InnovationEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreativityParams {
    pub divergent_thinking: f32,
    pub convergent_thinking: f32,
    pub originality_drive: f32,
    pub synthesis_ability: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImaginationSpace {
    pub dimensions: usize,
    pub exploration_intensity: f32,
    pub novelty_threshold: f32,
    pub conceptual_boundaries: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InnovationEvent {
    pub timestamp: f32,
    pub innovation_type: String,
    pub novelty_score: f32,
    pub utility_score: f32,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfRegulationSystem {
    pub emotional_regulation: EmotionalRegulation,
    pub attention_control: AttentionControl,
    pub motivation_management: MotivationManagement,
    pub behavioral_selection: BehavioralSelection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalRegulation {
    pub current_emotion: EmotionalState,
    pub emotional_stability: f32,
    pub regulation_strategies: Vec<RegulationStrategy>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum EmotionalState { Curious, Focused, Frustrated, Excited, Calm, Creative, }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulationStrategy {
    pub strategy_name: String,
    pub effectiveness: f32,
    pub usage_frequency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionControl {
    pub focus_level: f32,
    pub attention_span: f32,
    pub distraction_resistance: f32,
    pub selective_attention: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotivationManagement {
    pub intrinsic_drive: f32,
    pub goal_orientation: f32,
    pub persistence_level: f32,
    pub self_efficacy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralSelection {
    pub action_preferences: HashMap<String, f32>,
    pub risk_tolerance: f32,
    pub exploration_exploitation: f32,
    pub habit_strength: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfLearningParams {
    pub learning_rate: f32,
    pub adaptation_rate: f32,
    pub plasticity_decay: f32,
    pub memory_consolidation: f32,
    pub forgetting_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLearningFeedback {
    pub global_knowledge: Arc<Mutex<GlobalKnowledgePool>>,
    pub distributed_nodes: Vec<LearningNode>,
    pub collective_intelligence: CollectiveIntelligence,
    pub feedback_propagation: FeedbackPropagation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalKnowledgePool {
    pub shared_concepts: HashMap<String, SharedConcept>,
    pub knowledge_versions: HashMap<String, u32>,
    pub quality_assessment: HashMap<String, f32>,
    pub propagation_network: HashMap<String, Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedConcept {
    pub concept: ConceptNode,
    pub contributor_nodes: Vec<String>,
    pub consensus_level: f32,
    pub validation_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningNode {
    pub node_id: String,
    pub local_knowledge: KnowledgeGraph,
    pub expertise_areas: Vec<String>,
    pub learning_history: VecDeque<LearningEvent>,
    pub contribution_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningEvent {
    pub timestamp: f32,
    pub event_type: String,
    pub knowledge_gained: String,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectiveIntelligence {
    pub emergent_behaviors: Vec<EmergentBehavior>,
    pub swarm_optimization: SwarmOptimization,
    pub consensus_mechanisms: ConsensusMechanism,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergentBehavior {
    pub behavior_id: String,
    pub pattern: String,
    pub frequency: f32,
    pub utility: f32,
    pub participants: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmOptimization {
    pub particle_positions: Vec<Vec<f32>>,
    pub global_best: Vec<f32>,
    pub velocity_vectors: Vec<Vec<f32>>,
    pub convergence_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusMechanism {
    pub voting_weights: HashMap<String, f32>,
    pub consensus_threshold: f32,
    pub proposal_history: VecDeque<Proposal>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proposal {
    pub proposal_id: String,
    pub proposer: String,
    pub content: String,
    pub support_votes: u32,
    pub total_votes: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackPropagation {
    pub feedback_channels: Vec<FeedbackChannel>,
    pub amplification_factors: HashMap<String, f32>,
    pub damping_coefficients: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackChannel {
    pub channel_id: String,
    pub source_node: String,
    pub target_nodes: Vec<String>,
    pub signal_strength: f32,
    pub latency: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousWisdomAI {
    pub hand_ai: crate::hand::PhiTKAdvancedAI,
    pub network_feedback: NetworkLearningFeedback,
    pub hand_system_interface: HandSystemInterface,
    pub evolution_history: VecDeque<EvolutionEvent>,
    pub current_consciousness: ConsciousnessState,
    pub consciousness: ConsciousnessCore,
    pub birth_time: f32,
    pub experience_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandSystemInterface {
    pub ergonomic_system: ErgonomicHandSystem,
    pub performance_metrics: PerformanceMetrics,
    pub adaptation_history: VecDeque<AdaptationEvent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub accuracy: f32,
    pub speed: f32,
    pub efficiency: f32,
    pub creativity: f32,
    pub learning_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationEvent {
    pub timestamp: f32,
    pub adaptation_type: String,
    pub effectiveness: f32,
    pub context: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEvent {
    pub timestamp: f32,
    pub evolution_type: String,
    pub complexity_change: f32,
    pub new_capabilities: Vec<String>,
}

impl ConsciousnessCore {
    pub fn new() -> Self {
        Self {
            self_awareness: vec![0.5; 64],
            intrinsic_motivation: 0.8,
            curiosity_index: 0.9,
            evolution_goals: vec![
                EvolutionGoal {
                    goal_type: "知识获取".to_string(),
                    priority: 0.9,
                    progress: 0.0,
                    target_value: 1.0,
                },
                EvolutionGoal {
                    goal_type: "技能提升".to_string(),
                    priority: 0.8,
                    progress: 0.0,
                    target_value: 1.0,
                },
            ],
            consciousness_state: ConsciousnessState::Exploring,
            reflection_history: VecDeque::new(),
            metacognition: MetacognitionSystem::new(),
        }
    }

    pub fn update_consciousness(&mut self, context: &WisdomContext) {
        // 基于多维度因素智能调整意识状态
        let new_state = self.determine_optimal_state_intelligent(context);
        
        // 记录状态转换
        if new_state != self.consciousness_state {
            self.self_reflect(
                &format!("状态转换: {:?} -> {:?}", self.consciousness_state, new_state),
                &format!("复杂度:{:.2}, 挑战:{:.2}, 新颖性:{:.2}", 
                         context.complexity_level, context.challenge_level, context.novelty_score),
                0.8
            );
            self.consciousness_state = new_state;
        }
        
        // 更新自我认知
        self.update_self_awareness(context);
        
        // 调整内在动机
        self.adapt_motivation(context);
        
        // 更新好奇心
        self.update_curiosity(context);
        
        // 基于当前状态调整元认知
        self.update_metacognition(context);
    }

    /// 智能确定最优意识状态
    fn determine_optimal_state_intelligent(&self, context: &WisdomContext) -> ConsciousnessState {
        // 计算状态转换的适应性分数
        let mut state_scores = HashMap::new();
        
        // 基于复杂度的基础分数
        state_scores.insert(ConsciousnessState::Exploring, (1.0 - context.complexity_level) * 0.8);
        state_scores.insert(ConsciousnessState::Learning, 
                           (1.0 - (context.complexity_level - 0.3).abs()) * 0.9);
        state_scores.insert(ConsciousnessState::Optimizing, 
                           (1.0 - (context.complexity_level - 0.6).abs()) * 0.85);
        state_scores.insert(ConsciousnessState::Creating, context.complexity_level * 0.9);
        
        // 基于挑战水平的调整
        for (state, score) in state_scores.iter_mut() {
            match state {
                ConsciousnessState::Exploring => {
                    if context.challenge_level < 0.4 {
                        *score *= 1.2; // 低挑战时探索更有效
                    }
                },
                ConsciousnessState::Learning => {
                    if context.challenge_level > 0.3 && context.challenge_level < 0.7 {
                        *score *= 1.3; // 适中挑战时学习最佳
                    }
                },
                ConsciousnessState::Optimizing => {
                    if context.challenge_level > 0.5 {
                        *score *= 1.2; // 高挑战时优化更重要
                    }
                },
                ConsciousnessState::Creating => {
                    if context.challenge_level > 0.7 {
                        *score *= 1.4; // 极高挑战时创造性突破
                    }
                },
                ConsciousnessState::Reflecting => {
                    *score = context.novelty_score * 0.7; // 高新颖性时需要反思
                },
            }
        }
        
        // 基于当前状态的连续性调整（避免频繁切换）
        let continuity_bonus = match self.consciousness_state {
            ConsciousnessState::Exploring => 0.1,
            ConsciousnessState::Learning => 0.15,
            ConsciousnessState::Optimizing => 0.12,
            ConsciousnessState::Creating => 0.08,
            ConsciousnessState::Reflecting => 0.05,
        };
        
        if let Some(current_score) = state_scores.get_mut(&self.consciousness_state) {
            *current_score += continuity_bonus;
        }
        
        // 基于内在动机的调整
        if self.intrinsic_motivation > 0.8 {
            // 高动机时更倾向于创造和优化
            state_scores.entry(ConsciousnessState::Creating).and_modify(|s| *s *= 1.2);
            state_scores.entry(ConsciousnessState::Optimizing).and_modify(|s| *s *= 1.1);
        }
        
        // 基于好奇心的调整
        if self.curiosity_index > 0.7 {
            // 高好奇心时更倾向于探索和学习
            state_scores.entry(ConsciousnessState::Exploring).and_modify(|s| *s *= 1.3);
            state_scores.entry(ConsciousnessState::Learning).and_modify(|s| *s *= 1.2);
        }
        
        // 基于历史反思的调整
        let recent_reflections: Vec<_> = self.reflection_history.iter()
            .rev()
            .take(5)
            .collect();
        
        if !recent_reflections.is_empty() {
            let avg_impact = recent_reflections.iter()
                .map(|r| r.impact)
                .sum::<f32>() / recent_reflections.len() as f32;
            
            if avg_impact > 0.8 {
                // 高影响力的反思后，倾向于反思状态
                state_scores.entry(ConsciousnessState::Reflecting).and_modify(|s| *s *= 1.4);
            }
        }
        
        // 选择得分最高的状态
        *state_scores.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(state, _score)| state)
            .unwrap_or(&self.consciousness_state)
    }
    
    /// 更新元认知系统
    fn update_metacognition(&mut self, context: &WisdomContext) {
        // 基于当前状态调整认知灵活性
        match self.consciousness_state {
            ConsciousnessState::Exploring | ConsciousnessState::Creating => {
                self.metacognition.cognitive_flexibility = 
                    (self.metacognition.cognitive_flexibility * 0.9 + 0.1).min(1.0);
            },
            ConsciousnessState::Optimizing => {
                self.metacognition.cognitive_flexibility = 
                    (self.metacognition.cognitive_flexibility * 0.95).max(0.3);
            },
            _ => {}
        }
        
        // 基于性能调整学习效率
        let performance_avg = context.performance_vector.iter().sum::<f32>() / context.performance_vector.len() as f32;
        if performance_avg > 0.7 {
            self.metacognition.learning_efficiency = 
                (self.metacognition.learning_efficiency * 0.98 + 0.02).min(1.0);
        } else if performance_avg < 0.4 {
            self.metacognition.learning_efficiency = 
                (self.metacognition.learning_efficiency * 0.95).max(0.2);
        }
    }

    /// 更新自我认知
    fn update_self_awareness(&mut self, context: &WisdomContext) {
        for (i, &value) in context.performance_vector.iter().enumerate() {
            if i < self.self_awareness.len() {
                // 平滑更新自我认知
                self.self_awareness[i] = self.self_awareness[i] * 0.9 + value * 0.1;
            }
        }
    }

    /// 调整动机
    fn adapt_motivation(&mut self, context: &WisdomContext) {
        let motivation_drift = context.challenge_level - 0.5;
        self.intrinsic_motivation = (self.intrinsic_motivation + motivation_drift * 0.1).clamp(0.1, 1.0);
    }

    /// 更新好奇心
    fn update_curiosity(&mut self, context: &WisdomContext) {
        let novelty_factor = context.novelty_score;
        self.curiosity_index = (self.curiosity_index * 0.95 + novelty_factor * 0.05).clamp(0.0, 1.0);
    }

    /// 自我反思
    pub fn self_reflect(&mut self, trigger: &str, insight: &str, confidence: f32) {
        // 使用自省历史长度和内在动机计算时间戳
        let base_timestamp = self.reflection_history.len() as f32 * 10.0;
        let motivation_factor = self.intrinsic_motivation * 100.0;
        let timestamp = base_timestamp + motivation_factor + confidence * 50.0;
        
        let reflection = ReflectionEvent {
            timestamp,
            trigger: trigger.to_string(),
            insight: insight.to_string(),
            confidence,
            impact: confidence * self.intrinsic_motivation,
        };

        self.reflection_history.push_back(reflection);
        if self.reflection_history.len() > 100 {
            self.reflection_history.pop_front();
        }
    }
}

impl MetacognitionSystem {
    pub fn new() -> Self {
        Self {
            strategy_assessment: HashMap::new(),
            bias_detection: Vec::new(),
            learning_efficiency: 0.7,
            cognitive_flexibility: 0.8,
        }
    }

    /// 检测认知偏差
    pub fn detect_biases(&mut self, recent_decisions: &[HandDecision], performance_history: &[f32]) {
        self.bias_detection.clear();
        
        // 检测确认偏误：倾向于选择相同的手
        if recent_decisions.len() >= 5 {
            let left_count = recent_decisions.iter().filter(|&&d| d == HandDecision::Left).count();
            let right_count = recent_decisions.len() - left_count;
            
            let imbalance_ratio = if left_count > right_count {
                left_count as f32 / recent_decisions.len() as f32
            } else {
                right_count as f32 / recent_decisions.len() as f32
            };
            
            if imbalance_ratio > 0.8 {
                self.bias_detection.push(CognitiveBias {
                    bias_type: "确认偏误".to_string(),
                    strength: imbalance_ratio - 0.5,
                    detection_confidence: 0.8,
                });
            }
        }
        
        // 检测锚定效应：过度依赖初始决策
        if performance_history.len() >= 10 {
            let recent_avg = performance_history.iter().rev().take(5).sum::<f32>() / 5.0;
            let early_avg = performance_history.iter().take(5).sum::<f32>() / 5.0;
            
            if (recent_avg - early_avg).abs() > 0.3 {
                self.bias_detection.push(CognitiveBias {
                    bias_type: "锚定效应".to_string(),
                    strength: (recent_avg - early_avg).abs() / 2.0,
                    detection_confidence: 0.7,
                });
            }
        }
        
        // 检测可得性启发：基于最近的成功/失败模式
        if performance_history.len() >= 8 {
            let recent_success_rate = performance_history.iter().rev().take(4)
                .filter(|&&p| p > 0.7).count() as f32 / 4.0;
            let overall_success_rate = performance_history.iter()
                .filter(|&&p| p > 0.7).count() as f32 / performance_history.len() as f32;
            
            if (recent_success_rate - overall_success_rate).abs() > 0.4 {
                self.bias_detection.push(CognitiveBias {
                    bias_type: "可得性启发".to_string(),
                    strength: (recent_success_rate - overall_success_rate).abs(),
                    detection_confidence: 0.75,
                });
            }
        }
    }

    /// 评估学习策略
    pub fn evaluate_strategy(&mut self, strategy: &str, effectiveness: f32) {
        self.strategy_assessment.insert(strategy.to_string(), effectiveness);
    }
}





impl AdaptiveActivation {
    pub fn new() -> Self {
        Self {
            function_type: ActivationType::Adaptive,
            adaptability: 0.8,
            parameters: vec![1.0, 0.5, 0.1],
        }
    }

    pub fn activate(&mut self, x: f32) -> f32 {
        match self.function_type {
            ActivationType::Adaptive => {
                // 自适应激活函数
                let a = self.parameters[0];
                let b = self.parameters[1];
                let c = self.parameters[2];
                
                // 动态调整参数
                self.parameters[0] = a * 0.999 + x.abs() * 0.001;
                
                // 组合激活函数
                (x.tanh() * a + x.relu() * b + x * c) / (a + b + c)
            }
            ActivationType::ReLU => x.relu(),
            ActivationType::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationType::Tanh => x.tanh(),
            ActivationType::GELU => x * 0.5 * (1.0 + (x * 0.70710678).tanh()),
            ActivationType::Swish => x / (1.0 + (-x).exp()),
        }
    }
}

impl NeuronState {
    pub fn new() -> Self {
        // 基于适应性和连接强度计算初始状态
        let adaptation_level = 0.5;
        let connection_strength = 1.0;
        
        // 激活水平基于适应性和连接强度
        let activation = adaptation_level * connection_strength * 0.8;
        
        // 电位基于激活水平的微小变化
        let potential = activation * 0.1;
        
        Self {
            activation,
            potential,
            adaptation_level,
            connection_strength,
            firing_pattern: vec![0.0; 10],
        }
    }
}

impl KnowledgeGraph {
    pub fn new() -> Self {
        Self {
            concepts: HashMap::new(),
            relations: HashMap::new(),
            update_frequency: 1.0,
            abstraction_levels: vec![AbstractionLevel::new(0), AbstractionLevel::new(1)],
        }
    }

    pub fn integrate_experience(&mut self, input: &WisdomInput) {
        // 分析输入特征以确定概念类型和重要性
        let feature_analysis = self.analyze_features(&input.feature_vector);
        
        // 生成有意义的概念ID
        let concept_type = self.determine_concept_type(&input.feature_vector);
        let concept_id = format!("{}_{}_{}", concept_type, 
                                self.concepts.len(), 
                                (feature_analysis.signature * 1000.0) as u32);
        
        // 检查是否与现有概念相似
        let similar_concept = self.find_similar_concept(&input.feature_vector, 0.8);
        
        if let Some((similar_id, similarity)) = similar_concept {
            // 强化现有概念而非创建新概念
            self.reinforce_concept(&similar_id, &input.feature_vector, similarity);
        } else {
            // 创建新概念
            let concept = ConceptNode {
                concept_id: concept_id.clone(),
                embedding: input.feature_vector.clone(),
                importance: feature_analysis.importance,
                activation_level: feature_analysis.activation,
                abstraction_level: self.determine_abstraction_level(&input.feature_vector),
            };

            self.concepts.insert(concept_id.clone(), concept);
            
            // 建立与新概念的关联
            self.establish_concept_relations(&concept_id, &input.feature_vector);
        }
        
        // 更新抽象层级
        self.update_abstraction_levels();
    }
    
    /// 分析特征向量
    fn analyze_features(&self, features: &[f32]) -> FeatureAnalysis {
        let mean = features.iter().sum::<f32>() / features.len() as f32;
        let variance = features.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / features.len() as f32;
        let std_dev = variance.sqrt();
        
        // 计算特征签名（用于识别相似模式）
        let signature = features.iter()
            .enumerate()
            .map(|(i, &x)| x * ((i as f32 + 1.0) * 0.1).sin())
            .sum::<f32>().abs();
        
        // 基于特征统计确定重要性
        let importance = if std_dev > 0.5 {
            (std_dev * 0.7 + mean.abs() * 0.3).clamp(0.3, 1.0)
        } else {
            (mean.abs() * 0.8 + variance * 2.0).clamp(0.2, 0.8)
        };
        
        // 计算激活水平
        let activation = if mean > 0.6 {
            (mean + std_dev * 0.5).clamp(0.5, 1.0)
        } else {
            (mean * 0.7 + std_dev * 0.3).clamp(0.2, 0.8)
        };
        
        FeatureAnalysis {
            importance,
            activation,
            signature,
            complexity: std_dev,
            pattern_type: self.classify_pattern(features),
        }
    }
    
    /// 确定概念类型
    fn determine_concept_type(&self, features: &[f32]) -> &'static str {
        let analysis = self.analyze_features(features);
        
        match analysis.pattern_type {
            PatternType::Sequential => "sequence",
            PatternType::Spatial => "spatial",
            PatternType::Temporal => "temporal",
            PatternType::Complex => "complex",
            PatternType::Simple => "basic",
        }
    }
    
    /// 分类模式类型
    fn classify_pattern(&self, features: &[f32]) -> PatternType {
        if features.len() < 4 {
            return PatternType::Simple;
        }
        
        // 计算连续性
        let continuity_score = features.windows(2)
            .map(|w| 1.0 - (w[1] - w[0]).abs())
            .sum::<f32>() / (features.len() - 1) as f32;
        
        // 计算周期性
        let periodicity_score = if features.len() >= 8 {
            let first_half = &features[..features.len()/2];
            let second_half = &features[features.len()/2..];
            
            let correlation: f32 = first_half.iter()
                .zip(second_half.iter())
                .map(|(a, b)| 1.0 - (a - b).abs())
                .sum::<f32>() / first_half.len() as f32;
            
            correlation
        } else {
            0.0
        };
        
        // 基于分数分类
        if continuity_score > 0.8 {
            PatternType::Sequential
        } else if periodicity_score > 0.7 {
            PatternType::Temporal
        } else if features.iter().map(|x| x.abs()).sum::<f32>() / features.len() as f32 > 0.6 {
            PatternType::Spatial
        } else if continuity_score > 0.5 && periodicity_score > 0.3 {
            PatternType::Complex
        } else {
            PatternType::Simple
        }
    }
    
    /// 查找相似概念
    fn find_similar_concept(&self, features: &[f32], threshold: f32) -> Option<(String, f32)> {
        self.concepts.iter()
            .filter_map(|(id, concept)| {
                let similarity = self.calculate_cosine_similarity(features, &concept.embedding);
                if similarity > threshold {
                    Some((id.clone(), similarity))
                } else {
                    None
                }
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }
    
    /// 计算余弦相似度
    fn calculate_cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// 强化现有概念
    fn reinforce_concept(&mut self, concept_id: &str, new_features: &[f32], similarity: f32) {
        if let Some(concept) = self.concepts.get_mut(concept_id) {
            // 融合新特征
            for (i, &new_feature) in new_features.iter().enumerate() {
                if i < concept.embedding.len() {
                    concept.embedding[i] = concept.embedding[i] * 0.8 + new_feature * 0.2;
                }
            }
            
            // 更新重要性和激活水平
            concept.importance = (concept.importance * 0.9 + similarity * 0.1).clamp(0.0, 1.0);
            concept.activation_level = (concept.activation_level * 0.85 + 0.15).clamp(0.0, 1.0);
        }
    }
    
    /// 确定抽象层级
    fn determine_abstraction_level(&self, features: &[f32]) -> usize {
        let analysis = self.analyze_features(features);
        
        match analysis.pattern_type {
            PatternType::Simple => 0,
            PatternType::Sequential | PatternType::Spatial => 1,
            PatternType::Temporal => 2,
            PatternType::Complex => {
                if analysis.complexity > 0.7 { 3 } else { 2 }
            },
        }
    }
    
    /// 建立概念关联
    fn establish_concept_relations(&mut self, concept_id: &str, features: &[f32]) {
        // 找到最相关的现有概念
        let related_concepts: Vec<_> = self.concepts.iter()
            .filter(|(id, _)| id != &concept_id)
            .map(|(id, concept)| {
                let similarity = self.calculate_cosine_similarity(features, &concept.embedding);
                (id.clone(), similarity)
            })
            .filter(|(_, similarity)| *similarity > 0.3)
            .take(5) // 限制关联数量
            .collect();
        
        // 创建关系边
        for (related_id, similarity) in related_concepts {
            let relation_id = format!("{}_{}_{}", concept_id, related_id, similarity as u32);
            self.relations.insert(relation_id, RelationEdge {
                source: concept_id.to_string(),
                target: related_id,
                relation_type: "similarity".to_string(),
                strength: similarity,
                directionality: 0.5, // 双向关联
            });
        }
    }
    
    /// 更新抽象层级
    fn update_abstraction_levels(&mut self) {
        // 重新计算每个抽象层级的概念
        for level in &mut self.abstraction_levels {
            level.concepts.clear();
        }
        
        // 按抽象层级分组概念
        for (concept_id, concept) in &self.concepts {
            let level_idx = concept.abstraction_level.min(self.abstraction_levels.len() - 1);
            self.abstraction_levels[level_idx].concepts.push(concept_id.clone());
        }
        
        // 更新抽象程度
        for (i, level) in self.abstraction_levels.iter_mut().enumerate() {
            level.abstraction_degree = i as f32 * 0.5;
        }
    }
}

impl CreativityEngine {
    pub fn new() -> Self {
        Self {
            creativity_params: CreativityParams::default(),
            imagination_space: ImaginationSpace::new(),
            innovation_history: VecDeque::new(),
        }
    }

    pub fn generate_insights(&mut self, context: &WisdomContext) -> CreativeInsights {
        let novelty_score = self.creativity_params.originality_drive * context.novelty_score;
        
        CreativeInsights {
            novelty_score,
            creative_ideas: vec!["创新手部分配策略".to_string()],
            insight_confidence: 0.7,
        }
    }
}

impl SelfRegulationSystem {
    pub fn new() -> Self {
        Self {
            emotional_regulation: EmotionalRegulation::new(),
            attention_control: AttentionControl::default(),
            motivation_management: MotivationManagement::default(),
            behavioral_selection: BehavioralSelection::new(),
        }
    }

    pub fn regulate_behavior(&mut self, context: &WisdomContext) {
        self.emotional_regulation.update_emotion(context);
        self.attention_control.adjust_focus(context);
        self.motivation_management.update_motivation(context);
    }
}

// 辅助结构体和实现
#[derive(Debug, Clone)]
pub struct WisdomContext {
    pub complexity_level: f32,
    pub challenge_level: f32,
    pub novelty_score: f32,
    pub performance_vector: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct WisdomInput {
    pub feature_vector: Vec<f32>,
    pub note_sequence: Vec<Note>,
    pub current_hand_states: Vec<HandState>,
}

#[derive(Debug, Clone)]
pub struct HandState {
    pub hand: Hand,
    pub position: Vector2,
    pub fatigue: f32,
}

#[derive(Debug, Clone)]
pub struct WisdomOutput {
    pub decision: HandDecision,
    pub confidence: f32,
    pub reasoning: String,
    pub creativity_score: f32,
    pub consciousness_state: ConsciousnessState,
}

#[derive(Debug, Clone)]
pub struct RawOutput {
    pub activations: Vec<f32>,
    pub raw_confidence: f32,
}

#[derive(Debug, Clone)]
pub struct ConsciousOutput {
    pub decision: HandDecision,
    pub confidence: f32,
    pub reasoning: String,
}

#[derive(Debug, Clone)]
pub struct CreativeInsights {
    pub novelty_score: f32,
    pub creative_ideas: Vec<String>,
    pub insight_confidence: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HandDecision {
    Left,
    Right,
}

// 默认实现
impl Default for SelfLearningParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            adaptation_rate: 0.1,
            plasticity_decay: 0.001,
            memory_consolidation: 0.8,
            forgetting_rate: 0.0001,
        }
    }
}

impl Default for CreativityParams {
    fn default() -> Self {
        Self {
            divergent_thinking: 0.8,
            convergent_thinking: 0.7,
            originality_drive: 0.9,
            synthesis_ability: 0.75,
        }
    }
}

impl ImaginationSpace {
    pub fn new() -> Self {
        Self {
            dimensions: 64,
            exploration_intensity: 0.8,
            novelty_threshold: 0.5,
            conceptual_boundaries: vec![1.0; 64],
        }
    }
}

impl Default for EmotionalRegulation {
    fn default() -> Self {
        Self {
            current_emotion: EmotionalState::Curious,
            emotional_stability: 0.7,
            regulation_strategies: vec![],
        }
    }
}

impl EmotionalRegulation {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_emotion(&mut self, context: &WisdomContext) {
        self.current_emotion = if context.novelty_score > 0.7 {
            EmotionalState::Curious
        } else if context.complexity_level > 0.8 {
            EmotionalState::Focused
        } else {
            EmotionalState::Calm
        };
    }
}



impl BehavioralSelection {
    pub fn new() -> Self {
        Self {
            action_preferences: HashMap::new(),
            risk_tolerance: 0.5,
            exploration_exploitation: 0.7,
            habit_strength: HashMap::new(),
        }
    }
}

impl AbstractionLevel {
    pub fn new(level: usize) -> Self {
        Self {
            level,
            concepts: vec![],
            abstraction_degree: level as f32 * 0.5,
        }
    }
}

impl Default for AttentionControl {
    fn default() -> Self {
        Self {
            focus_level: 0.8,
            attention_span: 0.7,
            distraction_resistance: 0.6,
            selective_attention: 0.75,
        }
    }
}

impl AttentionControl {
    pub fn adjust_focus(&mut self, context: &WisdomContext) {
        self.focus_level = (self.focus_level * 0.9 + context.complexity_level * 0.1).clamp(0.0, 1.0);
    }
}

impl Default for MotivationManagement {
    fn default() -> Self {
        Self {
            intrinsic_drive: 0.9,
            goal_orientation: 0.8,
            persistence_level: 0.7,
            self_efficacy: 0.75,
        }
    }
}

impl MotivationManagement {
    pub fn update_motivation(&mut self, context: &WisdomContext) {
        self.intrinsic_drive = (self.intrinsic_drive * 0.95 + context.novelty_score * 0.05).clamp(0.1, 1.0);
    }
}

impl WisdomContext {
    pub fn from_input(input: &WisdomInput) -> Self {
        // 计算复杂度：基于特征向量的方差
        let mean = input.feature_vector.iter().sum::<f32>() / input.feature_vector.len() as f32;
        let variance = input.feature_vector.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / input.feature_vector.len() as f32;
        let complexity_level = (variance.sqrt() / 2.0).clamp(0.0, 1.0);
        
        // 计算挑战水平：基于音符序列的变化率
        let challenge_level = if input.note_sequence.len() > 1 {
            let time_diffs: Vec<f32> = input.note_sequence.windows(2)
                .map(|w| (w[1].time - w[0].time).abs())
                .collect();
            let avg_diff = time_diffs.iter().sum::<f32>() / time_diffs.len() as f32;
            let diff_variance = time_diffs.iter()
                .map(|x| (x - avg_diff).powi(2))
                .sum::<f32>() / time_diffs.len() as f32;
            (diff_variance.sqrt() / 5.0).clamp(0.0, 1.0)
        } else {
            0.5
        };
        
        // 计算新颖性：基于与历史模式的差异
        let novelty_score = if input.feature_vector.len() > 10 {
            // 使用特征向量的熵作为新颖性指标
            let mut entropy = 0.0;
            for &value in &input.feature_vector {
                if value > 0.0 {
                    entropy -= value * value.ln();
                }
            }
            (entropy / 10.0).clamp(0.0, 1.0)
        } else {
            0.5
        };

        Self {
            complexity_level,
            challenge_level,
            novelty_score,
            performance_vector: input.feature_vector.clone(),
        }
    }
}

// 扩展方法
trait ActivationExt {
    fn relu(self) -> f32;
}

impl ActivationExt for f32 {
    fn relu(self) -> f32 {
        self.max(0.0)
    }
}

impl AutonomousWisdomAI {
    /// 创建自主智慧AI
    pub fn new(rotation: f32) -> Self {
        // 使用旋转角度和系统时间计算出生时间
        let system_time_seed = (rotation * 100.0) as u32;
        let birth_time = system_time_seed as f32 + 1000.0;
        
        let mut wisdom_ai = Self {
            hand_ai: crate::hand::PhiTKAdvancedAI::new(rotation),
            network_feedback: NetworkLearningFeedback::new(),
            hand_system_interface: HandSystemInterface::new(),
            evolution_history: VecDeque::new(),
            current_consciousness: ConsciousnessState::Exploring,
            consciousness: ConsciousnessCore::new(),
            birth_time,
            experience_count: 0,
        };
        
        // 初始化智慧模块的GPU支持
        wisdom_ai.hand_ai.init_gpu_support();
        
        wisdom_ai
    }

    /// 处理音符分配决策 - 直接使用hand.rs中的AI系统
    pub fn process_note_assignment(&mut self, notes: &[Note], _config: &Config, ergonomic_system: &ErgonomicHandSystem) -> Vec<Hand> {
        self.experience_count += 1;

        // 创建智慧输入用于意识层面分析
        let wisdom_input = WisdomInput::from_notes(notes, ergonomic_system);
        
        // 更新意识状态
        let context = WisdomContext::from_input(&wisdom_input);
        self.consciousness.update_consciousness(&context);
        self.current_consciousness = self.determine_optimal_state(&context);

        // 使用hand.rs中的AI系统进行音符分配
        let mut notes_copy = notes.to_vec();
        self.hand_ai.analyze_and_assign(&mut notes_copy, _config, &crate::core::BpmList::default(), 0, 0.0);
        
        // 更新手部系统接口
        self.hand_system_interface.update_from_ergonomic(ergonomic_system);

        // 记录进化事件
        self.record_evolution_event(&wisdom_input);

        notes_copy.iter().map(|n| n.hand).collect()
    }
    
    /// 分析现有分配并提供意识层面的建议（不重新推理）
    pub fn analyze_existing_assignments(&mut self, notes: &[Note], ergonomic_system: &ErgonomicHandSystem) -> Vec<ConsciousnessSuggestion> {
        self.experience_count += 1;

        // 创建智慧输入用于意识层面分析
        let wisdom_input = WisdomInput::from_notes(notes, ergonomic_system);
        
        // 更新意识状态
        let context = WisdomContext::from_input(&wisdom_input);
        self.consciousness.update_consciousness(&context);
        self.current_consciousness = self.determine_optimal_state(&context);

        // 基于意识状态分析现有分配
        let mut suggestions = Vec::new();
        for (i, note) in notes.iter().enumerate() {
            let suggestion = self.generate_consciousness_suggestion(note, i, &context);
            suggestions.push(suggestion);
        }

        // 更新手部系统接口
        self.hand_system_interface.update_from_ergonomic(ergonomic_system);

        // 记录进化事件
        self.record_evolution_event(&wisdom_input);

        suggestions
    }

    /// 确定最优意识状态
    fn determine_optimal_state(&self, context: &WisdomContext) -> ConsciousnessState {
        match context.complexity_level {
            x if x < 0.3 => ConsciousnessState::Exploring,
            x if x < 0.6 => ConsciousnessState::Learning,
            x if x < 0.8 => ConsciousnessState::Optimizing,
            _ => ConsciousnessState::Creating,
        }
    }

    /// 生成意识层面的建议
    fn generate_consciousness_suggestion(&self, note: &Note, note_index: usize, context: &WisdomContext) -> ConsciousnessSuggestion {
        // 获取音符位置信息
        let x_pos = note.object.translation.0.now();
        let y_pos = note.object.translation.1.now();
        
        // 基于位置变化估算速度（如果没有velocity字段）
        let note_velocity = (x_pos.abs() + y_pos.abs()).sqrt();
        
        // 计算位置权重
        let left_preference = (-x_pos).max(0.0);
        let right_preference = x_pos.max(0.0);
        
        // 基于意识状态和音符特征生成建议
        let (suggested_hand, confidence, reasoning) = match self.current_consciousness {
            ConsciousnessState::Creating => {
                // 创造模式：基于音符速度和位置动态分配
                let speed_factor = (note_velocity / 10.0).clamp(0.0, 1.0);
                let creative_weight = if speed_factor > 0.7 {
                    // 高速音符倾向于快速手
                    if left_preference > right_preference { Hand::Left } else { Hand::Right }
                } else {
                    // 中低速音符考虑平衡
                    let balance_factor = (note_index as f32 * 0.1).sin();
                    if balance_factor > 0.0 { Hand::Left } else { Hand::Right }
                };
                
                let base_confidence = (left_preference + right_preference) / 2.0 + speed_factor * 0.3;
                (creative_weight, base_confidence.clamp(0.5, 0.9), 
                 format!("创造性分配(速度:{:.2}, 位置:{:.2})", speed_factor, x_pos))
            },
            ConsciousnessState::Optimizing => {
                // 优化模式：综合位置、速度和人体工程学
                let ergonomic_score = if y_pos < 0.0 {
                    // 下方音符，考虑手腕舒适度
                    left_preference * 1.2 - right_preference * 0.8
                } else {
                    // 上方音符，标准分配
                    right_preference * 1.1 - left_preference * 0.9
                };
                
                let optimized_hand = if ergonomic_score > 0.0 { Hand::Right } else { Hand::Left };
                let confidence = (ergonomic_score.abs() + 0.5).clamp(0.7, 0.95);
                (optimized_hand, confidence, "人体工程学优化".to_string())
            },
            ConsciousnessState::Learning => {
                // 学习模式：基于性能历史调整探索率
                let learning_rate = self.consciousness.metacognition.learning_efficiency;
                let exploration_threshold = (0.3 - learning_rate * 0.2).clamp(0.1, 0.4);
                
                if context.novelty_score > exploration_threshold {
                    // 高新颖性时探索
                    let exploratory_hand = if left_preference > right_preference { Hand::Right } else { Hand::Left };
                    (exploratory_hand, 0.6, "学习探索模式".to_string())
                } else {
                    // 低新颖性时保持当前分配
                    (note.hand, 0.7 + learning_rate * 0.2, "学习巩固模式".to_string())
                }
            },
            ConsciousnessState::Exploring => {
                // 探索模式：基于复杂度的系统探索
                let exploration_intensity = context.complexity_level;
                let pattern_complexity = (note_index as f32 * 0.157).sin(); // 黄金比例
                
                if exploration_intensity > 0.6 {
                    // 高复杂度时交叉探索
                    let cross_hand = if note_index % 3 == 0 { note.hand } else {
                        if note.hand == Hand::Left { Hand::Right } else { Hand::Left }
                    };
                    (cross_hand, 0.5, "复杂度驱动探索".to_string())
                } else {
                    // 低复杂度时模式探索
                    let pattern_hand = if pattern_complexity > 0.0 { Hand::Left } else { Hand::Right };
                    (pattern_hand, 0.4 + exploration_intensity * 0.2, "模式探索".to_string())
                }
            },
            ConsciousnessState::Reflecting => {
                // 反思模式：基于历史性能和上下文反思
                let performance_weight = context.performance_vector.iter().sum::<f32>() / context.performance_vector.len() as f32;
                let reflection_factor = (performance_weight * 2.0 - 1.0).clamp(-1.0, 1.0);
                
                let reflective_hand = if reflection_factor > 0.0 {
                    // 性能好时优化位置分配
                    if x_pos < 0.0 { Hand::Left } else { Hand::Right }
                } else {
                    // 性能差时保守分配
                    note.hand
                };
                
                let confidence = (0.6 + performance_weight.abs() * 0.3).clamp(0.5, 0.85);
                (reflective_hand, confidence, format!("反思模式(性能:{:.2})", performance_weight))
            },
        };
        
        ConsciousnessSuggestion {
            suggested_hand,
            confidence,
            reasoning,
        }
    }

    /// 记录进化事件
    fn record_evolution_event(&mut self, input: &WisdomInput) {
        // 基于经验计数和出生时间计算时间戳
        let base_time = self.birth_time + self.experience_count as f32 * 5.0;
        let consciousness_factor = match self.current_consciousness {
            ConsciousnessState::Exploring => 10.0,
            ConsciousnessState::Learning => 20.0,
            ConsciousnessState::Optimizing => 30.0,
            ConsciousnessState::Creating => 40.0,
            ConsciousnessState::Reflecting => 25.0,
        };
        let timestamp = base_time + consciousness_factor;
        
        let event = EvolutionEvent {
            timestamp,
            evolution_type: format!("意识状态: {:?}", self.current_consciousness),
            complexity_change: input.feature_vector.iter().sum::<f32>() / input.feature_vector.len() as f32,
            new_capabilities: vec![],
        };

        self.evolution_history.push_back(event);
        if self.evolution_history.len() > 1000 {
            self.evolution_history.pop_front();
        }
    }
}

impl HandSystemInterface {
    pub fn new() -> Self {
        Self {
            ergonomic_system: ErgonomicHandSystem::new(),
            performance_metrics: PerformanceMetrics::default(),
            adaptation_history: VecDeque::new(),
        }
    }

    pub fn update_from_ergonomic(&mut self, ergonomic: &ErgonomicHandSystem) {
        self.ergonomic_system = ergonomic.clone();
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            accuracy: 0.8,
            speed: 0.7,
            efficiency: 0.75,
            creativity: 0.6,
            learning_rate: 0.9,
        }
    }
}

impl NetworkLearningFeedback {
    pub fn new() -> Self {
        Self {
            global_knowledge: Arc::new(Mutex::new(GlobalKnowledgePool::new())),
            distributed_nodes: vec![],
            collective_intelligence: CollectiveIntelligence::new(),
            feedback_propagation: FeedbackPropagation::new(),
        }
    }

    pub fn propagate_learning(&self, input: &WisdomInput, output: &WisdomOutput) {
        // 提取关键知识
        let knowledge_extracted = self.extract_knowledge(input, output);
        
        // 更新全局知识池
        if let Ok(mut global_pool) = self.global_knowledge.lock() {
            global_pool.add_knowledge(knowledge_extracted.clone());
        }
        
        // 触发群体智慧涌现
        self.trigger_collective_emergence();
        
        // 传播反馈到分布式节点
        self.distribute_feedback(&knowledge_extracted);
    }
    
    fn extract_knowledge(&self, input: &WisdomInput, output: &WisdomOutput) -> SharedConcept {
        // 基于输入特征和输出特征生成确定性的概念ID
        let feature_hash = input.feature_vector.iter()
            .enumerate()
            .map(|(i, &f)| (f * (i as f32 + 1.0) * 0.1).sin())
            .sum::<f32>().abs() as u64;
        let creativity_factor = (output.creativity_score * 1000.0) as u64;
        let concept_id = format!("wisdom_{}_{}", feature_hash.wrapping_add(creativity_factor), output.confidence as u64);
        
        SharedConcept {
            concept: ConceptNode {
                concept_id: concept_id.clone(),
                embedding: input.feature_vector.clone(),
                importance: output.confidence,
                activation_level: output.creativity_score,
                abstraction_level: match output.consciousness_state {
                    ConsciousnessState::Creating => 3,
                    ConsciousnessState::Optimizing => 2,
                    ConsciousnessState::Learning => 1,
                    _ => 0,
                },
            },
            contributor_nodes: vec!["self".to_string()],
            consensus_level: output.confidence,
            validation_count: 1,
        }
    }
    
    fn trigger_collective_emergence(&self) {
        // 基于知识积累和模式识别触发群体智慧涌现
        if let Ok(global_pool) = self.global_knowledge.lock() {
            let concept_count = global_pool.shared_concepts.len();
            
            // 计算知识密度和质量
            let high_quality_count = global_pool.shared_concepts.values()
                .filter(|c| c.consensus_level > 0.7)
                .count();
            
            // 基于知识密度和质量计算涌现阈值
            let emergence_threshold = (high_quality_count as f32 / concept_count.max(1) as f32) * 10.0;
            
            // 当知识积累和质量达到阈值时触发涌现
            if concept_count > 10 && emergence_threshold > 3.0 {
                // 基于概念数量和质量生成确定性行为ID
                let behavior_hash = (concept_count as u64).wrapping_mul(high_quality_count as u64);
                let behavior_id = format!("emergent_{}", behavior_hash);
                
                let emergent_behavior = EmergentBehavior {
                    behavior_id,
                    pattern: "智慧协同".to_string(),
                    frequency: (high_quality_count as f32 / concept_count as f32).clamp(0.5, 1.0),
                    utility: emergence_threshold.clamp(0.3, 1.0),
                    participants: vec!["self".to_string()],
                };
                
                println!("[智慧涌现] 检测到新的集体智慧行为: {}", emergent_behavior.behavior_id);
            }
        }
    }
    
    fn distribute_feedback(&self, knowledge: &SharedConcept) {
        // 实现知识分发逻辑
        println!("[网络反馈] 分发知识: {}, 置信度: {:.3}", 
                 knowledge.concept.concept_id, 
                 knowledge.consensus_level);
        
        // 基于知识质量和抽象层级确定分发范围
        let distribution_scope = match knowledge.concept.abstraction_level {
            0..=1 => "local",      // 低级概念本地分发
            2 => "regional",       // 中级概念区域分发
            _ => "global",         // 高级概念全局分发
        };
        
        // 计算分发优先级
        let priority_score = knowledge.consensus_level * 
                           knowledge.concept.importance * 
                           (1.0 + knowledge.concept.abstraction_level as f32 * 0.2);
        
        // 基于优先级和分发范围执行分发
        match distribution_scope {
            "local" => {
                // 本地分发：更新相邻节点
                self.update_local_nodes(knowledge, priority_score);
            },
            "regional" => {
                // 区域分发：广播到区域网络
                self.broadcast_to_region(knowledge, priority_score);
            },
            "global" => {
                // 全局分发：同步到全局知识网络
                self.sync_to_global_network(knowledge, priority_score);
            },
            _ => {}
        }
        
        // 更新反馈传播参数
        self.update_propagation_parameters(knowledge, priority_score);
    }
    
    /// 更新本地节点
    fn update_local_nodes(&self, knowledge: &SharedConcept, priority: f32) {
        if let Ok(global_pool) = self.global_knowledge.lock() {
            // 获取当前概念的传播网络连接
            let local_connections = global_pool.propagation_network
                .get(&knowledge.concept.concept_id)
                .map(|conns| conns.clone())
                .unwrap_or_default();
            
            // 计算本地更新影响范围
            let impact_radius = (priority * 3.0).ceil() as usize;
            let affected_nodes = local_connections.iter()
                .take(impact_radius)
                .collect::<Vec<_>>();
            
            println!("[本地分发] 更新 {} 个邻近节点，优先级: {:.3}, 影响半径: {}", 
                     affected_nodes.len(), priority, impact_radius);
            
            // 对每个受影响的节点执行知识更新
            for node_id in &affected_nodes {
                self.update_single_local_node(node_id, knowledge, priority);
            }
            
            // 计算更新效果
            let update_success_rate = self.calculate_local_update_success(&affected_nodes, priority);
            println!("[本地分发] 更新成功率: {:.2}%", update_success_rate * 100.0);
        }
    }
    
    /// 更新单个本地节点
    fn update_single_local_node(&self, node_id: &str, knowledge: &SharedConcept, priority: f32) {
        if let Ok(global_pool) = self.global_knowledge.lock() {
            // 检查节点是否存在
            if let Some(node_concept) = global_pool.shared_concepts.get(node_id) {
                // 计算知识兼容性
                let compatibility = self.calculate_knowledge_compatibility(knowledge, node_concept);
                
                if compatibility > 0.3 {
                    // 兼容性足够高，执行更新
                    let update_strength = priority * compatibility;
                    
                    // 模拟节点更新过程
                    let update_result = self.simulate_node_update(node_id, knowledge, update_strength);
                    
                    if update_result.success {
                println!("[节点更新] {} 成功，强度: {:.3}, 更新置信度: {:.3}, 传播延迟: {:.2}ms", 
                         node_id, update_strength, update_result.updated_confidence, update_result.propagation_delay);
                
                // 记录更新统计
                self.record_update_statistics(node_id, update_strength, update_result.updated_confidence, update_result.propagation_delay);
            } else {
                println!("[节点更新] {} 失败，原因: {}", node_id, update_result.failure_reason);
            }
                } else {
                    println!("[节点更新] {} 跳过，兼容性过低: {:.3}", node_id, compatibility);
                }
            }
        }
    }
    
    /// 计算知识兼容性
    fn calculate_knowledge_compatibility(&self, new_knowledge: &SharedConcept, existing: &SharedConcept) -> f32 {
        // 计算概念相似性
        let concept_similarity = self.calculate_concept_similarity(&new_knowledge.concept, &existing.concept);
        
        // 计算抽象层级兼容性
        let abstraction_diff = (new_knowledge.concept.abstraction_level as i32 - 
                               existing.concept.abstraction_level as i32).abs();
        let abstraction_compatibility = match abstraction_diff {
            0 => 1.0,
            1 => 0.8,
            2 => 0.5,
            _ => 0.2,
        };
        
        // 计算共识水平兼容性
        let consensus_compatibility = 1.0 - (new_knowledge.consensus_level - existing.consensus_level).abs();
        
        // 综合兼容性评分
        (concept_similarity * 0.4 + abstraction_compatibility * 0.3 + consensus_compatibility * 0.3).max(0.0)
    }
    
    /// 模拟节点更新
    fn simulate_node_update(&self, node_id: &str, knowledge: &SharedConcept, strength: f32) -> NodeUpdateResult {
        // 基于强度和节点状态计算更新成功率
        let base_success_rate = 0.7;
        let strength_bonus = strength * 0.3;
        let success_rate = (base_success_rate + strength_bonus).min(0.95);
        
        // 使用确定性算法模拟更新结果
        let node_hash = node_id.chars().map(|c| c as u32).sum::<u32>();
        let knowledge_hash = knowledge.concept.concept_id.chars().map(|c| c as u32).sum::<u32>();
        let combined_hash = node_hash.wrapping_add(knowledge_hash);
        let success_factor = (combined_hash % 100) as f32 / 100.0;
        
        if success_factor < success_rate {
            NodeUpdateResult {
                success: true,
                failure_reason: String::new(),
                updated_confidence: knowledge.consensus_level * strength,
                propagation_delay: (1.0 - strength) * 100.0, // ms
            }
        } else {
            NodeUpdateResult {
                success: false,
                failure_reason: if success_factor < success_rate + 0.1 {
                    "网络延迟".to_string()
                } else if success_factor < success_rate + 0.2 {
                    "节点繁忙".to_string()
                } else {
                    "兼容性不足".to_string()
                },
                updated_confidence: 0.0,
                propagation_delay: 0.0,
            }
        }
    }
    
    /// 记录更新统计
    fn record_update_statistics(&self, node_id: &str, strength: f32, confidence: f32, delay: f32) {
        // 计算更新质量分数
        let quality_score = strength * confidence * (1.0 - delay / 1000.0).max(0.1);
        
        // 根据质量分数调整网络参数
        if quality_score > 0.8 {
            println!("[统计] 节点 {} 高质量更新，质量分数: {:.3}", node_id, quality_score);
        } else if quality_score < 0.3 {
            println!("[统计] 节点 {} 低质量更新，质量分数: {:.3}", node_id, quality_score);
        }
    }
    
    /// 计算本地更新成功率
    fn calculate_local_update_success(&self, affected_nodes: &[&String], priority: f32) -> f32 {
        if affected_nodes.is_empty() {
            return 0.0;
        }
        
        // 基于优先级和节点数量计算成功率
        let base_success = 0.8;
        let node_factor = (1.0 - (affected_nodes.len() as f32 / 10.0)).max(0.3);
        let priority_factor = priority.clamp(0.5, 1.0);
        
        base_success * node_factor * priority_factor
    }
    
    /// 广播到区域网络
    fn broadcast_to_region(&self, knowledge: &SharedConcept, priority: f32) {
        // 确定广播范围
        let broadcast_tier = self.determine_broadcast_tier(priority);
        let region_nodes = self.get_region_nodes(broadcast_tier);
        
        println!("[区域广播] 分发到 {} 级区域网络，{} 个节点，优先级: {:.3}", 
                 broadcast_tier, region_nodes.len(), priority);
        
        // 实现分层广播策略
        let mut successful_broadcasts = 0;
        let mut total_latency = 0.0;
        
        for (node_id, node_info) in &region_nodes {
            let broadcast_result = self.execute_region_broadcast(node_id, node_info.clone(), knowledge, priority);
            
            if broadcast_result.success {
                successful_broadcasts += 1;
                total_latency += broadcast_result.latency;
                
                // 记录成功广播的节点信息
                self.record_successful_broadcast(&broadcast_result.node_id, broadcast_result.latency, priority);
            } else {
                // 记录失败广播的节点信息
                self.record_failed_broadcast(&broadcast_result.node_id, priority);
            }
        }
        
        // 计算广播效果
        let success_rate = successful_broadcasts as f32 / region_nodes.len() as f32;
        let avg_latency = if successful_broadcasts > 0 {
            total_latency / successful_broadcasts as f32
        } else {
            0.0
        };
        
        println!("[区域广播] 完成，成功率: {:.2}%, 平均延迟: {:.2}ms", 
                 success_rate * 100.0, avg_latency);
    }
    
    /// 确定广播层级
    fn determine_broadcast_tier(&self, priority: f32) -> u8 {
        match priority {
            p if p >= 0.9 => 3, // 最高优先级，全网广播
            p if p >= 0.7 => 2, // 高优先级，区域广播
            p if p >= 0.5 => 1, // 中优先级，子区域广播
            _ => 0,             // 低优先级，本地广播
        }
    }
    
    /// 获取区域节点
    fn get_region_nodes(&self, tier: u8) -> Vec<(String, NodeInfo)> {
        // 基于层级生成区域节点列表
        let base_node_count = match tier {
            3 => 50,
            2 => 20,
            1 => 8,
            _ => 3,
        };
        
        let mut nodes = Vec::new();
        for i in 0..base_node_count {
            let node_id = format!("region_node_tier{}_{}", tier, i);
            let node_info = NodeInfo {
                address: format!("192.168.{}.{}", tier, i + 1),
                capacity: 100 - (i % 5) * 10,
                load: (i % 3) as f32 * 0.3,
                reliability: 0.8 + (i % 4) as f32 * 0.05,
            };
            nodes.push((node_id, node_info));
        }
        
        nodes
    }
    
    /// 执行区域广播
    fn execute_region_broadcast(&self, node_id: &str, node_info: NodeInfo, knowledge: &SharedConcept, priority: f32) -> BroadcastResult {
        // 计算广播成功率
        let load_factor = 1.0 - node_info.load;
        let reliability_factor = node_info.reliability;
        let capacity_factor = (node_info.capacity as f32 / 100.0).min(1.0); // 容量影响
        let priority_factor = priority;
        
        let success_probability = load_factor * reliability_factor * capacity_factor * priority_factor;
        
        // 基于节点特征计算确定性结果
        let node_hash = node_id.chars().map(|c| c as u32).sum::<u32>();
        let knowledge_hash = knowledge.concept.concept_id.chars().map(|c| c as u32).sum::<u32>();
        let combined_hash = node_hash.wrapping_add(knowledge_hash);
        let success_factor = (combined_hash % 100) as f32 / 100.0;
        
        let success = success_factor < success_probability;
        let latency = if success {
            // 成功时的延迟基于网络负载、地址距离和容量
            let base_latency = 50.0;
            let load_latency = node_info.load * 200.0;
            let priority_latency = 100.0 - priority * 50.0;
            let capacity_latency = (100.0 - node_info.capacity as f32) * 0.5;
            let address_latency = self.calculate_address_latency(&node_info.address);
            
            base_latency + load_latency + priority_latency + capacity_latency + address_latency
        } else {
            0.0
        };
        
        BroadcastResult {
            success,
            latency,
            node_id: node_id.to_string(),
        }
    }
    
    /// 计算地址延迟（基于地址计算网络距离）
    fn calculate_address_latency(&self, address: &str) -> f32 {
        // 解析IP地址并计算网络延迟
        if let Some(parts) = address.split('.').collect::<Vec<_>>().get(1..=3) {
            let network_delay: f32 = parts.iter()
                .filter_map(|part| part.parse::<u32>().ok())
                .map(|octet| octet as f32)
                .sum();
            
            network_delay * 0.1 // 将网络地址转换为延迟因子
        } else {
            10.0 // 默认延迟
        }
    }
    
    /// 记录成功广播
    fn record_successful_broadcast(&self, node_id: &str, latency: f32, priority: f32) {
        let efficiency = priority / (1.0 + latency / 100.0);
        
        if efficiency > 0.8 {
            println!("[广播统计] 节点 {} 高效广播，延迟: {:.2}ms, 效率: {:.3}", node_id, latency, efficiency);
        } else if efficiency < 0.3 {
            println!("[广播统计] 节点 {} 低效广播，延迟: {:.2}ms, 效率: {:.3}", node_id, latency, efficiency);
        }
    }
    
    /// 记录失败广播
    fn record_failed_broadcast(&self, node_id: &str, priority: f32) {
        println!("[广播统计] 节点 {} 广播失败，优先级: {:.3}", node_id, priority);
    }
    
    /// 同步到全局网络
    fn sync_to_global_network(&self, knowledge: &SharedConcept, priority: f32) {
        // 确定同步策略
        let sync_strategy = self.determine_sync_strategy(knowledge, priority);
        
        println!("[全局同步] 使用 {:?} 策略，优先级: {:.3}", sync_strategy, priority);
        
        // 执行全局同步
        let sync_result = self.execute_global_sync(knowledge, priority, sync_strategy);
        
        // 报告同步结果
        println!("[全局同步] 完成，成功: {}, 覆盖节点: {}, 同步时间: {:.2}ms, 策略: {:?}", 
                 sync_result.success, sync_result.covered_nodes, sync_result.sync_time, sync_result.strategy);
        
        // 记录策略效果
        self.record_sync_strategy_effectiveness(sync_result.strategy, sync_result.success, 
                                               sync_result.covered_nodes, sync_result.sync_time);
        
        // 如果同步成功，触发全局知识更新
        if sync_result.success {
            self.trigger_global_knowledge_update(knowledge, sync_result.covered_nodes);
        }
    }
    
    /// 确定同步策略
    fn determine_sync_strategy(&self, knowledge: &SharedConcept, priority: f32) -> SyncStrategy {
        match (knowledge.concept.abstraction_level, priority) {
            (level, pri) if level >= 3 && pri >= 0.8 => SyncStrategy::Immediate,
            (level, pri) if level >= 2 && pri >= 0.6 => SyncStrategy::Priority,
            (level, pri) if level >= 1 && pri >= 0.4 => SyncStrategy::Batched,
            _ => SyncStrategy::Deferred,
        }
    }
    
    /// 执行全局同步
    fn execute_global_sync(&self, knowledge: &SharedConcept, priority: f32, strategy: SyncStrategy) -> SyncResult {
        let (node_count, base_time) = match strategy {
            SyncStrategy::Immediate => (200, 100.0),
            SyncStrategy::Priority => (100, 200.0),
            SyncStrategy::Batched => (50, 500.0),
            SyncStrategy::Deferred => (20, 1000.0),
        };
        
        // 计算成功率
        let success_rate = match strategy {
            SyncStrategy::Immediate => 0.95,
            SyncStrategy::Priority => 0.85,
            SyncStrategy::Batched => 0.75,
            SyncStrategy::Deferred => 0.65,
        };
        
        // 基于优先级调整成功率
        let adjusted_success_rate = success_rate * priority.clamp(0.5, 1.0);
        
        // 确定性同步结果
        let knowledge_hash = knowledge.concept.concept_id.chars().map(|c| c as u32).sum::<u32>();
        let success_factor = (knowledge_hash % 100) as f32 / 100.0;
        let success = success_factor < adjusted_success_rate;
        
        let covered_nodes = if success {
            (node_count as f32 * adjusted_success_rate) as usize
        } else {
            0
        };
        
        let sync_time = base_time * (1.0 + (1.0 - priority) * 0.5);
        
        SyncResult {
            success,
            covered_nodes,
            sync_time,
            strategy,
        }
    }
    
    /// 记录同步策略效果
    fn record_sync_strategy_effectiveness(&self, strategy: SyncStrategy, success: bool, covered_nodes: usize, sync_time: f32) {
        let effectiveness = if success {
            let coverage_rate = covered_nodes as f32 / 200.0; // 假设全局有200个节点
            let time_efficiency = 1000.0 / sync_time.max(1.0);
            coverage_rate * time_efficiency.min(1.0)
        } else {
            0.0
        };
        
        match strategy {
            SyncStrategy::Immediate => {
                if effectiveness > 0.8 {
                    println!("[策略评估] 立即同步策略高效，效果: {:.3}", effectiveness);
                } else if effectiveness < 0.3 {
                    println!("[策略评估] 立即同步策略低效，效果: {:.3}，建议调整", effectiveness);
                }
            },
            SyncStrategy::Priority => {
                println!("[策略评估] 优先同步策略效果: {:.3}", effectiveness);
            },
            SyncStrategy::Batched => {
                println!("[策略评估] 批量同步策略效果: {:.3}", effectiveness);
            },
            SyncStrategy::Deferred => {
                println!("[策略评估] 延迟同步策略效果: {:.3}", effectiveness);
            },
        }
    }
    
    /// 触发全局知识更新
    fn trigger_global_knowledge_update(&self, knowledge: &SharedConcept, covered_nodes: usize) {
        if let Ok(mut global_pool) = self.global_knowledge.lock() {
            // 计算全局影响力
            let global_impact = covered_nodes as f32 / 200.0; // 假设全局有200个节点
            let impact_multiplier = 1.0 + global_impact * 0.5;
            
            // 更新知识的重要性
            if let Some(existing) = global_pool.shared_concepts.get_mut(&knowledge.concept.concept_id) {
                let old_importance = existing.concept.importance;
                existing.concept.importance = (old_importance * impact_multiplier).min(1.0);
                
                println!("[全局更新] 知识重要性: {:.3} -> {:.3}, 影响力: {:.2}%", 
                         old_importance, existing.concept.importance, global_impact * 100.0);
            }
        }
    }
    
    /// 更新传播参数
    fn update_propagation_parameters(&self, _knowledge: &SharedConcept, priority: f32) {
        // 基于知识分发效果更新传播参数
        let amplification = priority.clamp(0.5, 2.0);
        let damping = 1.0 / (1.0 + priority);
        
        println!("[参数更新] 放大系数: {:.3}, 阻尼系数: {:.3}", amplification, damping);
    }
    
    /// 计算概念相似性
    fn calculate_concept_similarity(&self, a: &ConceptNode, b: &ConceptNode) -> f32 {
        if a.embedding.len() != b.embedding.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.embedding.iter().zip(b.embedding.iter())
            .map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

impl GlobalKnowledgePool {
    pub fn new() -> Self {
        Self {
            shared_concepts: HashMap::new(),
            knowledge_versions: HashMap::new(),
            quality_assessment: HashMap::new(),
            propagation_network: HashMap::new(),
        }
    }
    
    pub fn add_knowledge(&mut self, concept: SharedConcept) {
        let concept_id = concept.concept.concept_id.clone();
        
        // 检查是否已存在相同概念
        if let Some(existing) = self.shared_concepts.get_mut(&concept_id) {
            // 智能合并知识
            GlobalKnowledgePool::merge_concepts(existing, &concept);
            
            // 更新版本
            let version = self.knowledge_versions.get(&concept_id).unwrap_or(&0) + 1;
            self.knowledge_versions.insert(concept_id.clone(), version);
            
            println!("[知识融合] 更新概念: {}, 共识水平: {:.3}, 版本: {}", 
                     concept_id, existing.consensus_level, version);
        } else {
            // 添加新概念
            self.shared_concepts.insert(concept_id.clone(), concept.clone());
            self.knowledge_versions.insert(concept_id.clone(), 1);
            self.quality_assessment.insert(concept_id.clone(), concept.consensus_level);
            
            // 建立智能传播网络连接
            self.establish_intelligent_propagation_links(&concept_id);
            
            println!("[新知识] 添加概念: {}, 重要性: {:.3}", 
                     concept_id, concept.concept.importance);
        }
        
        // 触发网络优化
        self.optimize_propagation_network();
    }
    
    /// 智能合并概念
    fn merge_concepts(existing: &mut SharedConcept, new: &SharedConcept) {
        // 基于贡献者数量和验证次数加权合并
        let total_weight = existing.validation_count as f32 + new.validation_count as f32;
        let existing_weight = existing.validation_count as f32 / total_weight;
        let new_weight = new.validation_count as f32 / total_weight;
        
        // 加权平均更新共识水平
        existing.consensus_level = existing.consensus_level * existing_weight + 
                                  new.consensus_level * new_weight;
        
        // 合并贡献者
        for contributor in &new.contributor_nodes {
            if !existing.contributor_nodes.contains(contributor) {
                existing.contributor_nodes.push(contributor.clone());
            }
        }
        
        // 累加验证次数
        existing.validation_count += new.validation_count;
        
        // 更新概念特征向量
        for (i, &new_val) in new.concept.embedding.iter().enumerate() {
            if i < existing.concept.embedding.len() {
                existing.concept.embedding[i] = existing.concept.embedding[i] * 0.8 + new_val * 0.2;
            }
        }
    }
    
    /// 建立智能传播网络连接
    fn establish_intelligent_propagation_links(&mut self, concept_id: &str) {
        let concept = if let Some(c) = self.shared_concepts.get(concept_id) { c } else { return };
        
        // 基于概念相似性、抽象层级和重要性计算连接权重
        let mut related_concepts: Vec<(String, f32)> = self.shared_concepts.keys()
            .filter(|&id| id != concept_id)
            .filter_map(|id| {
                if let Some(other) = self.shared_concepts.get(id) {
                    let similarity = self.calculate_concept_similarity(&concept.concept, &other.concept);
                    let abstraction_compatibility = self.calculate_abstraction_compatibility(
                        concept.concept.abstraction_level, 
                        other.concept.abstraction_level
                    );
                    let importance_factor = (concept.concept.importance + other.concept.importance) / 2.0;
                    
                    let connection_strength = similarity * abstraction_compatibility * importance_factor;
                    
                    if connection_strength > 0.3 {
                        Some((id.clone(), connection_strength))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        
        // 按连接强度排序并选择前N个
        related_concepts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let max_connections = (concept.concept.abstraction_level + 1).min(5) as usize;
        related_concepts.truncate(max_connections);
        
        // 建立连接
        self.propagation_network.insert(concept_id.to_string(), 
                                      related_concepts.into_iter().map(|(id, _)| id).collect());
    }
    
    /// 计算概念相似性
    fn calculate_concept_similarity(&self, a: &ConceptNode, b: &ConceptNode) -> f32 {
        if a.embedding.len() != b.embedding.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.embedding.iter().zip(b.embedding.iter())
            .map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
    
    /// 计算抽象层级兼容性
    fn calculate_abstraction_compatibility(&self, level_a: usize, level_b: usize) -> f32 {
        let diff = (level_a as i32 - level_b as i32).abs();
        match diff {
            0 => 1.0,      // 相同层级，完全兼容
            1 => 0.8,      // 相邻层级，高度兼容
            2 => 0.5,      // 间隔一层，中度兼容
            _ => 0.2,      // 间隔多层，低度兼容
        }
    }
    
    /// 优化传播网络
    fn optimize_propagation_network(&mut self) {
        // 移除低质量连接
        let mut connections_to_remove = Vec::new();
        
        for (concept_id, connections) in &self.propagation_network {
            for conn_id in connections {
                let should_remove = if let (Some(conn_concept), Some(main_concept)) = 
                    (self.shared_concepts.get(conn_id), self.shared_concepts.get(concept_id)) {
                    let similarity = self.calculate_concept_similarity(&main_concept.concept, &conn_concept.concept);
                    similarity <= 0.2
                } else {
                    true
                };
                
                if should_remove {
                    connections_to_remove.push((concept_id.clone(), conn_id.clone()));
                }
            }
        }
        
        // 移除标记的连接
        for (concept_id, conn_id) in connections_to_remove {
            if let Some(connections) = self.propagation_network.get_mut(&concept_id) {
                connections.retain(|id| id != &conn_id);
            }
        }
        
        // 平衡网络连接度
        self.balance_network_connectivity();
    }
    
    /// 平衡网络连接度
    fn balance_network_connectivity(&mut self) {
        let avg_connections = self.propagation_network.values()
            .map(|conns| conns.len())
            .sum::<usize>() / self.propagation_network.len().max(1);
        
        let mut connections_to_add = Vec::new();
        let mut connections_to_remove = Vec::new();
        
        for (concept_id, connections) in &self.propagation_network {
            if connections.len() > avg_connections * 2 {
                // 连接过多，标记移除较弱的连接
                connections_to_remove.push((concept_id.clone(), avg_connections));
            } else if connections.len() < avg_connections / 2 && connections.len() < 3 {
                // 连接过少，寻找新的连接
                if let Some(concept) = self.shared_concepts.get(concept_id) {
                    let mut candidates: Vec<(String, f32)> = self.shared_concepts.keys()
                        .filter(|&id| id != concept_id && !connections.contains(&id.to_string()))
                        .filter_map(|id| {
                            if let Some(other) = self.shared_concepts.get(id) {
                                let similarity = self.calculate_concept_similarity(&concept.concept, &other.concept);
                                if similarity > 0.1 {
                                    Some((id.clone(), similarity))
                                } else {
                                    None
                                }
                            } else {
                                None
                            }
                        })
                        .collect();
                    
                    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    
                    let needed = avg_connections - connections.len();
                    for (id, _) in candidates.into_iter().take(needed) {
                        connections_to_add.push((concept_id.clone(), id));
                    }
                }
            }
        }
        
        // 执行连接添加
        for (concept_id, new_conn) in connections_to_add {
            if let Some(connections) = self.propagation_network.get_mut(&concept_id) {
                connections.push(new_conn);
            }
        }
        
        // 执行连接移除
        for (concept_id, target_len) in connections_to_remove {
            if let Some(connections) = self.propagation_network.get_mut(&concept_id) {
                connections.truncate(target_len);
            }
        }
    }
    
    pub fn get_high_quality_knowledge(&self) -> Vec<&SharedConcept> {
        self.shared_concepts.values()
            .filter(|concept| concept.consensus_level > 0.7)
            .collect()
    }
    
    pub fn evolve_knowledge(&mut self) {
        // 智能知识进化过程
        let evolution_candidates: Vec<_> = self.shared_concepts.iter()
            .filter(|(_, concept)| {
                concept.validation_count > 5 && 
                concept.consensus_level > 0.8 &&
                concept.concept.abstraction_level < 5
            })
            .collect();
        
        let mut evolved_concepts = Vec::new();
        
        for (concept_id, concept) in evolution_candidates {
            // 检查进化条件
            if self.should_evolve_concept(concept) {
                let evolved = self.evolve_single_concept(concept);
                evolved_concepts.push((concept_id.clone(), evolved));
            }
        }
        
        for (old_id, evolved) in evolved_concepts {
            let new_id = format!("evolved_{}", old_id);
            self.shared_concepts.insert(new_id, evolved);
            println!("[知识进化] 概念 {} 已进化", old_id);
        }
    }
    
    /// 判断概念是否应该进化
    fn should_evolve_concept(&self, concept: &SharedConcept) -> bool {
        // 基于多个因素判断进化条件
        let validation_score = (concept.validation_count as f32 / 20.0).min(1.0);
        let consensus_score = concept.consensus_level;
        let importance_score = concept.concept.importance;
        let network_score = self.calculate_network_importance(concept);
        
        let evolution_score = (validation_score + consensus_score + importance_score + network_score) / 4.0;
        evolution_score > 0.7
    }
    
    /// 计算网络重要性
    fn calculate_network_importance(&self, concept: &SharedConcept) -> f32 {
        let connections = self.propagation_network.get(&concept.concept.concept_id)
            .map(|conns| conns.len())
            .unwrap_or(0);
        
        (connections as f32 / 10.0).min(1.0)
    }
    
    fn evolve_single_concept(&self, original: &SharedConcept) -> SharedConcept {
        let mut evolved_concept = original.clone();
        
        // 智能进化策略
        evolved_concept.concept.abstraction_level = 
            (evolved_concept.concept.abstraction_level + 1).min(5);
            
        // 基于当前重要性调整增长率
        let growth_rate = 1.0 + evolved_concept.concept.importance * 0.2;
        evolved_concept.concept.importance = 
            (evolved_concept.concept.importance * growth_rate).min(1.0);
            
        // 基于共识水平调整增长
        let consensus_growth = 1.0 + evolved_concept.consensus_level * 0.1;
        evolved_concept.consensus_level = 
            (evolved_concept.consensus_level * consensus_growth).min(1.0);
        
        // 增加验证计数
        evolved_concept.validation_count += 2;
            
        evolved_concept
    }
}

impl CollectiveIntelligence {
    pub fn new() -> Self {
        Self {
            emergent_behaviors: vec![],
            swarm_optimization: SwarmOptimization::new(),
            consensus_mechanisms: ConsensusMechanism::new(),
        }
    }
}

impl SwarmOptimization {
    pub fn new() -> Self {
        let mut swarm = Self {
            particle_positions: vec![vec![0.0; 32]; 10],
            global_best: vec![0.0; 32],
            velocity_vectors: vec![vec![0.0; 32]; 10],
            convergence_rate: 0.1,
        };
        
        // 初始化粒子位置
        for i in 0..10 {
            for j in 0..32 {
                swarm.particle_positions[i][j] = (i as f32 / 10.0) * ((j as f32 * 0.1).sin());
            }
        }
        
        swarm
    }
    
    /// 执行粒子群优化算法
    pub fn optimize(&mut self, fitness_function: impl Fn(&[f32]) -> f32, iterations: usize) -> f32 {
        let mut personal_best_positions = self.particle_positions.clone();
        let mut personal_best_fitness = vec![f32::INFINITY; 10];
        let mut global_best_fitness = f32::INFINITY;
        
        // 评估初始位置
        for i in 0..10 {
            let fitness = fitness_function(&self.particle_positions[i]);
            personal_best_fitness[i] = fitness;
            
            if fitness < global_best_fitness {
                global_best_fitness = fitness;
                self.global_best = self.particle_positions[i].clone();
            }
        }
        
        // PSO主循环
        for iter in 0..iterations {
            let w = 0.9 - (iter as f32 / iterations as f32) * 0.4; // 惯性权重线性递减
            let c1 = 2.0; // 个体学习因子
            let c2 = 2.0; // 社会学习因子
            
            for i in 0..10 {
                for j in 0..32 {
                    let r1 = (i * j + iter) as f32 * 0.1; // 确定性随机数
                    let r2 = ((i + 1) * (j + 1) - iter) as f32 * 0.1;
                    
                    // 更新速度
                    self.velocity_vectors[i][j] = w * self.velocity_vectors[i][j] +
                        c1 * r1 * (personal_best_positions[i][j] - self.particle_positions[i][j]) +
                        c2 * r2 * (self.global_best[j] - self.particle_positions[i][j]);
                    
                    // 更新位置
                    self.particle_positions[i][j] += self.velocity_vectors[i][j];
                    
                    // 边界处理
                    self.particle_positions[i][j] = self.particle_positions[i][j].clamp(-1.0, 1.0);
                }
                
                // 评估新位置
                let fitness = fitness_function(&self.particle_positions[i]);
                
                // 更新个体最优
                if fitness < personal_best_fitness[i] {
                    personal_best_fitness[i] = fitness;
                    personal_best_positions[i] = self.particle_positions[i].clone();
                    
                    // 更新全局最优
                    if fitness < global_best_fitness {
                        global_best_fitness = fitness;
                        self.global_best = self.particle_positions[i].clone();
                    }
                }
            }
            
            // 计算收敛率
            let avg_distance: f32 = self.particle_positions.iter()
                .map(|pos| {
                    pos.iter().zip(&self.global_best)
                        .map(|(p, g)| (p - g).powi(2))
                        .sum::<f32>()
                        .sqrt()
                })
                .sum::<f32>() / 10.0;
            
            self.convergence_rate = 1.0 / (1.0 + avg_distance);
            
            // 早停条件
            if self.convergence_rate > 0.95 {
                break;
            }
        }
        
        global_best_fitness
    }
    
    /// 获取群体多样性
    pub fn get_diversity(&self) -> f32 {
        if self.particle_positions.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.particle_positions.len() {
            for j in (i + 1)..self.particle_positions.len() {
                let distance: f32 = self.particle_positions[i].iter()
                    .zip(&self.particle_positions[j])
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                total_distance += distance;
                count += 1;
            }
        }
        
        total_distance / count as f32
    }
}

impl ConsensusMechanism {
    pub fn new() -> Self {
        let mut consensus = Self {
            voting_weights: HashMap::new(),
            consensus_threshold: 0.7,
            proposal_history: VecDeque::new(),
        };
        
        // 初始化节点权重
        consensus.voting_weights.insert("expert".to_string(), 0.8);
        consensus.voting_weights.insert("experienced".to_string(), 0.6);
        consensus.voting_weights.insert("novice".to_string(), 0.4);
        
        consensus
    }
    
    /// 提交提案并投票
    pub fn submit_proposal(&mut self, proposer: &str, content: &str, node_types: &[String]) -> bool {
        let proposal = Proposal {
            proposal_id: format!("prop_{}_{}", proposer, self.proposal_history.len()),
            proposer: proposer.to_string(),
            content: content.to_string(),
            support_votes: 0,
            total_votes: 0,
        };
        
        // 计算加权投票
        let mut total_weight = 0.0;
        let mut support_weight = 0.0;
        
        for node_type in node_types {
            if let Some(&weight) = self.voting_weights.get(node_type) {
                total_weight += weight;
                
                // 基于提案内容和节点类型决定投票倾向
                let vote_probability = self.calculate_vote_probability(content, node_type);
                if vote_probability > 0.5 {
                    support_weight += weight;
                }
            }
        }
        
        let consensus_level = if total_weight > 0.0 {
            support_weight / total_weight
        } else {
            0.0
        };
        
        // 记录提案
        let mut final_proposal = proposal;
        final_proposal.support_votes = (support_weight * 10.0) as u32;
        final_proposal.total_votes = (total_weight * 10.0) as u32;
        
        self.proposal_history.push_back(final_proposal);
        if self.proposal_history.len() > 100 {
            self.proposal_history.pop_front();
        }
        
        // 返回是否达成共识
        consensus_level >= self.consensus_threshold
    }
    
    /// 计算节点对提案的投票概率
    fn calculate_vote_probability(&self, content: &str, node_type: &str) -> f32 {
        let base_probability = match node_type {
            "expert" => 0.7,
            "experienced" => 0.6,
            "novice" => 0.5,
            _ => 0.5,
        };
        
        // 基于内容关键词调整概率
        let content_factor = if content.contains("优化") || content.contains("改进") {
            0.1
        } else if content.contains("创新") || content.contains("新") {
            match node_type {
                "expert" => 0.05,
                "experienced" => 0.1,
                "novice" => 0.15,
                _ => 0.0,
            }
        } else if content.contains("保守") || content.contains("稳定") {
            match node_type {
                "expert" => 0.1,
                "experienced" => 0.05,
                "novice" => -0.05,
                _ => 0.0,
            }
        } else {
            0.0
        };
        
        f32::clamp(base_probability + content_factor, 0.0, 1.0)
    }
    
    /// 动态调整共识阈值
    pub fn adjust_consensus_threshold(&mut self, recent_success_rate: f32) {
        // 成功率高时提高阈值，成功率低时降低阈值
        let adjustment = (recent_success_rate - 0.5) * 0.2;
        self.consensus_threshold = (0.7 + adjustment).clamp(0.5, 0.9);
    }
    
    /// 获取最近的共识趋势
    pub fn get_consensus_trend(&self) -> f32 {
        if self.proposal_history.len() < 5 {
            return 0.5;
        }
        
        let recent_proposals: Vec<_> = self.proposal_history.iter()
            .rev()
            .take(5)
            .collect();
        
        let success_rate = recent_proposals.iter()
            .filter(|p| p.total_votes > 0 && p.support_votes as f32 / p.total_votes as f32 >= self.consensus_threshold)
            .count() as f32 / recent_proposals.len() as f32;
        
        success_rate
    }
}

impl FeedbackPropagation {
    pub fn new() -> Self {
        Self {
            feedback_channels: vec![],
            amplification_factors: HashMap::new(),
            damping_coefficients: HashMap::new(),
        }
    }
}

impl WisdomInput {
    pub fn from_notes(notes: &[Note], ergonomic: &ErgonomicHandSystem) -> Self {
        let feature_vector = notes.iter()
            .take(64)
            .flat_map(|note| vec![note.time, note.object.translation.0.now(), note.object.translation.1.now()])
            .collect();

        let current_hand_states = vec![
            HandState {
                hand: Hand::Left,
                position: ergonomic.left_hand.position,
                fatigue: ergonomic.left_hand.fatigue,
            },
            HandState {
                hand: Hand::Right,
                position: ergonomic.right_hand.position,
                fatigue: ergonomic.right_hand.fatigue,
            },
        ];

        Self {
            feature_vector,
            note_sequence: notes.to_vec(),
            current_hand_states,
        }
    }
}