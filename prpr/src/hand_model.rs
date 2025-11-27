use crate::core::note::Hand;
use crate::core::NoteKind;
use fastrand;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

/// 人体工程学手部模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandModel {
    /// 手部位置 (世界坐标)
    pub position: Vector2,
    /// 手部速度
    pub velocity: Vector2,
    /// 手部加速度
    pub acceleration: Vector2,
    /// 手部朝向角度 (弧度)
    pub rotation: f32,
    /// 手部张开程度 (0.0-1.0)
    pub openness: f32,
    /// 手部疲劳度 (0.0-1.0)
    pub fatigue: f32,
    /// 手部灵活性 (0.0-1.0)
    pub dexterity: f32,
    /// 上次更新时间
    pub last_update_time: f32,
    /// 手部类型
    pub hand_type: Hand,
}

/// 手指模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerModel {
    /// 手指位置 (相对于手部)
    pub position: Vector2,
    /// 手指弯曲角度 (弧度)
    pub bend_angle: f32,
    /// 手指长度
    pub length: f32,
    /// 手指粗细
    pub thickness: f32,
    /// 手指疲劳度
    pub fatigue: f32,
    /// 手指灵活性
    pub dexterity: f32,
    /// 是否正在按下
    pub is_pressed: bool,
    /// 按下时间
    pub press_time: f32,
    /// 手指类型
    pub finger_type: FingerType,
    /// 统计功能
    #[serde(default)]
    pub last_time: f32,
    /// 置信度 (0.0-1.0)
    #[serde(default)]
    pub confidence: f32,
    /// 成功连续记录
    #[serde(default)]
    pub success_streak: u32,
    /// 总操作次数
    #[serde(default)]
    pub total_actions: u32,
    /// 性能评分 (0.0-1.0)
    #[serde(default)]
    pub performance_score: f32,
    /// 是否忙碌
    #[serde(default)]
    pub is_busy: bool,
    /// 忙碌结束时间
    #[serde(default)]
    pub busy_until: f32,
}

/// 手指类型
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum FingerType {
    Thumb,    // 拇指
    Index,    // 食指
    Middle,   // 中指
    Ring,     // 无名指
    Pinky,    // 小指
}

/// 手臂模型
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmModel {
    /// 肩膀位置
    pub shoulder_position: Vector2,
    /// 肘部位置
    pub elbow_position: Vector2,
    /// 手腕位置
    pub wrist_position: Vector2,
    /// 手臂角度
    pub angle: f32,
    /// 手臂长度
    pub length: f32,
    /// 手臂粗细
    pub thickness: f32,
    /// 手臂疲劳度
    pub fatigue: f32,
    /// 手臂力量
    pub strength: f32,
    /// 手臂灵活性
    pub flexibility: f32,
}

/// 游戏模式枚举
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GameMode {
    TwoFinger,   // 2指模式：只使用食指
    FourFinger,  // 4指模式：使用食指和中指
}

/// 人体工程学手部系统
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErgonomicHandSystem {
    /// 左手模型
    pub left_hand: HandModel,
    /// 右手模型
    pub right_hand: HandModel,
    /// 左手手指模型
    pub left_fingers: Vec<FingerModel>,
    /// 右手手指模型
    pub right_fingers: Vec<FingerModel>,
    /// 左臂模型
    pub left_arm: ArmModel,
    /// 右臂模型
    pub right_arm: ArmModel,
    /// 身体中心位置
    pub body_center: Vector2,
    /// 身体倾斜角度
    pub body_tilt: f32,
    /// 游戏难度系数
    pub difficulty_factor: f32,
    /// 游戏模式
    pub game_mode: GameMode,
    /// 智能指纹模式选择器
    pub finger_mode_selector: SmartFingerModeSelector,
    /// 当前谱面时间
    pub current_time: f32,
    /// 最近的音符序列（用于模式分析）
    #[serde(skip)]
    pub recent_notes: Vec<crate::core::Note>,
    /// 指纹模式切换冷却时间
    pub mode_switch_cooldown: f32,
    /// 上次模式切换时间
    pub last_mode_switch_time: f32,
}

/// 碰撞检测结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionResult {
    /// 是否有碰撞
    pub has_collision: bool,
    /// 碰撞的手指索引对
    pub colliding_pairs: Vec<(usize, usize, f32)>,
    /// 最小分离距离（用于位置修正）
    pub min_separation_distance: f32,
}

/// 2D向量
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// 计算向量长度
    pub fn magnitude(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// 计算到另一个点的距离
    pub fn distance_to(&self, other: &Vector2) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    /// 归一化向量
    pub fn normalize(&self) -> Vector2 {
        let mag = self.magnitude();
        if mag > 0.0 {
            Vector2::new(self.x / mag, self.y / mag)
        } else {
            Vector2::new(0.0, 0.0)
        }
    }

    /// 向量加法
    pub fn add(&self, other: &Vector2) -> Vector2 {
        Vector2::new(self.x + other.x, self.y + other.y)
    }

    /// 向量减法
    pub fn subtract(&self, other: &Vector2) -> Vector2 {
        Vector2::new(self.x - other.x, self.y - other.y)
    }

    /// 向量点积
    pub fn dot(&self, other: &Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    /// 向量叉积
    pub fn cross(&self, other: &Vector2) -> f32 {
        self.x * other.y - self.y * other.x
    }
    
    /// 向量标量乘法
    pub fn multiply_scalar(&self, scalar: f32) -> Vector2 {
        Vector2::new(self.x * scalar, self.y * scalar)
    }
}

// 运算符重载
impl std::ops::Sub for Vector2 {
    type Output = Vector2;
    fn sub(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x - other.x, self.y - other.y)
    }
}

impl std::ops::Add<Vector2> for Vector2 {
    type Output = Vector2;
    fn add(self, other: Vector2) -> Vector2 {
        Vector2::new(self.x + other.x, self.y + other.y)
    }
}

impl std::ops::Mul<f32> for Vector2 {
    type Output = Vector2;
    fn mul(self, scalar: f32) -> Vector2 {
        Vector2::new(self.x * scalar, self.y * scalar)
    }
}

impl HandModel {
    pub fn new(position: Vector2, hand_type: Hand) -> Self {
        Self {
            position,
            velocity: Vector2::new(0.0, 0.0),
            acceleration: Vector2::new(0.0, 0.0),
            rotation: 0.0,
            openness: 0.8, // 默认稍微张开
            fatigue: 0.0,
            dexterity: 1.0,
            last_update_time: 0.0,
            hand_type,
        }
    }

    /// 更新手部状态
    pub fn update(&mut self, new_position: Vector2, time: f32) {
        let dt = time - self.last_update_time;
        if dt > 0.0 {
            // 计算速度（基于位置变化和时间间隔）
            let position_change = new_position.subtract(&self.position);
            let new_velocity = position_change.multiply_scalar(1.0 / dt.max(0.016));
            
            // 计算加速度 - 基于速度变化
            let velocity_change = new_velocity.subtract(&self.velocity);
            self.acceleration = velocity_change.multiply_scalar(1.0 / dt.max(0.016));
            self.velocity = new_velocity;
            
            // 更新手部朝向（跟随运动方向）
            if new_velocity.magnitude() > 0.001 {
                self.rotation = new_velocity.y.atan2(new_velocity.x);
            }
            
            self.position = new_position;
            self.last_update_time = time;
            
            // 动态更新手部张开程度（基于速度）
            let speed = self.velocity.magnitude();
            self.openness = (0.7 + speed * 0.3).min(1.0);
            
            // 更新疲劳度（基于运动强度）
            self.fatigue = (self.fatigue + speed * 0.01).min(1.0);
            
            // 恢复机制
            self.fatigue = (self.fatigue - 0.001).max(0.0);
        }
    }

    /// 计算到达目标位置的难度
    pub fn calculate_movement_difficulty(&self, target: &Vector2) -> f32 {
        let distance = self.position.distance_to(target);
        let direction = target.subtract(&self.position).normalize();
        let angle_diff = (self.rotation - direction.dot(&Vector2::new(1.0, 0.0))).abs();
        
        // 考虑距离、角度和疲劳度
        let distance_factor = (distance / 10.0).min(1.0);
        let angle_factor = angle_diff / PI;
        let fatigue_factor = self.fatigue;
        
        (distance_factor + angle_factor + fatigue_factor) / 3.0
    }
}

impl FingerModel {
    pub fn new(position: Vector2, finger_type: FingerType) -> Self {
        let (length, thickness, dexterity) = match finger_type {
            FingerType::Thumb => (0.6, 0.3, 0.7),    // 拇指灵活性中等
            FingerType::Index => (1.0, 0.25, 1.0),  // 食指灵活性最高
            FingerType::Middle => (1.1, 0.28, 0.8), // 中指长度最长但灵活性较低
            FingerType::Ring => (0.95, 0.24, 0.6),  // 无名指灵活性较低
            FingerType::Pinky => (0.8, 0.2, 0.4),   // 小指灵活性最低
        };
        
        Self {
            position,
            bend_angle: 0.0,
            length,
            thickness,
            fatigue: 0.0,
            dexterity,
            is_pressed: false,
            press_time: 0.0,
            finger_type,
            last_time: -1.0,
            confidence: 1.0,
            success_streak: 0,
            total_actions: 0,
            performance_score: 1.0,
            is_busy: false,
            busy_until: -1.0,
        }
    }

    /// 更新手指状态
    pub fn update(&mut self, target_position: Option<Vector2>, time: f32) {
        if let Some(target) = target_position {
            // 计算手指弯曲角度（基于目标距离和动态变化）
            let distance = target.distance_to(&self.position);
            let base_bend_ratio = (distance / self.length).min(1.0);
            
            // 添加时间相关的动态变化
            let dynamic_factor = 0.8 + 0.2 * (time * 0.5 + self.finger_type as i32 as f32 * 0.1).sin();
            let bend_ratio = base_bend_ratio * dynamic_factor;
            
            self.bend_angle = bend_ratio * PI / 2.0; // 最大弯曲90度
        } else if self.is_pressed {
            // 如果正在按下但没有目标位置，按下状态会维持弯曲
            self.bend_angle = PI / 3.0; // 默认按下弯曲60度
        } else {
            // 恢复自然弯曲状态（带时间变化）
            let natural_bend = 0.1 + 0.05 * (time * 0.3).sin();
            self.bend_angle = self.bend_angle * 0.95 + natural_bend * 0.05;
        }
        
        if self.is_pressed {
            // 按下状态的疲劳度累积
            self.press_time += 0.016; // 每帧增加16ms
            self.fatigue = (self.fatigue + 0.002).min(1.0);
            
            // 按下时的动态弯曲
            self.bend_angle = self.bend_angle.max(PI / 4.0); // 最小弯曲45度
        } else {
            // 恢复机制
            self.fatigue = (self.fatigue - 0.001).max(0.0);
            
            // 逐渐恢复弯曲角度到自然状态
            if self.bend_angle > 0.1 {
                self.bend_angle = (self.bend_angle - 0.005).max(0.0);
            }
            
            // 恢复按下状态（如果按下时间过长）
            if self.press_time > 2.0 { // 超过2秒自动恢复
                self.press_time = 0.0;
            }
        }
        
        // 更新灵活性（基于疲劳度，保留基础灵活性）
        let base_dexterity = match self.finger_type {
            FingerType::Thumb => 0.7,
            FingerType::Index => 1.0,
            FingerType::Middle => 0.8,
            FingerType::Ring => 0.6,
            FingerType::Pinky => 0.4,
        };
        self.dexterity = (base_dexterity - self.fatigue * base_dexterity * 0.5).max(0.2);
    }

    /// 更新统计状态（从 FingerState 迁移）
    pub fn update_state(&mut self, new_pos: Vector2, time: f32, success: bool, note_kind: &NoteKind) {
        let time_diff = time - self.last_time;

        if time_diff > 0.001 {
            let distance = new_pos.distance_to(&self.position);
            let new_velocity = (new_pos - self.position).multiply_scalar(1.0 / time_diff);
            // 简化的速度更新
            if self.last_time > 0.0 {
                // 基于时间差的动态速度计算
                let base_velocity = new_velocity.multiply_scalar(0.3);
                self.position = self.position.add(&base_velocity.multiply_scalar(time_diff));
            }
            
            // 疲劳计算
            let base_movement_cost = distance * 0.12;
            let speed_cost = (new_velocity.magnitude() / 10.0).powf(1.5) * 0.08;
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
        self.last_time = time;

        // 信心更新
        self.total_actions += 1;
        if success {
            self.success_streak += 1;
            // 信心增长有上限，避免过于自信
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
        let random_variance = fastrand::f32() * 0.1 - 0.05;
        self.performance_score = (
            recent_success_rate * 0.4 +
                self.confidence * 0.35 +
                (1.0 - self.fatigue) * 0.25 +
                random_variance
        ).clamp(0.1, 0.95);
    }

    /// 清理和验证状态
    pub fn clean(&mut self) {
        if !self.last_time.is_finite() {
            self.last_time = -1.0;
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

    /// 计算按下目标的适合度
    pub fn calculate_suitability(&self, target: &Vector2) -> f32 {
        let distance = self.position.distance_to(target);
        let reach = distance / self.length;
        
        // 考虑距离、疲劳度和灵活性
        let distance_factor = 1.0 - reach.min(1.0);
        let fatigue_factor = 1.0 - self.fatigue;
        let dexterity_factor = self.dexterity;
        
        (distance_factor + fatigue_factor + dexterity_factor) / 3.0
    }
}

impl ArmModel {
    pub fn new(shoulder_position: Vector2, hand_type: Hand) -> Self {
        let (length, thickness) = match hand_type {
            Hand::Left => (3.0, 0.5),
            Hand::Right => (3.0, 0.5),
        };
        
        Self {
            shoulder_position,
            elbow_position: Vector2::new(0.0, 0.0),
            wrist_position: Vector2::new(0.0, 0.0),
            angle: 0.0,
            length,
            thickness,
            fatigue: 0.0,
            strength: 1.0,
            flexibility: 1.0,
        }
    }

    /// 更新手臂状态
    pub fn update(&mut self, hand_position: &Vector2) {
        self.wrist_position = *hand_position;
        
        // 简化的肘部位置计算，带动态变化
        let shoulder_to_hand = hand_position.subtract(&self.shoulder_position);
        let mid_point = self.shoulder_position.add(&Vector2::new(shoulder_to_hand.x / 2.0, shoulder_to_hand.y / 2.0));
        
        // 添加自然的弯曲变化
        let natural_bend = Vector2::new(
            -shoulder_to_hand.y * 0.15 + (self.fatigue * 0.1),
            shoulder_to_hand.x * 0.15 + (self.fatigue * 0.05)
        );
        
        self.elbow_position = Vector2::new(
            mid_point.x + natural_bend.x,
            mid_point.y + natural_bend.y,
        );
        
        // 计算手臂角度 - 使用atan2正确计算弧度角
        if shoulder_to_hand.magnitude() > 0.0 {
            self.angle = shoulder_to_hand.y.atan2(shoulder_to_hand.x);
        }
        
        // 更新手臂疲劳度（基于角度和持续使用）
        let angle_stress = (self.angle.abs() / PI).min(1.0);
        self.fatigue = (self.fatigue + angle_stress * 0.001).min(1.0);
        
        // 恢复机制
        self.fatigue = (self.fatigue - 0.0005).max(0.0);
    }

    /// 计算手臂移动的舒适度
    pub fn calculate_comfort(&self) -> f32 {
        let angle_comfort = 1.0 - (self.angle.abs() / PI).min(1.0);
        let fatigue_factor = 1.0 - self.fatigue;
        let strength_factor = self.strength;
        
        (angle_comfort + fatigue_factor + strength_factor) / 3.0
    }
}

impl ErgonomicHandSystem {
    pub fn new() -> Self {
        let body_center = Vector2::new(0.0, 0.0);
        let left_hand = HandModel::new(Vector2::new(-1.5, 0.0), Hand::Left);
        let right_hand = HandModel::new(Vector2::new(1.5, 0.0), Hand::Right);
        
        // 初始化手指
        let left_fingers = vec![
            FingerModel::new(Vector2::new(-1.7, 0.1), FingerType::Thumb),
            FingerModel::new(Vector2::new(-1.6, 0.3), FingerType::Index),
            FingerModel::new(Vector2::new(-1.5, 0.35), FingerType::Middle),
            FingerModel::new(Vector2::new(-1.4, 0.3), FingerType::Ring),
            FingerModel::new(Vector2::new(-1.3, 0.2), FingerType::Pinky),
        ];
        
        let right_fingers = vec![
            FingerModel::new(Vector2::new(1.3, 0.2), FingerType::Thumb),
            FingerModel::new(Vector2::new(1.4, 0.3), FingerType::Index),
            FingerModel::new(Vector2::new(1.5, 0.35), FingerType::Middle),
            FingerModel::new(Vector2::new(1.6, 0.3), FingerType::Ring),
            FingerModel::new(Vector2::new(1.7, 0.1), FingerType::Pinky),
        ];
        
        // 初始化手臂
        let left_arm = ArmModel::new(Vector2::new(-2.5, 1.0), Hand::Left);
        let right_arm = ArmModel::new(Vector2::new(2.5, 1.0), Hand::Right);
        
        Self {
            left_hand,
            right_hand,
            left_fingers,
            right_fingers,
            left_arm,
            right_arm,
            body_center,
            body_tilt: 0.0,
            difficulty_factor: 1.0,
            game_mode: GameMode::TwoFinger, // 默认2指模式
            finger_mode_selector: SmartFingerModeSelector::new(),
            current_time: 0.0,
            recent_notes: Vec::new(),
            mode_switch_cooldown: 2.0, // 2秒切换冷却
            last_mode_switch_time: -2.0,
        }
    }

    /// 更新整个手部系统
    pub fn update(&mut self, time: f32) {
        // 更新手部状态 - 添加位置变化来计算速度和加速度
        let _previous_left_pos = self.left_hand.position;
        let _previous_right_pos = self.right_hand.position;
        
        // 生成动态的手部位置变化（模拟实际游戏中的移动）
        let left_movement = Vector2::new(
            (time * 0.1).sin() * 0.2,
            (time * 0.15).cos() * 0.1
        );
        let right_movement = Vector2::new(
            (time * 0.12).sin() * 0.2,
            (time * 0.18).cos() * 0.1
        );
        
        let new_left_position = Vector2::new(-1.5, 0.0).add(&left_movement);
        let new_right_position = Vector2::new(1.5, 0.0).add(&right_movement);
        
        // 更新手部状态（计算速度和加速度）
        self.left_hand.update(new_left_position, time);
        self.right_hand.update(new_right_position, time);
        
        // 更新手臂
        self.left_arm.update(&self.left_hand.position);
        self.right_arm.update(&self.right_hand.position);
        
        // 更新手指 - 动态计算目标位置和状态
        for (i, finger) in &mut self.left_fingers.iter_mut().enumerate() {
            // 基于手部位置的动态目标位置
            let base_position = self.left_hand.position.add(&Vector2::new(
                (finger.finger_type as i32 - 2) as f32 * 0.15,
                0.1 + finger.bend_angle * 0.1,
            ));
            
            // 动态变化：添加时间相关的波动
            let dynamic_offset = Vector2::new(
                (time * 0.5 + i as f32 * 0.2).sin() * 0.02,
                (time * 0.3 + i as f32 * 0.1).cos() * 0.01
            );
            
            // 模拟手指状态变化（每3-5秒切换一次按下状态）
            let should_be_pressed = (time / 4.0 + i as f32 * 0.5).sin() > 0.5;
            if should_be_pressed != finger.is_pressed {
                finger.is_pressed = should_be_pressed;
                if should_be_pressed {
                    finger.press_time = time;
                }
            }
            
            // 如果正在按下，目标位置稍微向外延伸
            let target_position = if finger.is_pressed {
                base_position.add(&Vector2::new(0.05, -0.1)).add(&dynamic_offset)
            } else {
                base_position.add(&dynamic_offset)
            };
            
            finger.position = base_position;
            finger.update(Some(target_position), time);
        }
        
        for (i, finger) in &mut self.right_fingers.iter_mut().enumerate() {
            // 基于手部位置的动态目标位置
            let base_position = self.right_hand.position.add(&Vector2::new(
                (finger.finger_type as i32 - 2) as f32 * 0.15,
                0.1 + finger.bend_angle * 0.1,
            ));
            
            // 动态变化：添加时间相关的波动
            let dynamic_offset = Vector2::new(
                (time * 0.4 + i as f32 * 0.3).sin() * 0.02,
                (time * 0.25 + i as f32 * 0.15).cos() * 0.01
            );
            
            // 模拟手指状态变化（每3-5秒切换一次按下状态）
            let should_be_pressed = (time / 3.5 + i as f32 * 0.4).sin() > 0.3;
            if should_be_pressed != finger.is_pressed {
                finger.is_pressed = should_be_pressed;
                if should_be_pressed {
                    finger.press_time = time;
                }
            }
            
            // 如果正在按下，目标位置稍微向外延伸
            let target_position = if finger.is_pressed {
                base_position.add(&Vector2::new(-0.05, -0.1)).add(&dynamic_offset)
            } else {
                base_position.add(&dynamic_offset)
            };
            
            finger.position = base_position;
            finger.update(Some(target_position), time);
        }
        
        // 更新身体中心（模拟轻微的呼吸运动）
        self.body_center = Vector2::new(
            (time * 0.05).sin() * 0.02,
            (time * 0.03).cos() * 0.01
        );
        
        // 更新身体倾斜
        self.body_tilt = (time * 0.02).sin() * 0.05;

        // 检测并解决碰撞（在所有更新完成后进行）
        self.resolve_all_collisions();
    }

    /// 为音符分配最佳手部
    pub fn assign_note_hand(&mut self, note_position: Vector2, note_kind: &NoteKind, time: f32) -> (Hand, usize, f32) {
        // 更新系统状态
        self.update(time);
        
        // 计算左手和右手到达目标的难度
        let left_difficulty = self.calculate_hand_difficulty(&self.left_hand, &note_position, note_kind);
        let right_difficulty = self.calculate_hand_difficulty(&self.right_hand, &note_position, note_kind);
        
        // 选择难度较低的手
        let (selected_hand, base_difficulty) = if left_difficulty < right_difficulty {
            (Hand::Left, left_difficulty)
        } else {
            (Hand::Right, right_difficulty)
        };
        
        // 为选中的手选择最佳手指
        let (finger_index, finger_suitability) = self.select_best_finger(selected_hand, &note_position);
        
        // 计算总体置信度
        let confidence = 1.0 - base_difficulty * finger_suitability;
        
        (selected_hand, finger_index, confidence)
    }

    /// 计算手部到达目标的难度
    pub fn calculate_hand_difficulty(&self, hand: &HandModel, target: &Vector2, note_kind: &NoteKind) -> f32 {
        let movement_difficulty = hand.calculate_movement_difficulty(target);
        let arm_comfort = match hand.hand_type {
            Hand::Left => self.left_arm.calculate_comfort(),
            Hand::Right => self.right_arm.calculate_comfort(),
        };
        
        // 根据音符类型调整难度
        let note_factor = match note_kind {
            NoteKind::Click => 1.0,
            NoteKind::Drag => 1.2,
            NoteKind::Flick => 1.3,
            NoteKind::Hold { .. } => 1.5,
        };
        
        // 综合难度计算
        (movement_difficulty * 0.6 + (1.0 - arm_comfort) * 0.4) * note_factor * self.difficulty_factor
    }

    /// 为指定手选择最佳手指
    fn select_best_finger(&self, hand: Hand, target: &Vector2) -> (usize, f32) {
        let fingers = match hand {
            Hand::Left => &self.left_fingers,
            Hand::Right => &self.right_fingers,
        };
        
        let mut best_index = 0;
        let mut best_suitability = 0.0;
        
        // 根据游戏模式限制可选择的手指
        let valid_indices = match self.game_mode {
            GameMode::TwoFinger => vec![1], // 只选择食指（索引1）
            GameMode::FourFinger => vec![1, 2], // 选择食指和中指（索引1和2）
        };
        
        for &i in &valid_indices {
            if i < fingers.len() {
                let finger = &fingers[i];
                let suitability = finger.calculate_suitability(target);
                if suitability > best_suitability {
                    best_suitability = suitability;
                    best_index = i;
                }
            }
        }
        
        (best_index, best_suitability)
    }

    /// 应用手指按下状态
    pub fn apply_finger_press(&mut self, hand: Hand, finger_index: usize, time: f32) {
        let fingers = match hand {
            Hand::Left => &mut self.left_fingers,
            Hand::Right => &mut self.right_fingers,
        };
        
        if let Some(finger) = fingers.get_mut(finger_index) {
            finger.is_pressed = true;
            finger.press_time = time;
        }
    }

    /// 重置手指状态
    pub fn reset_finger_state(&mut self, hand: Hand, finger_index: usize) {
        let fingers = match hand {
            Hand::Left => &mut self.left_fingers,
            Hand::Right => &mut self.right_fingers,
        };
        
        if let Some(finger) = fingers.get_mut(finger_index) {
            finger.is_pressed = false;
        }
    }
    /// 获取指定手的统计信息
    pub fn get_hand_statistics(&self, hand: Hand) -> (Vec<FingerStatistics>, HandStatistics) {
        let (fingers, hand_model) = match hand {
            Hand::Left => (&self.left_fingers, &self.left_hand),
            Hand::Right => (&self.right_fingers, &self.right_hand),
        };

        let finger_stats: Vec<FingerStatistics> = fingers.iter().map(|finger| FingerStatistics {
            finger_type: finger.finger_type,
            success_streak: finger.success_streak,
            total_actions: finger.total_actions,
            performance_score: finger.performance_score,
            confidence: finger.confidence,
            is_busy: finger.is_busy,
            fatigue: finger.fatigue,
        }).collect();

        let hand_stats = HandStatistics {
            hand_type: hand,
            position: hand_model.position,
            fatigue: hand_model.fatigue,
            dexterity: hand_model.dexterity,
            openness: hand_model.openness,
        };

        (finger_stats, hand_stats)
    }

    /// 应用手指按压状态更新（从 FingerState 迁移）
    pub fn update_finger_state(&mut self, hand: Hand, finger_type: FingerType, new_position: Vector2, time: f32, success: bool, note_kind: &NoteKind) {
        let fingers = match hand {
            Hand::Left => &mut self.left_fingers,
            Hand::Right => &mut self.right_fingers,
        };

        if let Some(finger) = fingers.iter_mut().find(|f| f.finger_type == finger_type) {
            finger.update_state(new_position, time, success, note_kind);
        }
    }

    /// 获取指定手指的性能评分
    pub fn get_finger_performance(&self, hand: Hand, finger_type: FingerType) -> f32 {
        let fingers = match hand {
            Hand::Left => &self.left_fingers,
            Hand::Right => &self.right_fingers,
        };

        if let Some(finger) = fingers.iter().find(|f| f.finger_type == finger_type) {
            finger.performance_score
        } else {
            0.5 // 默认中等性能
        }
    }

    /// 获取指定手指的可用状态
    pub fn is_finger_available(&self, hand: Hand, finger_type: FingerType, current_time: f32) -> bool {
        let fingers = match hand {
            Hand::Left => &self.left_fingers,
            Hand::Right => &self.right_fingers,
        };

        if let Some(finger) = fingers.iter().find(|f| f.finger_type == finger_type) {
            // 根据游戏模式检查手指是否可用
            match self.game_mode {
                GameMode::TwoFinger => {
                    finger_type == FingerType::Index && !finger.is_busy && current_time >= finger.busy_until
                }
                GameMode::FourFinger => {
                    (finger_type == FingerType::Index || finger_type == FingerType::Middle) && 
                    !finger.is_busy && current_time >= finger.busy_until
                }
            }
        } else {
            false
        }
    }

    /// 清理所有手指状态
    pub fn clean_all_finger_states(&mut self) {
        for finger in &mut self.left_fingers {
            finger.clean();
        }
        for finger in &mut self.right_fingers {
            finger.clean();
        }
    }

    /// 基于性能和状态选择最佳手指
    pub fn select_optimal_finger(&self, hand: Hand, target_position: &Vector2, current_time: f32) -> Option<(FingerType, f32)> {
        let fingers = match hand {
            Hand::Left => &self.left_fingers,
            Hand::Right => &self.right_fingers,
        };

        // 根据游戏模式过滤可用的手指
        let available_fingers: Vec<&FingerModel> = fingers.iter().filter(|finger| {
            match self.game_mode {
                GameMode::TwoFinger => finger.finger_type == FingerType::Index,
                GameMode::FourFinger => finger.finger_type == FingerType::Index || finger.finger_type == FingerType::Middle,
            }
        }).filter(|finger| {
            !finger.is_busy && current_time >= finger.busy_until
        }).collect();

        let mut best_finger: Option<(FingerType, f32)> = None;
        let mut best_score = -1.0;

        for finger in &available_fingers {
            let distance_score = 1.0 - (finger.position.distance_to(target_position) / 2.0).min(1.0);
            let performance_score = finger.performance_score;
            let confidence_score = finger.confidence;
            let fatigue_penalty = finger.fatigue * 0.3;

            let combined_score = (distance_score * 0.3 + performance_score * 0.4 + confidence_score * 0.3 - fatigue_penalty).max(0.0);

            if combined_score > best_score {
                best_score = combined_score;
                best_finger = Some((finger.finger_type, combined_score));
            }
        }

        best_finger
    }
    
    /// 基于物理模型判断音符是否成功击中
    pub fn evaluate_note_success(
        &self,
        hand: Hand,
        target_position: &Vector2,
        note_time: f32,
        current_time: f32,
        note_kind: &crate::core::NoteKind,
    ) -> (bool, f32, f32, f32) {
        let hand_model = match hand {
            Hand::Left => &self.left_hand,
            Hand::Right => &self.right_hand,
        };
        
        let arm_model = match hand {
            Hand::Left => &self.left_arm,
            Hand::Right => &self.right_arm,
        };
        
        // 1. 计算位置误差
        let position_error = hand_model.position.distance_to(target_position);
        
        // 2. 计算时间误差
        let timing_error = (current_time - note_time).abs();
        
        // 3. 计算速度因子（太快或太慢都不好）
        let speed = hand_model.velocity.magnitude();
        let optimal_speed = 3.5; // 更严格的最佳速度（降低以强制更慢更精确的动作）
        let speed_error = ((speed - optimal_speed).abs() / optimal_speed).min(1.0);
        
        // 4. 计算各因素的分数
        // 位置分数（距离越小越好）- 更严格的可接受距离
        const REAL_FINGER_REACH: f32 = 0.25; // 真实的单指可达距离
        const COMFORTABLE_REACH: f32 = 0.15; // 舒适的按键距离
        let position_score = if position_error <= COMFORTABLE_REACH {
            // 舒适区域内：完美得分
            1.0
        } else if position_error <= REAL_FINGER_REACH {
            // 可达区域内：线性递减
            1.0 - ((position_error - COMFORTABLE_REACH) / (REAL_FINGER_REACH - COMFORTABLE_REACH)) * 0.5
        } else {
            // 超范围：严重惩罚
            0.0
        };
        
        // 时间分数（基于音符类型有不同的容差）- 更严格的时间要求
        let time_tolerance = match note_kind {
            crate::core::NoteKind::Click => 0.04,   // 40ms（严格）
            crate::core::NoteKind::Hold { .. } => 0.08, // 80ms（严格）
            crate::core::NoteKind::Drag => 0.10,    // 100ms（严格）
            crate::core::NoteKind::Flick => 0.06,   // 60ms（严格）
        };
        let timing_score = if timing_error <= time_tolerance {
            // 精确时间内：优秀得分
            1.0 - (timing_error / time_tolerance) * 0.2
        } else {
            // 超时：严重惩罚
            0.0
        };
        
        // 速度分数 - 更严格的速度控制
        let speed_score = if speed <= optimal_speed * 1.2 {
            // 低于最佳速度：良好
            1.0 - speed_error * 0.3
        } else {
            // 超过最佳速度：严重惩罚
            (1.0 - speed_error) * 0.3
        };
        
        // 5. 计算更严格的疲劳度惩罚
        let fatigue_penalty = hand_model.fatigue * 0.5 + arm_model.fatigue * 0.4; // 增加疲劳惩罚
        
        // 6. 计算灵活性加成 - 降低加成以防止过度依赖灵活性
        let dexterity_bonus = hand_model.dexterity * 0.1; // 降低加成
        
        // 7. 综合计算成功率 - 增加位置和时间权重
        let total_score = (
            position_score * 0.5 +  // 增加位置权重
            timing_score * 0.3 +    // 保持时间权重
            speed_score * 0.05 +    // 降低速度权重
            dexterity_bonus
        ) - fatigue_penalty;
        
        // 8. 判断是否成功（更严格阈值）
        let strict_success_threshold = match note_kind {
            crate::core::NoteKind::Click => 0.8,    // 80%（严格）
            crate::core::NoteKind::Hold { .. } => 0.75, // 75%（严格）
            crate::core::NoteKind::Drag => 0.7,     // 70%（严格）
            crate::core::NoteKind::Flick => 0.78,   // 78%（严格）
        };
        
        let is_successful = total_score >= strict_success_threshold;
        
        // 9. 计算物理置信度（0-1之间）
        let physical_confidence = total_score.clamp(0.0, 1.0);
        
        (is_successful, position_error, timing_error, physical_confidence)
    }

    /// 智能更新指纹模式选择
    pub fn smart_update_finger_mode(&mut self, new_notes: &[crate::core::Note]) {
        // 更新当前时间
        if let Some(last_note) = new_notes.last() {
            self.current_time = last_note.time;
        }
        
        // 更新音符历史
        self.update_note_history(new_notes);
        
        // 检查是否需要切换模式
        if self.current_time - self.last_mode_switch_time >= self.mode_switch_cooldown {
            let optimal_mode = self.finger_mode_selector.select_optimal_finger_mode(
                &self.recent_notes, 
                self.current_time
            );
            
            if optimal_mode != self.game_mode {
                self.game_mode = optimal_mode;
                self.last_mode_switch_time = self.current_time;
                // 重置手指状态以适应新模式
                self.reset_finger_states_for_mode();
            }
        }
    }

    /// 更新音符历史
    fn update_note_history(&mut self, new_notes: &[crate::core::Note]) {
        // 保持最近30秒的音符历史
        let cutoff_time = self.current_time - 30.0;
        
        // 移除过时的音符
        self.recent_notes.retain(|note| note.time >= cutoff_time);
        
        // 添加新音符
        for note in new_notes {
            if note.time >= cutoff_time {
                self.recent_notes.push(note.clone());
            }
        }
        
        // 保持最多1000个音符
        if self.recent_notes.len() > 1000 {
            self.recent_notes.drain(0..self.recent_notes.len() - 1000);
        }
        
        // 按时间排序
        self.recent_notes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    }

    /// 为新模式重置手指状态
    fn reset_finger_states_for_mode(&mut self) {
        match self.game_mode {
            GameMode::TwoFinger => {
                // 2指模式：只使用食指，禁用其他手指
                for finger in &mut self.left_fingers {
                    if finger.finger_type != FingerType::Index {
                        finger.is_busy = false;
                        finger.busy_until = -1.0;
                        finger.is_pressed = false;
                        finger.press_time = 0.0;
                    }
                }
                for finger in &mut self.right_fingers {
                    if finger.finger_type != FingerType::Index {
                        finger.is_busy = false;
                        finger.busy_until = -1.0;
                        finger.is_pressed = false;
                        finger.press_time = 0.0;
                    }
                }
            }
            GameMode::FourFinger => {
                // 4指模式：使用食指和中指
                for finger in &mut self.left_fingers {
                    if finger.finger_type == FingerType::Index || finger.finger_type == FingerType::Middle {
                        finger.is_busy = false;
                        finger.busy_until = -1.0;
                    }
                }
                for finger in &mut self.right_fingers {
                    if finger.finger_type == FingerType::Index || finger.finger_type == FingerType::Middle {
                        finger.is_busy = false;
                        finger.busy_until = -1.0;
                    }
                }
            }
        }
    }

    /// 手动设置游戏模式（覆盖智能选择）
    pub fn set_game_mode(&mut self, mode: GameMode) {
        if mode != self.game_mode {
            self.game_mode = mode;
            self.last_mode_switch_time = self.current_time;
            self.reset_finger_states_for_mode();
        }
    }

    /// 强制更新模式性能（用于训练反馈）
    pub fn update_mode_performance(&mut self, mode: GameMode, success: bool, reward: f32) {
        self.finger_mode_selector.update_mode_performance(mode, success, reward);
    }

    /// 获取当前指纹模式信息
    pub fn get_finger_mode_info(&self) -> FingerModeInfo {
        let _performance = self.finger_mode_selector.mode_performance.get(&self.game_mode)
            .map(|p| (p.success_rate, p.usage_count))
            .unwrap_or((0.5, 0));
        
        FingerModeInfo {
            current_mode: self.game_mode,
            two_finger_performance: self.finger_mode_selector.mode_performance.get(&GameMode::TwoFinger)
                .map(|p| ModeStats {
                    success_rate: p.success_rate,
                    average_reward: p.average_reward,
                    usage_count: p.usage_count,
                }).unwrap_or(ModeStats {
                    success_rate: 0.5,
                    average_reward: 0.0,
                    usage_count: 0,
                }),
            four_finger_performance: self.finger_mode_selector.mode_performance.get(&GameMode::FourFinger)
                .map(|p| ModeStats {
                    success_rate: p.success_rate,
                    average_reward: p.average_reward,
                    usage_count: p.usage_count,
                }).unwrap_or(ModeStats {
                    success_rate: 0.8,
                    average_reward: 0.0,
                    usage_count: 0,
                }),
            recent_notes_count: self.recent_notes.len(),
            time_since_last_switch: self.current_time - self.last_mode_switch_time,
        }
    }

    /// 重置所有性能统计
    pub fn reset_all_performance(&mut self) {
        self.finger_mode_selector.reset_performance();
        self.clean_all_finger_states();
        self.recent_notes.clear();
        self.last_mode_switch_time = -self.mode_switch_cooldown;
    }

    /// 获取当前活跃的手指列表（基于游戏模式）
    pub fn get_active_fingers(&self, _hand: Hand) -> Vec<FingerType> {
        match self.game_mode {
            GameMode::TwoFinger => vec![FingerType::Index],
            GameMode::FourFinger => vec![FingerType::Index, FingerType::Middle],
        }
    }

    /// 检查是否可以执行音符（考虑模式和手指状态）
    pub fn can_execute_note(&self, hand: Hand, current_time: f32) -> bool {
        match self.game_mode {
            GameMode::TwoFinger => {
                // 2指模式：检查食指是否可用
                self.is_finger_available(hand, FingerType::Index, current_time)
            }
            GameMode::FourFinger => {
                // 4指模式：检查食指或中指是否可用
                self.is_finger_available(hand, FingerType::Index, current_time) ||
                self.is_finger_available(hand, FingerType::Middle, current_time)
            }
        }
    }

    /// 检测手指之间的碰撞
    pub fn detect_finger_collisions(&self, hand: Hand) -> CollisionResult {
        let fingers = match hand {
            Hand::Left => &self.left_fingers,
            Hand::Right => &self.right_fingers,
        };

        let mut colliding_pairs = Vec::new();
        let mut min_separation_distance = f32::MAX;

        // 检测每对手指之间的碰撞
        for i in 0..fingers.len() {
            for j in (i + 1)..fingers.len() {
                let finger1 = &fingers[i];
                let finger2 = &fingers[j];

                // 跳过非活跃手指（在当前游戏模式下）
                if !self.is_finger_active_in_mode(finger1.finger_type) ||
                   !self.is_finger_active_in_mode(finger2.finger_type) {
                    continue;
                }

                let distance = finger1.position.distance_to(&finger2.position);
                let min_required_distance = (finger1.thickness + finger2.thickness) * 0.5;
                let separation_distance = distance - min_required_distance;

                if separation_distance < 0.0 {
                    // 发生碰撞，记录碰撞信息
                    colliding_pairs.push((i, j, -separation_distance));
                } else if separation_distance < min_separation_distance {
                    min_separation_distance = separation_distance;
                }
            }
        }

        CollisionResult {
            has_collision: !colliding_pairs.is_empty(),
            colliding_pairs,
            min_separation_distance: if min_separation_distance == f32::MAX {
                0.0
            } else {
                min_separation_distance
            },
        }
    }

    /// 检查手指在当前游戏模式下是否活跃
    fn is_finger_active_in_mode(&self, finger_type: FingerType) -> bool {
        match self.game_mode {
            GameMode::TwoFinger => finger_type == FingerType::Index,
            GameMode::FourFinger => finger_type == FingerType::Index || finger_type == FingerType::Middle,
        }
    }

    /// 修正手指位置以避免碰撞
    pub fn resolve_finger_collisions(&mut self, hand: Hand) {
        let (fingers, game_mode) = match hand {
            Hand::Left => (&mut self.left_fingers, self.game_mode),
            Hand::Right => (&mut self.right_fingers, self.game_mode),
        };

        // 多次迭代以确保完全解决碰撞
        for _iteration in 0..3 {
            let mut any_collision_resolved = false;

            // 使用索引进行迭代，避免借用冲突
            let mut i = 0;
            while i < fingers.len() {
                let mut j = i + 1;
                while j < fingers.len() {
                    // 检查两个手指是否都活跃
                    let finger1_active = match game_mode {
                        GameMode::TwoFinger => fingers[i].finger_type == FingerType::Index,
                        GameMode::FourFinger => {
                            fingers[i].finger_type == FingerType::Index || 
                            fingers[i].finger_type == FingerType::Middle
                        }
                    };
                    
                    let finger2_active = match game_mode {
                        GameMode::TwoFinger => fingers[j].finger_type == FingerType::Index,
                        GameMode::FourFinger => {
                            fingers[j].finger_type == FingerType::Index || 
                            fingers[j].finger_type == FingerType::Middle
                        }
                    };

                    if finger1_active && finger2_active {
                        let (finger1_pos, finger1_thickness, finger1_dexterity) = {
                            let finger = &fingers[i];
                            (finger.position, finger.thickness, finger.dexterity)
                        };

                        let (finger2_pos, finger2_thickness, finger2_dexterity) = {
                            let finger = &fingers[j];
                            (finger.position, finger.thickness, finger.dexterity)
                        };

                        let distance = finger1_pos.distance_to(&finger2_pos);
                        let min_required_distance = (finger1_thickness + finger2_thickness) * 0.5;

                        if distance < min_required_distance && distance > 0.0 {
                            // 计算分离向量
                            let direction = finger2_pos.subtract(&finger1_pos).normalize();
                            let penetration = min_required_distance - distance;
                            
                            // 基于灵活性分配分离距离
                            let total_flexibility = finger1_dexterity + finger2_dexterity;
                            if total_flexibility > 0.0 {
                                let separation1 = penetration * (finger2_dexterity / total_flexibility) * 0.6;
                                let separation2 = penetration * (finger1_dexterity / total_flexibility) * 0.6;

                                // 应用分离
                                let new_pos1 = finger1_pos.subtract(&direction.multiply_scalar(separation1));
                                let new_pos2 = finger2_pos.add(&direction.multiply_scalar(separation2));

                                // 更新位置（需要重新借用）
                                {
                                    let finger1_mut = &mut fingers[i];
                                    finger1_mut.position = new_pos1;
                                }
                                {
                                    let finger2_mut = &mut fingers[j];
                                    finger2_mut.position = new_pos2;
                                }

                                any_collision_resolved = true;
                            }
                        }
                    }
                    j += 1;
                }
                i += 1;
            }

            // 如果没有解决任何碰撞，提前退出
            if !any_collision_resolved {
                break;
            }
        }
    }

    /// 检测并解决手部内部的所有碰撞
    pub fn resolve_all_collisions(&mut self) {
        // 检测并解决左手碰撞
        let left_collision = self.detect_finger_collisions(Hand::Left);
        if left_collision.has_collision {
            self.resolve_finger_collisions(Hand::Left);
        }

        // 检测并解决右手碰撞
        let right_collision = self.detect_finger_collisions(Hand::Right);
        if right_collision.has_collision {
            self.resolve_finger_collisions(Hand::Right);
        }

        // 检测并解决左右手之间的碰撞（如果距离足够近）
        self.resolve_left_right_hand_collisions();
    }

    /// 解决左右手之间的碰撞
    fn resolve_left_right_hand_collisions(&mut self) {
        let min_hand_distance = 1.0; // 最小手部间距

        let left_center = self.calculate_hand_center(&self.left_fingers);
        let right_center = self.calculate_hand_center(&self.right_fingers);

        let hand_distance = left_center.distance_to(&right_center);

        if hand_distance < min_hand_distance {
            // 计算分离向量
            let direction = right_center.subtract(&left_center).normalize();
            let penetration = min_hand_distance - hand_distance;

            // 将两只手分离
            let separation = direction.multiply_scalar(penetration * 0.5);
            self.left_hand.position = self.left_hand.position.subtract(&separation);
            self.right_hand.position = self.right_hand.position.add(&separation);

            // 更新相关手指位置
            for finger in &mut self.left_fingers {
                finger.position = finger.position.subtract(&separation);
            }
            for finger in &mut self.right_fingers {
                finger.position = finger.position.add(&separation);
            }
        }
    }

    /// 计算手部中心位置
    fn calculate_hand_center(&self, fingers: &[FingerModel]) -> Vector2 {
        let active_fingers: Vec<&FingerModel> = fingers.iter()
            .filter(|f| self.is_finger_active_in_mode(f.finger_type))
            .collect();

        if active_fingers.is_empty() {
            return Vector2::new(0.0, 0.0);
        }

        let sum: Vector2 = active_fingers.iter()
            .fold(Vector2::new(0.0, 0.0), |acc, finger| acc.add(&finger.position));

        Vector2::new(sum.x / active_fingers.len() as f32, sum.y / active_fingers.len() as f32)
    }

    /// 获取碰撞检测统计信息
    pub fn get_collision_statistics(&self) -> (usize, usize, f32) {
        let left_collision = self.detect_finger_collisions(Hand::Left);
        let right_collision = self.detect_finger_collisions(Hand::Right);

        let _total_collisions = left_collision.colliding_pairs.len() + right_collision.colliding_pairs.len();
        let left_collisions = left_collision.colliding_pairs.len();
        let right_collisions = right_collision.colliding_pairs.len();

        (left_collisions, right_collisions, (left_collision.min_separation_distance + right_collision.min_separation_distance) / 2.0)
    }
}

/// 指纹模式信息结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerModeInfo {
    pub current_mode: GameMode,
    pub two_finger_performance: ModeStats,
    pub four_finger_performance: ModeStats,
    pub recent_notes_count: usize,
    pub time_since_last_switch: f32,
}

/// 模式统计数据结构体
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeStats {
    pub success_rate: f32,
    pub average_reward: f32,
    pub usage_count: u32,
}

/// 手指统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerStatistics {
    pub finger_type: FingerType,
    pub success_streak: u32,
    pub total_actions: u32,
    pub performance_score: f32,
    pub confidence: f32,
    pub is_busy: bool,
    pub fatigue: f32,
}

/// 手部统计信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandStatistics {
    pub hand_type: Hand,
    pub position: Vector2,
    pub fatigue: f32,
    pub dexterity: f32,
    pub openness: f32,
}

/// 智能指纹模式选择器
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartFingerModeSelector {
    /// 2指模式权重 (优先级最高)
    pub two_finger_weight: f32,
    /// 4指模式权重
    pub four_finger_weight: f32,
    /// 时间间隔分析
    pub timing_analysis_window: f32,
    /// 同时间音符数量阈值
    pub simultaneous_notes_threshold: usize,
    /// 时间差分析窗口
    pub time_difference_analysis: f32,
    /// 性能统计
    pub mode_performance: HashMap<GameMode, ModePerformance>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModePerformance {
    pub success_rate: f32,
    pub average_reward: f32,
    pub usage_count: u32,
    pub last_used: f32,
}

impl SmartFingerModeSelector {
    pub fn new() -> Self {
        let mut mode_performance = HashMap::new();
        mode_performance.insert(GameMode::TwoFinger, ModePerformance {
            success_rate: 0.9, // 降低默认成功率，更现实
            average_reward: 0.0,
            usage_count: 0,
            last_used: -1.0,
        });
        mode_performance.insert(GameMode::FourFinger, ModePerformance {
            success_rate: 0.7, // 4指模式初始成功率较低，需要学习
            average_reward: 0.0,
            usage_count: 0,
            last_used: -1.0,
        });

        Self {
            two_finger_weight: 1.0,  // 2指模式最高权重
            four_finger_weight: 0.6, // 更严格地降低4指模式权重
            timing_analysis_window: 0.4, // 400ms分析窗口，更严格
            simultaneous_notes_threshold: 2, // 更严格的2个音符阈值
            time_difference_analysis: 0.15, // 150ms时间差分析
            mode_performance,
        }
    }

    /// 基于音符特征选择最佳指纹模式
    pub fn select_optimal_finger_mode(
        &mut self,
        notes: &[crate::core::Note],
        current_time: f32,
    ) -> GameMode {
        // 1. 分析时间特征
        let timing_features = self.analyze_timing_features(notes, current_time);
        
        // 2. 计算模式评分
        let two_finger_score = self.calculate_two_finger_score(&timing_features);
        let four_finger_score = self.calculate_four_finger_score(&timing_features);
        
        // 3. 考虑历史性能
        let performance_factor = self.get_performance_factor(GameMode::TwoFinger);
        let two_finger_final = two_finger_score * performance_factor;
        
        let four_performance_factor = self.get_performance_factor(GameMode::FourFinger);
        let four_finger_final = four_finger_score * four_performance_factor;
        
        // 4. 选择最佳模式
        let selected_mode = if two_finger_final >= four_finger_final {
            GameMode::TwoFinger
        } else {
            GameMode::FourFinger
        };
        
        // 5. 更新使用统计
        self.update_usage_stats(selected_mode, current_time);
        
        selected_mode
    }

    /// 分析音符的时间特征
    fn analyze_timing_features(&self, notes: &[crate::core::Note], current_time: f32) -> TimingFeatures {
        let analysis_window_start = current_time - self.timing_analysis_window;
        let analysis_window_end = current_time + self.timing_analysis_window;
        
        let mut window_notes: Vec<_> = notes.iter()
            .filter(|note| note.time >= analysis_window_start && note.time <= analysis_window_end)
            .collect();
        
        // 按时间排序
        window_notes.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
        
        // 分析时间间隔
        let time_intervals = self.calculate_time_intervals(&window_notes);
        
        // 分析同时间音符
        let simultaneous_notes = self.count_simultaneous_notes(&window_notes);
        
        // 分析时间差模式
        let time_difference_pattern = self.analyze_time_differences(&window_notes);
        
        TimingFeatures {
            total_notes: window_notes.len(),
            average_interval: time_intervals.iter().sum::<f32>() / time_intervals.len().max(1) as f32,
            min_interval: time_intervals.iter().copied().fold(f32::INFINITY, f32::min),
            max_interval: time_intervals.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            max_simultaneous: simultaneous_notes,
            time_density: self.calculate_time_density(&window_notes),
            pattern_complexity: time_difference_pattern.complexity_score,
            rhythm_variance: time_difference_pattern.rhythm_variance,
            fast_sequence_ratio: time_difference_pattern.fast_sequence_ratio,
        }
    }

    /// 计算时间间隔
    fn calculate_time_intervals(&self, notes: &[&crate::core::Note]) -> Vec<f32> {
        let mut intervals = Vec::new();
        for i in 1..notes.len() {
            let interval = (notes[i].time - notes[i-1].time).abs();
            if interval > 0.0 && interval < self.timing_analysis_window {
                intervals.push(interval);
            }
        }
        intervals
    }

    /// 统计同时间音符数量
    fn count_simultaneous_notes(&self, notes: &[&crate::core::Note]) -> usize {
        let mut max_simultaneous = 0;
        let tolerance = 0.05; // 50ms容忍度
        
        for note in notes.iter() {
            let mut count = 1;
            for other_note in notes.iter() {
                if note != other_note {
                    let time_diff = (note.time - other_note.time).abs();
                    if time_diff <= tolerance {
                        count += 1;
                    }
                }
            }
            max_simultaneous = max_simultaneous.max(count);
        }
        
        max_simultaneous
    }

    /// 计算时间密度
    fn calculate_time_density(&self, notes: &[&crate::core::Note]) -> f32 {
        if notes.len() < 2 {
            return 0.0;
        }
        
        let time_span = notes.last().unwrap().time - notes.first().unwrap().time;
        if time_span > 0.0 {
            notes.len() as f32 / time_span
        } else {
            0.0
        }
    }

    /// 分析时间差模式
    fn analyze_time_differences(&self, notes: &[&crate::core::Note]) -> TimeDifferencePattern {
        let mut pattern_complexity = 0.0;
        let mut rhythm_variance = 0.0;
        let mut fast_sequences = 0;
        
        if notes.len() >= 3 {
            for i in 2..notes.len() {
                let interval1 = notes[i-1].time - notes[i-2].time;
                let interval2 = notes[i].time - notes[i-1].time;
                
                // 节奏变化程度
                let variance = (interval1 - interval2).abs();
                rhythm_variance += variance;
                
                // 模式复杂度（基于节奏变化）
                pattern_complexity += (variance / 0.1).min(1.0);
                
                // 快速序列检测（间隔 < 150ms）
                if interval1 < 0.15 && interval2 < 0.15 {
                    fast_sequences += 1;
                }
            }
        }
        
        let total_intervals = (notes.len() - 1).max(1);
        TimeDifferencePattern {
            complexity_score: (pattern_complexity / total_intervals as f32).min(1.0),
            rhythm_variance: rhythm_variance / total_intervals as f32,
            fast_sequence_ratio: fast_sequences as f32 / total_intervals as f32,
        }
    }

    /// 计算2指模式评分
    fn calculate_two_finger_score(&self, features: &TimingFeatures) -> f32 {
        let mut score = self.two_finger_weight;
        
        // 更严格的2指模式要求
        // 2指模式适合简单、间隔较大的谱面 - 提高要求
        if features.min_interval > 0.4 {
            score += 0.25; // 稍微降低长间隔奖励，但仍然优选
        } else if features.min_interval < 0.15 {
            score -= 0.3; // 太密集的谱面严重不适合2指
        }
        
        // 基于平均间隔的评估 - 更严格的间隔要求
        if features.average_interval > 0.6 {
            score += 0.2; // 优秀的长间隔奖励
        } else if features.average_interval > 0.3 {
            score += 0.1; // 可接受的间隔
        } else if features.average_interval < 0.2 {
            score -= 0.2; // 太密集的谱面不适合2指
        } else {
            score -= 0.1; // 一般密集度减分
        }
        
        // 基于最大间隔的评估 - 更严格的间隔控制
        if features.max_interval > 1.5 {
            score -= 0.1; // 间隔过大可能难以控制
        }
        
        // 2指模式严格限制同时多音符
        if features.max_simultaneous == 1 {
            score += 0.3; // 单一音符奖励
        } else if features.max_simultaneous >= 3 {
            score -= 0.4; // 多个同时音符严重不适合2指
        } else {
            score -= 0.2; // 2个同时音符减分
        }
        
        // 更严格的时间密度控制
        if features.time_density < 1.5 {
            score += 0.25; // 极低密度奖励
        } else if features.time_density < 3.0 {
            score += 0.1; // 低密度奖励
        } else if features.time_density > 5.0 {
            score -= 0.3; // 高密度严重减分
        } else {
            score -= 0.1; // 中等密度减分
        }
        
        // 更严格的节奏稳定性要求
        if features.rhythm_variance < 0.15 {
            score += 0.2; // 极稳定节奏奖励
        } else if features.rhythm_variance < 0.3 {
            score += 0.1; // 稳定节奏奖励
        } else if features.rhythm_variance > 0.7 {
            score -= 0.3; // 节奏变化太大严重减分
        } else {
            score -= 0.1; // 一般节奏变化减分
        }
        
        // 更严格的模式复杂度要求
        if features.pattern_complexity < 0.2 {
            score += 0.2; // 极简单模式奖励
        } else if features.pattern_complexity < 0.4 {
            score += 0.05; // 简单模式轻微奖励
        } else if features.pattern_complexity > 0.6 {
            score -= 0.2; // 复杂模式减分
        } else {
            score -= 0.05; // 一般复杂模式轻微减分
        }
        
        // 基于快速序列的严格控制 - 2指不适合快速序列
        if features.fast_sequence_ratio < 0.1 {
            score += 0.15; // 很少快速序列奖励
        } else if features.fast_sequence_ratio > 0.3 {
            score -= 0.3; // 快速序列太多严重减分
        } else {
            score -= 0.1; // 有一些快速序列减分
        }
        
        // 更严格的音符数量控制
        if features.total_notes < 15 {
            score += 0.15; // 少音符奖励
        } else if features.total_notes < 30 {
            score += 0.05; // 适中音符轻微奖励
        } else if features.total_notes > 80 {
            score -= 0.2; // 多音符减分
        } else {
            score -= 0.05; // 一般数量轻微减分
        }
        
        score
    }

    /// 计算4指模式评分
    fn calculate_four_finger_score(&self, features: &TimingFeatures) -> f32 {
        let mut score = self.four_finger_weight;
        
        // 4指模式适合较复杂的谱面
        if features.max_simultaneous > 1 {
            score += 0.2; // 多音符支持
        }
        
        // 基于平均间隔的评估 - 中等间隔最适合4指
        if features.average_interval >= 0.2 && features.average_interval <= 0.6 {
            score += 0.15;
        } else if features.average_interval < 0.1 {
            score += 0.2; // 极密集的音符序列4指表现更好
        }
        
        // 基于最大间隔的适应性 - 4指能处理更大的间隔变化
        if features.max_interval > 2.0 && features.min_interval < 0.1 {
            score += 0.1; // 间隔变化大，4指更灵活
        }
        
        // 中等密度谱面适合4指
        if features.time_density >= 2.0 && features.time_density <= 8.0 {
            score += 0.25;
        }
        
        // 复杂节奏支持 - 基于节奏方差
        if features.rhythm_variance >= 0.2 && features.rhythm_variance <= 0.7 {
            score += 0.2;
        }
        
        // 复杂节奏支持
        if features.pattern_complexity >= 0.3 && features.pattern_complexity <= 0.7 {
            score += 0.3;
        }
        
        // 快速序列支持
        if features.fast_sequence_ratio > 0.3 {
            score += 0.15;
        }
        
        // 基于总音符数的评估 - 4指更适合处理大量音符
        if features.total_notes >= 50 {
            score += 0.1;
        } else if features.total_notes < 10 {
            score -= 0.05; // 音符太少时2指更合适
        }
        
        score
    }

    /// 获取历史性能因子
    fn get_performance_factor(&self, mode: GameMode) -> f32 {
        if let Some(performance) = self.mode_performance.get(&mode) {
            performance.success_rate
        } else {
            0.5
        }
    }

    /// 更新使用统计
    fn update_usage_stats(&mut self, mode: GameMode, current_time: f32) {
        if let Some(performance) = self.mode_performance.get_mut(&mode) {
            performance.usage_count += 1;
            performance.last_used = current_time;
        }
    }

    /// 更新模式性能
    pub fn update_mode_performance(&mut self, mode: GameMode, success: bool, reward: f32) {
        if let Some(performance) = self.mode_performance.get_mut(&mode) {
            // 使用指数移动平均更新成功率
            let alpha = 0.1;
            let success_rate = if success { 1.0 } else { 0.0 };
            performance.success_rate = performance.success_rate * (1.0 - alpha) + success_rate * alpha;
            
            // 更新平均奖励
            performance.average_reward = performance.average_reward * 0.95 + reward * 0.05;
        }
    }

    /// 重置性能统计
    pub fn reset_performance(&mut self) {
        for performance in self.mode_performance.values_mut() {
            performance.success_rate = 0.5;
            performance.average_reward = 0.0;
            performance.usage_count = 0;
        }
    }
}

impl Default for SmartFingerModeSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// 时间特征结构体
#[derive(Debug, Clone)]
struct TimingFeatures {
    total_notes: usize,
    average_interval: f32,
    min_interval: f32,
    max_interval: f32,
    max_simultaneous: usize,
    time_density: f32,
    pattern_complexity: f32,
    rhythm_variance: f32,
    fast_sequence_ratio: f32,
}

/// 时间差模式结构体
#[derive(Debug, Clone)]
struct TimeDifferencePattern {
    complexity_score: f32,
    rhythm_variance: f32,
    fast_sequence_ratio: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_operations() {
        let v1 = Vector2::new(3.0, 4.0);
        let v2 = Vector2::new(1.0, 2.0);
        
        assert_eq!(v1.magnitude(), 5.0);
        assert_eq!(v1.distance_to(&v2), (8.0f32).sqrt());
        assert_eq!(v1.dot(&v2), 11.0);
    }

    #[test]
    fn test_hand_model() {
        let mut hand = HandModel::new(Vector2::new(0.0, 0.0), Hand::Left);
        let target = Vector2::new(1.0, 1.0);
        let difficulty = hand.calculate_movement_difficulty(&target);
        assert!(difficulty >= 0.0 && difficulty <= 1.0);
    }

    #[test]
    fn test_finger_model() {
        let mut finger = FingerModel::new(Vector2::new(0.0, 0.0), FingerType::Index);
        let target = Vector2::new(0.5, 0.0);
        let suitability = finger.calculate_suitability(&target);
        assert!(suitability >= 0.0 && suitability <= 1.0);
    }

    #[test]
    fn test_hand_assignment() {
        let mut system = ErgonomicHandSystem::new();
        let note_pos = Vector2::new(1.0, 0.0);
        let note_kind = NoteKind::Click;
        let (hand, finger_index, confidence) = system.assign_note_hand(note_pos, &note_kind, 0.0);
        
        assert!(hand == Hand::Left || hand == Hand::Right);
        assert!(finger_index < 5);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }

    #[test]
    fn test_smart_finger_mode_selector() {
        let mut selector = SmartFingerModeSelector::new();
        
        // 创建测试音符序列 - 简单谱面
        let notes = vec![
            crate::core::Note {
                time: 0.0,
                speed: 1.0,
                height: 1.0,
                kind: NoteKind::Click,
                judge: crate::judge::JudgeStatus::None,
                hand: Hand::Left,
                object: crate::core::Object::default(),
                above: false,
                multiple_hint: false,
                fake: false,
                end_speed: 1.0,
                start_height: 1.0,
                format: None,
            },
            crate::core::Note {
                time: 0.8, // 较大间隔
                speed: 1.0,
                height: 1.0,
                kind: NoteKind::Click,
                judge: crate::judge::JudgeStatus::None,
                hand: Hand::Right,
                object: crate::core::Object::default(),
                above: false,
                multiple_hint: false,
                fake: false,
                end_speed: 1.0,
                start_height: 1.0,
                format: None,
            },
        ];
        
        // 测试模式选择 - 应该选择2指模式
        let selected_mode = selector.select_optimal_finger_mode(&notes, 0.4);
        assert_eq!(selected_mode, GameMode::TwoFinger);
        
        // 创建复杂谱面测试
        let complex_notes = vec![
            crate::core::Note {
                time: 0.0,
                speed: 1.0,
                height: 1.0,
                kind: NoteKind::Click,
                judge: crate::judge::JudgeStatus::None,
                hand: Hand::Left,
                object: crate::core::Object::default(),
                above: false,
                multiple_hint: false,
                fake: false,
                end_speed: 1.0,
                start_height: 1.0,
                format: None,
            },
            crate::core::Note {
                time: 0.02, // 很小的间隔
                speed: 1.0,
                height: 1.0,
                kind: NoteKind::Click,
                judge: crate::judge::JudgeStatus::None,
                hand: Hand::Right,
                object: crate::core::Object::default(),
                above: false,
                multiple_hint: false,
                fake: false,
                end_speed: 1.0,
                start_height: 1.0,
                format: None,
            },
            crate::core::Note {
                time: 0.03, // 很小的间隔
                speed: 1.0,
                height: 1.0,
                kind: NoteKind::Click,
                judge: crate::judge::JudgeStatus::None,
                hand: Hand::Left,
                object: crate::core::Object::default(),
                above: false,
                multiple_hint: false,
                fake: false,
                end_speed: 1.0,
                start_height: 1.0,
                format: None,
            },
        ];
        
        // 复杂谱面可能选择4指模式
        let complex_mode = selector.select_optimal_finger_mode(&complex_notes, 0.02);
        // 复杂谱面应该优先考虑4指模式
        assert!(complex_mode == GameMode::TwoFinger || complex_mode == GameMode::FourFinger);
    }

    #[test]
    fn test_finger_mode_system_integration() {
        let mut system = ErgonomicHandSystem::new();
        
        // 测试初始状态
        assert_eq!(system.game_mode, GameMode::TwoFinger);
        let mode_info = system.get_finger_mode_info();
        assert_eq!(mode_info.current_mode, GameMode::TwoFinger);
        
        // 测试模式切换功能
        system.set_game_mode(GameMode::FourFinger);
        assert_eq!(system.game_mode, GameMode::FourFinger);
        
        // 测试活跃手指获取
        let active_fingers = system.get_active_fingers(Hand::Left);
        assert_eq!(active_fingers.len(), 2); // 4指模式有2个活跃手指
        assert!(active_fingers.contains(&FingerType::Index));
        assert!(active_fingers.contains(&FingerType::Middle));
        
        // 测试音符执行检查
        assert!(system.can_execute_note(Hand::Left, 0.0));
        assert!(system.can_execute_note(Hand::Right, 0.0));
    }

    #[test]
    fn test_two_finger_vs_four_finger_weights() {
        let selector = SmartFingerModeSelector::new();
        
        // 验证2指模式权重更高
        assert!(selector.two_finger_weight > selector.four_finger_weight);
        
        // 验证配置参数
        assert!(selector.timing_analysis_window > 0.0);
        assert!(selector.simultaneous_notes_threshold > 0);
        assert!(selector.time_difference_analysis > 0.0);
    }

    #[test]
    fn test_finger_collision_detection() {
        let mut system = ErgonomicHandSystem::new();

        // 模拟手指碰撞情况：将两个手指放在非常接近的位置
        if let Some(finger1) = system.left_fingers.get_mut(1) {
            if let Some(finger2) = system.left_fingers.get_mut(2) {
                // 食指和中指放在几乎相同的位置（应该触发碰撞）
                finger1.position = Vector2::new(0.0, 0.0);
                finger2.position = Vector2::new(0.05, 0.0); // 很小的间距
                
                // 临时设置较小的厚度以确保触发碰撞
                finger1.thickness = 0.5;
                finger2.thickness = 0.5;
            }
        }

        // 检测碰撞
        let collision_result = system.detect_finger_collisions(Hand::Left);
        
        // 应该有碰撞发生
        assert!(collision_result.has_collision);
        assert!(!collision_result.colliding_pairs.is_empty());
        assert!(collision_result.min_separation_distance >= 0.0);
    }

    #[test]
    fn test_finger_collision_resolution() {
        let mut system = ErgonomicHandSystem::new();

        // 设置碰撞的手指位置
        if let Some(finger1) = system.left_fingers.get_mut(1) {
            if let Some(finger2) = system.left_fingers.get_mut(2) {
                finger1.position = Vector2::new(0.0, 0.0);
                finger2.position = Vector2::new(0.02, 0.0); // 很接近，会碰撞
                finger1.thickness = 0.3;
                finger2.thickness = 0.3;
            }
        }

        // 解决碰撞
        system.resolve_finger_collisions(Hand::Left);

        // 验证碰撞是否被解决
        let collision_result = system.detect_finger_collisions(Hand::Left);
        assert!(!collision_result.has_collision || collision_result.min_separation_distance > 0.01);
    }

    #[test]
    fn test_all_collisions_resolution() {
        let mut system = ErgonomicHandSystem::new();

        // 制造复杂的碰撞情况
        system.left_fingers[1].position = Vector2::new(0.0, 0.0);
        system.left_fingers[2].position = Vector2::new(0.05, 0.01);
        system.right_fingers[1].position = Vector2::new(1.0, 0.0);
        system.right_fingers[2].position = Vector2::new(1.02, 0.01);

        // 增加厚度以确保碰撞
        for finger in &mut system.left_fingers {
            finger.thickness = 0.4;
        }
        for finger in &mut system.right_fingers {
            finger.thickness = 0.4;
        }

        // 解决所有碰撞
        system.resolve_all_collisions();

        // 获取碰撞统计
        let (left_collisions, right_collisions, avg_separation) = system.get_collision_statistics();
        
        // 验证结果合理
        assert!(left_collisions <= 10); // 应该没有或很少碰撞
        assert!(right_collisions <= 10); // 应该没有或很少碰撞
        assert!(avg_separation >= 0.0);
    }

    #[test]
    fn test_hand_center_calculation() {
        let system = ErgonomicHandSystem::new();

        // 测试手部中心计算
        let left_center = system.calculate_hand_center(&system.left_fingers);
        let right_center = system.calculate_hand_center(&system.right_fingers);

        // 中心应该在合理范围内
        assert!(left_center.x.abs() < 5.0);
        assert!(left_center.y.abs() < 5.0);
        assert!(right_center.x.abs() < 5.0);
        assert!(right_center.y.abs() < 5.0);
    }

    #[test]
    fn test_active_finger_filtering() {
        let mut system = ErgonomicHandSystem::new();

        // 测试2指模式下的活跃手指过滤
        system.game_mode = GameMode::TwoFinger;
        assert!(system.is_finger_active_in_mode(FingerType::Index));
        assert!(!system.is_finger_active_in_mode(FingerType::Middle));
        assert!(!system.is_finger_active_in_mode(FingerType::Thumb));

        // 测试4指模式下的活跃手指过滤
        system.game_mode = GameMode::FourFinger;
        assert!(system.is_finger_active_in_mode(FingerType::Index));
        assert!(system.is_finger_active_in_mode(FingerType::Middle));
        assert!(!system.is_finger_active_in_mode(FingerType::Thumb));
    }
}