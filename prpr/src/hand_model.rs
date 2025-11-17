use crate::core::note::Hand;
use crate::core::NoteKind;
use serde::{Deserialize, Serialize};
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
            // 计算速度和加速度
            let new_velocity = new_position.subtract(&self.position).normalize();
            self.acceleration = new_velocity.subtract(&self.velocity).normalize();
            self.velocity = new_velocity;
            self.position = new_position;
            self.last_update_time = time;
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
        let (length, thickness) = match finger_type {
            FingerType::Thumb => (0.6, 0.3),
            FingerType::Index => (1.0, 0.25),
            FingerType::Middle => (1.1, 0.28),
            FingerType::Ring => (0.95, 0.24),
            FingerType::Pinky => (0.8, 0.2),
        };
        
        Self {
            position,
            bend_angle: 0.0,
            length,
            thickness,
            fatigue: 0.0,
            dexterity: 1.0,
            is_pressed: false,
            press_time: 0.0,
            finger_type,
        }
    }

    /// 更新手指状态
    pub fn update(&mut self, target_position: Option<Vector2>, time: f32) {
        if let Some(target) = target_position {
            // 计算手指弯曲角度
            self.bend_angle = target.distance_to(&self.position) / self.length;
            self.bend_angle = self.bend_angle.min(PI / 2.0); // 最大弯曲90度
        }
        
        if self.is_pressed {
            self.press_time += time - self.press_time;
            self.fatigue = (self.fatigue + 0.01 * time).min(1.0);
        } else {
            // 恢复
            self.fatigue = (self.fatigue - 0.005 * time).max(0.0);
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
        // 简化的肘部位置计算
        let shoulder_to_hand = hand_position.subtract(&self.shoulder_position);
        let mid_point = self.shoulder_position.add(&Vector2::new(shoulder_to_hand.x / 2.0, shoulder_to_hand.y / 2.0));
        self.elbow_position = Vector2::new(
            mid_point.x - shoulder_to_hand.y * 0.2,
            mid_point.y + shoulder_to_hand.x * 0.2,
        );
        self.angle = shoulder_to_hand.dot(&Vector2::new(1.0, 0.0)) / shoulder_to_hand.magnitude();
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
        }
    }

    /// 更新整个手部系统
    pub fn update(&mut self, time: f32) {
        // 更新手臂
        self.left_arm.update(&self.left_hand.position);
        self.right_arm.update(&self.right_hand.position);
        
        // 更新手指
        for finger in &mut self.left_fingers {
            // 简化：手指位置跟随手部
            finger.position = self.left_hand.position.add(&Vector2::new(
                (finger.finger_type as i32 - 2) as f32 * 0.1,
                0.1,
            ));
            finger.update(None, time);
        }
        
        for finger in &mut self.right_fingers {
            finger.position = self.right_hand.position.add(&Vector2::new(
                (finger.finger_type as i32 - 2) as f32 * 0.1,
                0.1,
            ));
            finger.update(None, time);
        }
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
        
        for (i, finger) in fingers.iter().enumerate() {
            let suitability = finger.calculate_suitability(target);
            if suitability > best_suitability {
                best_suitability = suitability;
                best_index = i;
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
}