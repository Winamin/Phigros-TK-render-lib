use crate::core::note::Hand;
use crate::core::{Note, NoteKind};
use crate::judge::{Judgement, LIMIT_BAD, LIMIT_GOOD, LIMIT_PERFECT};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}

impl Vector2 {
    pub const ZERO: Vector2 = Vector2 { x: 0.0, y: 0.0 };

    #[inline]
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    #[inline]
    pub fn magnitude(&self) -> f32 {
        self.x.hypot(self.y)
    }

    #[inline]
    pub fn squared_magnitude(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    #[inline]
    pub fn distance_to(&self, other: &Vector2) -> f32 {
        (*self - *other).magnitude()
    }

    #[inline]
    pub fn normalize(&self) -> Vector2 {
        let m = self.magnitude();
        if m > 1e-8 {
            Vector2::new(self.x / m, self.y / m)
        } else {
            Vector2::ZERO
        }
    }

    #[inline]
    pub fn dot(&self, other: &Vector2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    #[inline]
    pub fn cross(&self, other: &Vector2) -> f32 {
        self.x * other.y - self.y * other.x
    }

    #[inline]
    pub fn add(&self, other: &Vector2) -> Vector2 {
        Vector2::new(self.x + other.x, self.y + other.y)
    }

    #[inline]
    pub fn subtract(&self, other: &Vector2) -> Vector2 {
        Vector2::new(self.x - other.x, self.y - other.y)
    }

    #[inline]
    pub fn multiply_scalar(&self, s: f32) -> Vector2 {
        Vector2::new(self.x * s, self.y * s)
    }

    #[inline]
    pub fn rotate(&self, rad: f32) -> Vector2 {
        let (s, c) = rad.sin_cos();
        Vector2::new(self.x * c - self.y * s, self.x * s + self.y * c)
    }

    #[inline]
    pub fn transform_by(&self, origin: &Vector2, rotation_rad: f32) -> Vector2 {
        self.rotate(rotation_rad) + *origin
    }
}

impl std::ops::Add for Vector2 {
    type Output = Vector2;
    fn add(self, r: Vector2) -> Vector2 {
        Vector2::new(self.x + r.x, self.y + r.y)
    }
}
impl std::ops::Sub for Vector2 {
    type Output = Vector2;
    fn sub(self, r: Vector2) -> Vector2 {
        Vector2::new(self.x - r.x, self.y - r.y)
    }
}
impl std::ops::Mul<f32> for Vector2 {
    type Output = Vector2;
    fn mul(self, s: f32) -> Vector2 {
        Vector2::new(self.x * s, self.y * s)
    }
}
impl std::ops::Neg for Vector2 {
    type Output = Vector2;
    fn neg(self) -> Vector2 {
        Vector2::new(-self.x, -self.y)
    }
}

/// Simple AABB / circle for collision.
#[derive(Debug, Clone, Copy)]
pub struct CollisionCapsule {
    pub center: Vector2,
    pub radius: f32,
}

impl CollisionCapsule {
    #[inline]
    pub fn overlaps(&self, other: &CollisionCapsule) -> f32 {
        let d = self.center.distance_to(&other.center);
        let gap = d - (self.radius + other.radius);
        gap
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FingerType {
    Thumb,
    Index,
    Middle,
    Ring,
    Pinky,
}

impl FingerType {
    /// Canonical iteration order (thumb → pinky).
    pub const ALL: [FingerType; 5] = [
        FingerType::Thumb,
        FingerType::Index,
        FingerType::Middle,
        FingerType::Ring,
        FingerType::Pinky,
    ];

    pub fn index(self) -> usize {
        match self {
            FingerType::Thumb => 0,
            FingerType::Index => 1,
            FingerType::Middle => 2,
            FingerType::Ring => 3,
            FingerType::Pinky => 4,
        }
    }

    pub fn from_index(i: usize) -> Option<FingerType> {
        match i {
            0 => Some(FingerType::Thumb),
            1 => Some(FingerType::Index),
            2 => Some(FingerType::Middle),
            3 => Some(FingerType::Ring),
            4 => Some(FingerType::Pinky),
            _ => None,
        }
    }
}

/// One revolute joint with hard range-of-motion limits and current state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Joint {
    /// Current flexion angle (rad). Positive = flexion (bending toward palm).
    pub angle: f32,
    /// Angular velocity (rad/s).
    pub velocity: f32,
    /// Minimum allowed angle (often negative = extension).
    pub min_angle: f32,
    /// Maximum allowed angle (flexion).
    pub max_angle: f32,
    /// Peak voluntary torque at this joint (N·m). Used for effort.
    pub max_torque: f32,
    /// 0 = fresh, 1 = fully fatigued (reduces max_torque).
    pub fatigue: f32,
}

impl Joint {
    pub fn new(min_angle: f32, max_angle: f32, max_torque: f32) -> Self {
        Self {
            angle: 0.0,
            velocity: 0.0,
            min_angle,
            max_angle,
            max_torque,
            fatigue: 0.0,
        }
    }

    /// Clamp angle into anatomical limits. Returns the post-clamp angle.
    #[inline]
    pub fn clamp_angle(&mut self) -> f32 {
        self.angle = self.angle.clamp(self.min_angle, self.max_angle);
        self.angle
    }

    /// Effective torque capacity after fatigue (linear degradation).
    #[inline]
    pub fn effective_torque(&self) -> f32 {
        self.max_torque * (1.0 - 0.7 * self.fatigue).max(0.0)
    }
}

/// One bone segment in the kinematic chain.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BoneSegment {
    pub length: f32,       // cm
    pub thickness: f32,    // cm (for collision capsules)
    pub mass_g: f32,       // grams
}

impl BoneSegment {
    pub fn new(length: f32, thickness: f32, mass_g: f32) -> Self {
        Self { length, thickness, mass_g }
    }
}

/// Per-finger skeletal structure with MCP (2-DOF), PIP, DIP joints.
///
/// For the thumb, MCP_abduction models opposition (a much larger range than
/// the other fingers), and DIP is nearly independent of PIP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerSkeleton {
    pub finger_type: FingerType,

    /// Lateral offset from palm center where this finger is mounted (cm).
    /// Positive X = toward the pinky side of the hand.
    pub mount_offset: Vector2,

    /// Static mounting angle of the metacarpal relative to the palm axis.
    pub mount_angle: f32,

    pub metacarpal: BoneSegment,
    pub proximal: BoneSegment,
    pub middle: BoneSegment,
    pub distal: BoneSegment,

    /// MCP abduction/adduction (rad; positive = spreading fingers apart).
    pub mcp_abduction: Joint,
    /// MCP flexion (rad).
    pub mcp_flexion: Joint,
    /// PIP flexion.
    pub pip: Joint,
    /// DIP flexion.
    pub dip: Joint,

    // Derived state (cached from FK).
    #[serde(skip)]
    pub fingertip_in_palm: Vector2,
    #[serde(skip)]
    pub pip_position_in_palm: Vector2,
    #[serde(skip)]
    pub dip_position_in_palm: Vector2,
    #[serde(skip)]
    pub mcp_position_in_palm: Vector2,

    // Biomechanical state.
    pub exertion_integral: f32,   // ∫ effort dt, drives fatigue
    pub tendon_coupling: f32,    // 0..1; 1 = fully coupled FDP pull
}

impl FingerSkeleton {
    /// Build a finger from the canonical Kapandji/Buczek measurements.
    pub fn from_anatomy(finger_type: FingerType) -> Self {
        // Bone lengths (cm) and thicknesses for an average adult hand.
        let (meta_l, prox_l, mid_l, dist_l, thick) = match finger_type {
            FingerType::Thumb  => (4.6, 3.2, 0.0, 2.3, 2.0),
            FingerType::Index  => (6.7, 4.0, 2.3, 1.6, 1.7),
            FingerType::Middle => (6.7, 4.5, 2.8, 1.8, 1.8),
            FingerType::Ring   => (6.2, 4.2, 2.6, 1.7, 1.7),
            FingerType::Pinky  => (5.2, 3.2, 1.9, 1.5, 1.5),
        };

        // Joint range-of-motion (rad). Values from [K] vol 1 + [B].
        // (min, max, peak_torque_Nm)
        let (abd_range, mcp_range, pip_range, dip_range) = match finger_type {
            FingerType::Thumb => (
                (-0.20,  1.20, 0.70),   // opposition / palmar abduction
                (-0.30,  1.00, 1.10),
                (-0.10,  1.60, 1.20),
                (-0.15,  1.80, 0.90),
            ),
            FingerType::Index => (
                (-0.35,  0.35, 0.25),
                (-0.15,  1.60, 1.40),
                ( 0.00,  1.80, 1.70),
                ( 0.00,  1.55, 1.20),
            ),
            FingerType::Middle => (
                (-0.30,  0.30, 0.25),
                (-0.15,  1.60, 1.50),
                ( 0.00,  1.80, 1.80),
                ( 0.00,  1.55, 1.25),
            ),
            FingerType::Ring => (
                (-0.30,  0.30, 0.22),
                (-0.15,  1.60, 1.40),
                ( 0.00,  1.80, 1.60),
                ( 0.00,  1.55, 1.10),
            ),
            FingerType::Pinky => (
                (-0.40,  0.45, 0.18),
                (-0.15,  1.70, 1.20),
                ( 0.00,  1.90, 1.40),
                ( 0.00,  1.65, 0.95),
            ),
        };

        let (abd_min, abd_max, abd_tq) = abd_range;
        let (mcp_min, mcp_max, mcp_tq) = mcp_range;
        let (pip_min, pip_max, pip_tq) = pip_range;
        let (dip_min, dip_max, dip_tq) = dip_range;

        // Mounting offset along the distal edge of the palm.
        // (Palm is ~8 cm wide at the MCP line.)
        let mount_x = match finger_type {
            FingerType::Thumb  => -3.6,
            FingerType::Index  => -2.5,
            FingerType::Middle => -0.8,
            FingerType::Ring   =>  0.9,
            FingerType::Pinky  =>  2.5,
        };
        let mount_y = match finger_type {
            FingerType::Thumb => -3.5,
            _                 =>  4.2,
        };

        let mount_angle = match finger_type {
            FingerType::Thumb  => -1.10,   // thumb points "down-out"
            FingerType::Index  =>  0.10,
            FingerType::Middle =>  0.00,
            FingerType::Ring   => -0.08,
            FingerType::Pinky  => -0.18,
        };

        // Approximate bone masses (g) from Drillis et al. (1964) hand segment data.
        let mm = |l: f32| l * thick * 0.11;
        let metacarpal = BoneSegment::new(meta_l, thick * 1.2, mm(meta_l));
        let proximal   = BoneSegment::new(prox_l, thick, mm(prox_l));
        let mid_effective: f32 = if mid_l < 0.1 { 0.1 } else { mid_l };
        let middle     = BoneSegment::new(mid_effective, thick * 0.9, mm(mid_effective));
        let distal     = BoneSegment::new(dist_l, thick * 0.8, mm(dist_l));

        Self {
            finger_type,
            mount_offset: Vector2::new(mount_x, mount_y),
            mount_angle,
            metacarpal,
            proximal,
            middle,
            distal,
            mcp_abduction: Joint::new(abd_min, abd_max, abd_tq),
            mcp_flexion:   Joint::new(mcp_min, mcp_max, mcp_tq),
            pip:           Joint::new(pip_min, pip_max, pip_tq),
            dip:           Joint::new(dip_min, dip_max, dip_tq),
            fingertip_in_palm: Vector2::ZERO,
            pip_position_in_palm: Vector2::ZERO,
            dip_position_in_palm: Vector2::ZERO,
            mcp_position_in_palm: Vector2::ZERO,
            exertion_integral: 0.0,
            tendon_coupling: 0.0,
        }
    }

    // ─── Kinematics ─────────────────────────────────────────────────────

    /// Forward kinematics. Updates the cached `*_in_palm` positions and
    /// returns the fingertip position in the palm frame.
    pub fn forward_kinematics(&mut self) -> Vector2 {
        // Enforce joint limits before any computation.
        self.mcp_abduction.clamp_angle();
        self.mcp_flexion.clamp_angle();
        self.pip.clamp_angle();
        self.dip.clamp_angle();

        // Thumb has no middle phalanx.
        let mid_l = if self.finger_type == FingerType::Thumb { 0.0 } else { self.middle.length };

        // MCP base position (end of metacarpal along mount_angle).
        let mcp = Vector2::new(self.metacarpal.length, 0.0).rotate(self.mount_angle)
            + self.mount_offset;
        self.mcp_position_in_palm = mcp;

        // Proximal phalanx: rotated by mcp_abduction + mcp_flexion (2-D we fold
        // abduction into the rotation; a full 3-D model would keep them separate).
        let prox_dir = self.mount_angle + self.mcp_abduction.angle * 0.4
            + self.mcp_flexion.angle;
        let pip = mcp + Vector2::new(self.proximal.length, 0.0).rotate(prox_dir);
        self.pip_position_in_palm = pip;

        // Middle phalanx: adds PIP flexion.
        let mid_dir = prox_dir + self.pip.angle;
        let dip = pip + Vector2::new(mid_l, 0.0).rotate(mid_dir);
        self.dip_position_in_palm = dip;

        // Distal phalanx: adds DIP flexion.
        let dist_dir = mid_dir + self.dip.angle;
        let tip = dip + Vector2::new(self.distal.length, 0.0).rotate(dist_dir);
        self.fingertip_in_palm = tip;
        tip
    }

    /// Reach envelope: max distance from MCP base to fingertip when fully
    /// extended, useful as a cheap pre-filter.
    pub fn max_reach_from_mcp(&self) -> f32 {
        self.proximal.length
            + self.middle.length
            + self.distal.length
    }

    /// World-space fingertip position given a palm origin and rotation.
    pub fn fingertip_world(&self, palm_origin: Vector2, palm_rot: f32) -> Vector2 {
        self.fingertip_in_palm.transform_by(&palm_origin, palm_rot)
    }

    /// Analytical 2-D IK for the (proximal + middle + distal) subchain,
    /// treating the DIP as coupled to the PIP (coupling factor ≈ 0.7 from FDP
    /// tendon sharing; see [Chalfoun et al. 2006]).
    ///
    /// `target` is the desired fingertip position *in the MCP frame* (i.e.
    /// with MCP at the origin and the proximal bone along +X when angle = 0).
    ///
    /// Returns `(mcp_flexion, pip_flexion, dip_flexion, reached)` where
    /// `reached` is false if the target is outside the reach envelope.
    pub fn solve_ik(&self, target: Vector2) -> (f32, f32, f32, bool) {
        const COUPLING: f32 = 0.70; // DIP ≈ 0.7·PIP

        // Thumb: only two bones (no middle phalanx).
        let mid_l = if self.finger_type == FingerType::Thumb { 0.0 } else { self.middle.length };

        // Effective 2-link arm: L1 = proximal, L2 = middle + distal (because
        // DIP is slaved to PIP, the distal + middle act as one rigid-ish link).
        let l1 = self.proximal.length;
        let l2 = mid_l + self.distal.length;

        let d2 = target.squared_magnitude();
        let d = d2.sqrt();
        let max_reach = l1 + l2;
        let min_reach = (l1 - l2).abs();

        if d > max_reach * 1.001 || d < min_reach * 0.999 {
            // Out of envelope. Return best-effort angles pointing at target.
            let dir = if d > 1e-6 { target.y.atan2(target.x) } else { 0.0 };
            return (dir, self.pip.max_angle, self.dip.max_angle, false);
        }

        // Standard 2-link planar IK (elbow-up solution).
        let cos_pip = ((d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
        let pip_angle = (cos_pip).acos();  // angle between L1 and L2
        let k1 = l1 + l2 * cos_pip;
        let k2 = l2 * (1.0 - cos_pip * cos_pip).sqrt();
        let mcp_angle = target.y.atan2(target.x) - k2.atan2(k1);

        // PIP flexion is how much we bend past straight: π - acos(cos).
        let pip_flexion = PI - pip_angle;
        let dip_flexion = pip_flexion * COUPLING;

        (mcp_angle, pip_flexion, dip_flexion, true)
    }

    /// Apply the IK solution, clamping to joint limits.
    pub fn apply_ik(&mut self, mcp: f32, pip: f32, dip: f32) {
        self.mcp_flexion.angle = mcp;
        self.pip.angle = pip;
        self.dip.angle = dip;
        self.mcp_flexion.clamp_angle();
        self.pip.clamp_angle();
        self.dip.clamp_angle();
    }

    // ─── Biomechanics ───────────────────────────────────────────────────

    /// Total muscle effort (0..1) for the current joint state.
    /// Computed as sum of |angle|/range weighted by the torque demand.
    pub fn instantaneous_effort(&self) -> f32 {
        let norm = |j: &Joint| -> f32 {
            let span = (j.max_angle - j.min_angle).max(1e-3);
            let rel = (j.angle - j.min_angle) / span; // 0..1
            // U-shaped effort: both extremes are costly.
            let u = (rel - 0.5).abs() * 2.0;
            u * u
        };
        let abd_effort = norm(&self.mcp_abduction) * 0.25;
        let mcp_effort = norm(&self.mcp_flexion) * 0.30;
        let pip_effort = norm(&self.pip) * 0.25;
        let dip_effort = norm(&self.dip) * 0.20;
        (abd_effort + mcp_effort + pip_effort + dip_effort).min(1.0)
    }

    /// Step the fatigue state forward by `dt` seconds.
    ///
    /// Uses a simple first-order model:
    ///
    /// ```text
    /// exertion_integral += effort · dt
    /// fatigue           = 1 - exp(-exertion_integral / τ_rise)
    /// fatigue          += recovery_rate · dt   (when effort ≈ 0)
    /// ```
    pub fn step_fatigue(&mut self, dt: f32) {
        const TAU_RISE: f32 = 40.0;     // seconds to saturate
        const RECOVERY_RATE: f32 = 0.015; // per second at rest

        let effort = self.instantaneous_effort();
        self.exertion_integral += effort * dt;

        let target_fatigue = 1.0 - (-self.exertion_integral / TAU_RISE).exp();
        // Low-pass filter toward the target.
        let alpha = (dt / (dt + 2.0)).clamp(0.0, 1.0);
        let new_fatigue = (1.0 - alpha) * self.fatigue_avg() + alpha * target_fatigue;

        // Recovery at rest.
        let recovery = if effort < 0.1 { RECOVERY_RATE * dt } else { 0.0 };
        let final_fatigue = (new_fatigue - recovery).clamp(0.0, 1.0);

        // Write back into each joint (same value – we model a shared muscle
        // pool per finger).
        for j in [
            &mut self.mcp_abduction,
            &mut self.mcp_flexion,
            &mut self.pip,
            &mut self.dip,
        ] {
            j.fatigue = final_fatigue;
        }
    }

    fn fatigue_avg(&self) -> f32 {
        (self.mcp_abduction.fatigue
            + self.mcp_flexion.fatigue
            + self.pip.fatigue
            + self.dip.fatigue)
            * 0.25
    }

    /// Collision capsule for the fingertip (used for inter-finger checks).
    pub fn fingertip_capsule(&self) -> CollisionCapsule {
        CollisionCapsule {
            center: self.fingertip_in_palm,
            radius: self.distal.thickness * 0.5 + 0.2, // small safety margin
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §3  Skeletal hand + forearm
// ─────────────────────────────────────────────────────────────────────────────

/// One skeletal hand (palm + 5 fingers).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletalHand {
    pub hand_type: Hand,

    /// Palm origin in world space (cm; origin = player midline).
    pub palm_position: Vector2,
    /// Palm rotation (rad; 0 = fingers pointing +Y, palm plane parallel to screen).
    pub palm_rotation: f32,
    /// Palm velocity (cm/s) – used by Fitts' Law to estimate next-MT.
    pub palm_velocity: Vector2,

    pub fingers: [FingerSkeleton; 5],

    // Aggregate state (derived; updated by `step`).
    pub grip_force_capacity: f32,   // N, degrades with fatigue + velocity
    pub hand_fatigue: f32,          // mean over fingers
    pub dexterity: f32,             // 0..1 composite score
}

impl SkeletalHand {
    pub fn new(hand_type: Hand) -> Self {
        let mut fingers: [FingerSkeleton; 5] = [
            FingerSkeleton::from_anatomy(FingerType::Thumb),
            FingerSkeleton::from_anatomy(FingerType::Index),
            FingerSkeleton::from_anatomy(FingerType::Middle),
            FingerSkeleton::from_anatomy(FingerType::Ring),
            FingerSkeleton::from_anatomy(FingerType::Pinky),
        ];

        // Mirror the X axis for the right hand (the whole skeleton is defined
        // in a left-hand frame).
        if matches!(hand_type, Hand::Right) {
            for f in &mut fingers {
                f.mount_offset.x = -f.mount_offset.x;
                f.mount_angle = -f.mount_angle;
                f.mcp_abduction.angle = -f.mcp_abduction.angle;
            }
        }

        Self {
            hand_type,
            palm_position: Vector2::new(
                if matches!(hand_type, Hand::Left) { -15.0 } else { 15.0 },
                0.0,
            ),
            palm_rotation: 0.0,
            palm_velocity: Vector2::ZERO,
            fingers,
            grip_force_capacity: 300.0, // ~30 kg, average adult
            hand_fatigue: 0.0,
            dexterity: 1.0,
        }
    }

    /// Run forward kinematics for every finger and return their world-space
    /// fingertip positions.
    pub fn update_kinematics(&mut self) -> [Vector2; 5] {
        let mut tips = [Vector2::ZERO; 5];
        for (i, f) in self.fingers.iter_mut().enumerate() {
            f.forward_kinematics();
            tips[i] = f.fingertip_world(self.palm_position, self.palm_rotation);
        }
        self.hand_fatigue = self.fingers.iter().map(|f| f.fatigue_avg()).sum::<f32>() / 5.0;
        // Force-velocity: faster movement → lower peak grip force.
        //   F(v) = F0 · (1 - |v| / v_max)  (linearized Hill)
        let v = self.palm_velocity.magnitude();
        const V_MAX: f32 = 150.0; // cm/s, peak hand-transport speed
        let velocity_factor = (1.0 - v / V_MAX).max(0.2);
        self.grip_force_capacity = 300.0 * velocity_factor * (1.0 - 0.5 * self.hand_fatigue);
        // Dexterity: 1 - fatigue, weighted toward the active fingers.
        self.dexterity = (1.0 - self.hand_fatigue).clamp(0.0, 1.0);
        tips
    }

    /// Move the palm toward a target over `dt` seconds using a bell-shaped
    /// velocity profile (minimum-jerk trajectory). Returns whether the target
    /// was reached within this step.
    pub fn step_palm_toward(&mut self, target: Vector2, dt: f32, movement_time: f32) -> bool {
        if movement_time < 1e-3 {
            self.palm_position = target;
            self.palm_velocity = Vector2::ZERO;
            return true;
        }
        let elapsed = dt.min(movement_time);
        let t_norm = elapsed / movement_time; // 0..1
        // Minimum-jerk position: s(t) = 10t³ - 15t⁴ + 6t⁵
        let s = 10.0 * t_norm.powi(3) - 15.0 * t_norm.powi(4) + 6.0 * t_norm.powi(5);
        let new_pos = self.palm_position + (target - self.palm_position) * s;
        self.palm_velocity = (new_pos - self.palm_position) * (1.0 / dt.max(1e-3));
        self.palm_position = new_pos;
        t_norm >= 1.0
    }

    /// World-space fingertip position for one finger.
    pub fn fingertip_world(&self, finger: FingerType) -> Vector2 {
        self.fingers[finger.index()].fingertip_world(self.palm_position, self.palm_rotation)
    }

    /// Fitts'-Law movement time (s) to move this hand's *index finger tip*
    /// from its current world position to `target` (also world-space, cm).
    ///
    /// `effective_target_width` is the "W" in Fitts' Law – for a Phigros note
    /// it is approximately the judgement-line width in cm.
    pub fn fitts_movement_time(&self, target: Vector2, effective_target_width: f32) -> f32 {
        let current_tip = self.fingers[FingerType::Index.index()]
            .fingertip_world(self.palm_position, self.palm_rotation);
        let d = current_tip.distance_to(&target).max(0.1);
        let w = effective_target_width.max(0.5);
        fitts_movement_time(d, w)
    }

    /// Solve IK for one finger so its fingertip lands at `world_target`.
    /// Returns `true` if the target is inside the reachable envelope.
    pub fn aim_finger_at(
        &mut self,
        finger: FingerType,
        world_target: Vector2,
    ) -> bool {
        // Transform world target into the MCP frame of the finger.
        let f = &self.fingers[finger.index()];
        let mcp_world = f.mcp_position_in_palm.transform_by(&self.palm_position, self.palm_rotation);
        let local = (world_target - mcp_world).rotate(-self.palm_rotation - f.mount_angle);

        let (mcp, pip, dip, reached) = self.fingers[finger.index()].solve_ik(local);
        self.fingers[finger.index()].apply_ik(mcp, pip, dip);
        self.fingers[finger.index()].forward_kinematics();
        reached
    }

    /// Advance the whole hand by `dt` seconds (fatigue + kinematics).
    pub fn step(&mut self, dt: f32) {
        for f in &mut self.fingers {
            f.step_fatigue(dt);
            f.forward_kinematics();
        }
        self.update_kinematics();
    }

    /// Check inter-finger collisions within this hand.
    pub fn detect_internal_collisions(&self) -> Vec<(FingerType, FingerType, f32)> {
        let mut pairs = Vec::new();
        for i in 0..5 {
            for j in (i + 1)..5 {
                let a = self.fingers[i].fingertip_capsule();
                let b = self.fingers[j].fingertip_capsule();
                let gap = a.overlaps(&b);
                if gap < 0.0 {
                    if let (Some(fi), Some(fj)) =
                        (FingerType::from_index(i), FingerType::from_index(j))
                    {
                        pairs.push((fi, fj, -gap));
                    }
                }
            }
        }
        pairs
    }
}

/// Forearm: shoulder → elbow → wrist (the wrist is the palm origin).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletalArm {
    pub side: Hand,

    pub shoulder: Vector2,     // cm; world-space
    pub elbow: Vector2,
    pub wrist: Vector2,        // = SkeletalHand::palm_position

    pub upper_arm_length: f32, // cm (acromion → lateral epicondyle)
    pub forearm_length: f32,   // cm (lateral epicondyle → radial styloid)

    pub shoulder_angle: f32,   // rad; 0 = arm hanging down
    pub elbow_angle: f32,      // rad; 0 = straight, positive = flexion

    pub shoulder_fatigue: f32,
    pub elbow_fatigue: f32,
}

impl SkeletalArm {
    pub fn new(side: Hand) -> Self {
        let x = if matches!(side, Hand::Left) { -22.0 } else { 22.0 };
        Self {
            side,
            shoulder: Vector2::new(x, 45.0), // shoulder sits above the origin
            elbow:    Vector2::new(x, 20.0),
            wrist:    Vector2::new(x, 0.0),
            upper_arm_length: 30.0,
            forearm_length: 26.0,
            shoulder_angle: 0.0,
            elbow_angle: 0.0,
            shoulder_fatigue: 0.0,
            elbow_fatigue: 0.0,
        }
    }

    /// 2-link IK so the wrist lands at `target`. Returns `(shoulder, elbow)`
    /// angles in radians, and whether the target was reachable.
    pub fn solve_wrist_ik(&self, target: Vector2) -> (f32, f32, bool) {
        let d_vec = target - self.shoulder;
        let d2 = d_vec.squared_magnitude();
        let d = d2.sqrt();
        let l1 = self.upper_arm_length;
        let l2 = self.forearm_length;
        let max_reach = l1 + l2;
        if d > max_reach * 1.001 || d < (l1 - l2).abs() * 0.999 {
            // Unreachable: aim straight at the target.
            let dir = d_vec.y.atan2(d_vec.x);
            return (dir, 0.0, false);
        }
        let cos_e = ((d2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)).clamp(-1.0, 1.0);
        let elbow = (cos_e).acos();
        let k1 = l1 + l2 * cos_e;
        let k2 = l2 * (1.0 - cos_e * cos_e).sqrt();
        let shoulder = d_vec.y.atan2(d_vec.x) - k2.atan2(k1);
        (shoulder, PI - elbow, true)
    }

    /// Apply the IK solution and recompute elbow/wrist positions.
    pub fn apply_ik(&mut self, shoulder: f32, elbow_flexion: f32) {
        self.shoulder_angle = shoulder;
        self.elbow_angle = elbow_flexion;
        let elbow = self.shoulder
            + Vector2::new(self.upper_arm_length, 0.0).rotate(shoulder);
        let wrist = elbow
            + Vector2::new(self.forearm_length, 0.0).rotate(shoulder + elbow_flexion);
        self.elbow = elbow;
        self.wrist = wrist;
    }

    /// Advance fatigue.
    pub fn step(&mut self, dt: f32, effort: f32) {
        // Simple first-order model, same idea as the finger version.
        let rise = (effort * dt * 0.02).min(0.05);
        let recovery = 0.01 * dt;
        self.shoulder_fatigue = (self.shoulder_fatigue + rise - recovery).clamp(0.0, 1.0);
        self.elbow_fatigue    = (self.elbow_fatigue    + rise - recovery).clamp(0.0, 1.0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// §4  Motion dynamics
// ─────────────────────────────────────────────────────────────────────────────

/// Fitts' Law: movement time to acquire a target of width `W` at distance `D`.
///
/// `MT = a + b · log₂(D/W + 1)`.
///
/// Constants from [Soukoreff & MacKenzie 2004] meta-analysis for rapid aimed
/// hand movements: a ≈ 0.0 s, b ≈ 0.10 s/bit (well-practiced users).
/// We add a floor so MT never drops below a physiologically plausible value.
#[inline]
pub fn fitts_movement_time(distance: f32, target_width: f32) -> f32 {
    const A: f32 = 0.040;   // intercept (s) – reaction + trigger
    const B: f32 = 0.095;   // slope (s/bit)
    const MIN_MT: f32 = 0.050; // 50 ms floor
    let id = ((distance / target_width.max(1e-3)) + 1.0).log2();
    (A + B * id).max(MIN_MT)
}

/// Predict the timing error (in seconds) of a movement of duration `mt` that
/// was initiated `initiation_delay` seconds after the ideal time. Returns the
/// absolute error vs. `desired_arrival_time`.
#[inline]
pub fn predicted_timing_error(
    mt: f32,
    initiation_delay: f32,
    desired_arrival_time_after_now: f32,
) -> f32 {
    let arrival = mt + initiation_delay;
    (arrival - desired_arrival_time_after_now).abs()
}

#[derive(Debug, Clone, Copy)]
pub struct NotePrediction {
    pub judgement: Judgement,
    /// Predicted timing error (s, non-negative).
    pub dt: f32,
    /// World-space fingertip-vs-note distance (cm).
    pub position_error: f32,
    /// True if the hand can physically reach the note in time.
    pub feasible: bool,
    /// 0..1 composite score (legacy; prefer `loss` directly).
    pub confidence: f32,
    /// Scalar loss from `phigros_loss::note_loss`.
    pub loss: f32,
}

impl SkeletalHand {
    pub fn predict_action_outcome(
        &self,
        world_target: Vector2,
        note_kind: &NoteKind,
        note_time: f32,
        current_time: f32,
    ) -> NotePrediction {
        let idx = &self.fingers[FingerType::Index.index()];
        let mcp_world = idx.mcp_position_in_palm
            .transform_by(&self.palm_position, self.palm_rotation);
        let reach = mcp_world.distance_to(&world_target);
        let max_reach = idx.max_reach_from_mcp();
        let reachable = reach <= max_reach * 1.02;

        let target_width = match note_kind {
            NoteKind::Click => 3.0,
            NoteKind::Hold { .. } => 4.0,
            NoteKind::Drag => 5.0,
            NoteKind::Flick => 4.0,
        };
        let mt = self.fitts_movement_time(world_target, target_width);

        let dt = predicted_timing_error(mt, 0.0, note_time - current_time);

        let mut judgement = if !reachable {
            Judgement::Miss
        } else if dt <= LIMIT_PERFECT {
            Judgement::Perfect
        } else if dt <= LIMIT_GOOD {
            Judgement::Good
        } else if dt <= LIMIT_BAD {
            Judgement::Bad
        } else {
            Judgement::Miss
        };

        if matches!(note_kind, NoteKind::Flick | NoteKind::Drag)
            && matches!(judgement, Judgement::Bad)
        {
            judgement = Judgement::Good;
        }
        // Hold: as long as we reach it, we at least get Good.
        if matches!(note_kind, NoteKind::Hold { .. }) && reachable
            && matches!(judgement, Judgement::Bad | Judgement::Miss)
        {
            judgement = Judgement::Good;
        }

        let feasible = reachable && dt <= LIMIT_BAD;
        let loss = crate::loss::note_loss(judgement, dt, feasible);
        let confidence = (1.0 - loss * 0.5).clamp(0.0, 1.0);

        NotePrediction {
            judgement,
            dt,
            position_error: reach,
            feasible,
            confidence,
            loss,
        }
    }

    pub fn choose_best_hand(
        left: &SkeletalHand,
        right: &SkeletalHand,
        note: &Note,
    ) -> (Hand, NotePrediction) {
        let world = Vector2::new(note.object.translation.0.now(), 0.0);
        let lp = left.predict_action_outcome(world, &note.kind, note.time, note.time);
        let rp = right.predict_action_outcome(world, &note.kind, note.time, note.time);
        if lp.loss <= rp.loss {
            (Hand::Left, lp)
        } else {
            (Hand::Right, rp)
        }
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GameMode {
    TwoFinger,
    FourFinger,
}

/// Legacy finger-state record. Fields are now *derived* from
/// `FingerSkeleton` state inside `ErgonomicHandSystem::sync_legacy_views`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerModel {
    pub position: Vector2,
    pub bend_angle: f32,
    pub length: f32,
    pub thickness: f32,
    pub fatigue: f32,
    pub dexterity: f32,
    pub is_pressed: bool,
    pub press_time: f32,
    pub finger_type: FingerType,
    #[serde(default)] pub last_time: f32,
    #[serde(default)] pub confidence: f32,
    #[serde(default)] pub success_streak: u32,
    #[serde(default)] pub total_actions: u32,
    #[serde(default)] pub performance_score: f32,
    #[serde(default)] pub is_busy: bool,
    #[serde(default)] pub busy_until: f32,
}

impl FingerModel {
    pub fn from_skeleton(fs: &FingerSkeleton) -> Self {
        Self {
            position: fs.fingertip_in_palm,
            bend_angle: fs.mcp_flexion.angle + fs.pip.angle + fs.dip.angle,
            length: fs.proximal.length + fs.middle.length + fs.distal.length,
            thickness: fs.proximal.thickness,
            fatigue: fs.fatigue_avg(),
            dexterity: (1.0 - fs.fatigue_avg()).clamp(0.0, 1.0),
            is_pressed: false,
            press_time: 0.0,
            finger_type: fs.finger_type,
            last_time: -1.0,
            confidence: 1.0,
            success_streak: 0,
            total_actions: 0,
            performance_score: 1.0,
            is_busy: false,
            busy_until: -1.0,
        }
    }

    /// Backward-compat: scalar "suitability" (higher is better) for pressing
    /// `target` from this finger's current pose.
    pub fn calculate_suitability(&self, target: &Vector2) -> f32 {
        let d = self.position.distance_to(target);
        // Reach cost: 1 at origin, 0 at length+margin, negative beyond.
        let reach = (1.0 - (d / (self.length + 0.5)).min(1.0)).clamp(0.0, 1.0);
        let fatigue_factor = 1.0 - self.fatigue;
        let dexterity_factor = self.dexterity;
        ((reach + fatigue_factor + dexterity_factor) / 3.0).clamp(0.0, 1.0)
    }
}

/// Legacy palm descriptor. `position` stays public because `hand.rs` reads
/// and writes it directly; the real kinematics are in `SkeletalHand`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandModel {
    pub position: Vector2,
    pub velocity: Vector2,
    pub acceleration: Vector2,
    pub rotation: f32,
    pub openness: f32,
    pub fatigue: f32,
    pub dexterity: f32,
    pub last_update_time: f32,
    pub hand_type: Hand,
}

impl HandModel {
    pub fn from_skeletal(sh: &SkeletalHand) -> Self {
        Self {
            position: sh.palm_position,
            velocity: sh.palm_velocity,
            acceleration: Vector2::ZERO,
            rotation: sh.palm_rotation,
            openness: 1.0,
            fatigue: sh.hand_fatigue,
            dexterity: sh.dexterity,
            last_update_time: 0.0,
            hand_type: sh.hand_type,
        }
    }

    /// Backward-compat: a coarse "movement difficulty" scalar used by `hand.rs`
    /// as one input to the neural-network feature vector. The real kinematics
    /// live inside `SkeletalHand`; this is a cheap stand-in.
    pub fn calculate_movement_difficulty(&self, target: &Vector2) -> f32 {
        let d = self.position.distance_to(target);
        // Map distance into [0, 1] with saturation at ~15 cm (≈ hand span).
        let d_norm = (d / 15.0).min(1.0);
        // Penalise if we are already moving fast (momentum cost).
        let v_norm = (self.velocity.magnitude() / 150.0).min(1.0);
        // Penalise fatigue.
        ((d_norm + v_norm + self.fatigue) / 3.0).clamp(0.0, 1.0)
    }
}

/// Legacy arm descriptor; now sourced from `SkeletalArm`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmModel {
    pub shoulder_position: Vector2,
    pub elbow_position: Vector2,
    pub wrist_position: Vector2,
    pub angle: f32,
    pub length: f32,
    pub thickness: f32,
    pub fatigue: f32,
    pub strength: f32,
    pub flexibility: f32,
}

impl ArmModel {
    pub fn from_skeletal(sa: &SkeletalArm) -> Self {
        Self {
            shoulder_position: sa.shoulder,
            elbow_position: sa.elbow,
            wrist_position: sa.wrist,
            angle: sa.shoulder_angle,
            length: sa.upper_arm_length + sa.forearm_length,
            thickness: 4.0,
            fatigue: 0.5 * (sa.shoulder_fatigue + sa.elbow_fatigue),
            strength: 1.0 - 0.5 * (sa.shoulder_fatigue + sa.elbow_fatigue),
            flexibility: 1.0 - 0.5 * (sa.shoulder_fatigue + sa.elbow_fatigue),
        }
    }

    /// Backward-compat: comfort score in [0, 1] from joint angle + fatigue.
    pub fn calculate_comfort(&self) -> f32 {
        let angle_comfort = (1.0 - (self.angle.abs() / PI).min(1.0)).clamp(0.0, 1.0);
        let fatigue_factor = 1.0 - self.fatigue;
        ((angle_comfort + fatigue_factor + self.strength) / 3.0).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollisionResult {
    pub has_collision: bool,
    pub colliding_pairs: Vec<(usize, usize, f32)>,
    pub min_separation_distance: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartFingerModeSelector {
    pub mode_performance: HashMap<GameMode, ModePerformance>,
    pub two_finger_weight: f32,
    pub four_finger_weight: f32,
    pub timing_analysis_window: f32,
    pub simultaneous_notes_threshold: usize,
    pub time_difference_analysis: f32,
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
        let mut m = HashMap::new();
        m.insert(GameMode::TwoFinger, ModePerformance {
            success_rate: 0.9, average_reward: 0.0, usage_count: 0, last_used: -1.0,
        });
        m.insert(GameMode::FourFinger, ModePerformance {
            success_rate: 0.7, average_reward: 0.0, usage_count: 0, last_used: -1.0,
        });
        Self {
            mode_performance: m,
            two_finger_weight: 1.0,
            four_finger_weight: 0.6,
            timing_analysis_window: 0.4,
            simultaneous_notes_threshold: 2,
            time_difference_analysis: 0.15,
        }
    }
    pub fn reset_performance(&mut self) {
        for p in self.mode_performance.values_mut() {
            p.success_rate = 0.5;
            p.average_reward = 0.0;
            p.usage_count = 0;
        }
    }
}

impl Default for SmartFingerModeSelector {
    fn default() -> Self { Self::new() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerModeInfo {
    pub current_mode: GameMode,
    pub two_finger_performance: ModeStats,
    pub four_finger_performance: ModeStats,
    pub recent_notes_count: usize,
    pub time_since_last_switch: f32,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeStats {
    pub success_rate: f32,
    pub average_reward: f32,
    pub usage_count: u32,
}
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandStatistics {
    pub hand_type: Hand,
    pub position: Vector2,
    pub fatigue: f32,
    pub dexterity: f32,
    pub openness: f32,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErgonomicHandSystem {
    // ── Skeletal (source of truth) ──────────────────────────────────────
    pub left:  SkeletalHand,
    pub right: SkeletalHand,
    pub left_skeleton_arm:  SkeletalArm,
    pub right_skeleton_arm: SkeletalArm,

    // ── Legacy mirrors (kept public for hand.rs) ────────────────────────
    pub left_hand:  HandModel,
    pub right_hand: HandModel,
    pub left_fingers:  Vec<FingerModel>,
    pub right_fingers: Vec<FingerModel>,
    pub left_arm:  ArmModel,
    pub right_arm: ArmModel,
    pub body_center: Vector2,
    pub body_tilt: f32,
    pub difficulty_factor: f32,
    pub game_mode: GameMode,
    pub finger_mode_selector: SmartFingerModeSelector,
    pub current_time: f32,
    #[serde(skip)] pub recent_notes: Vec<Note>,
    pub mode_switch_cooldown: f32,
    pub last_mode_switch_time: f32,
}

impl ErgonomicHandSystem {
    pub fn new() -> Self {
        let left  = SkeletalHand::new(Hand::Left);
        let right = SkeletalHand::new(Hand::Right);
        let left_skeleton_arm  = SkeletalArm::new(Hand::Left);
        let right_skeleton_arm = SkeletalArm::new(Hand::Right);

        let left_fingers: Vec<FingerModel> = left.fingers.iter().map(FingerModel::from_skeleton).collect();
        let right_fingers: Vec<FingerModel> = right.fingers.iter().map(FingerModel::from_skeleton).collect();

        Self {
            left_hand: HandModel::from_skeletal(&left),
            right_hand: HandModel::from_skeletal(&right),
            left_fingers,
            right_fingers,
            left_arm: ArmModel::from_skeletal(&left_skeleton_arm),
            right_arm: ArmModel::from_skeletal(&right_skeleton_arm),
            left, right, left_skeleton_arm, right_skeleton_arm,
            body_center: Vector2::ZERO,
            body_tilt: 0.0,
            difficulty_factor: 1.0,
            game_mode: GameMode::TwoFinger,
            finger_mode_selector: SmartFingerModeSelector::new(),
            current_time: 0.0,
            recent_notes: Vec::new(),
            mode_switch_cooldown: 2.0,
            last_mode_switch_time: -2.0,
        }
    }

    /// Copy skeletal state into the legacy public mirrors. Call this after
    /// any operation that mutates the skeletons.
    fn sync_legacy_views(&mut self) {
        self.left_hand  = HandModel::from_skeletal(&self.left);
        self.right_hand = HandModel::from_skeletal(&self.right);
        self.left_arm  = ArmModel::from_skeletal(&self.left_skeleton_arm);
        self.right_arm = ArmModel::from_skeletal(&self.right_skeleton_arm);
        self.left_fingers  = self.left.fingers.iter().map(FingerModel::from_skeleton).collect();
        self.right_fingers = self.right.fingers.iter().map(FingerModel::from_skeleton).collect();
    }

    /// Ingest any writes that legacy code made to the legacy mirrors (e.g.
    /// `hand.rs` setting `left_hand.position` directly) and push them back
    /// into the skeletal model.
    fn ingest_legacy_writes(&mut self) {
        self.left.palm_position  = self.left_hand.position;
        self.left.palm_rotation  = self.left_hand.rotation;
        self.left.palm_velocity  = self.left_hand.velocity;
        self.right.palm_position = self.right_hand.position;
        self.right.palm_rotation = self.right_hand.rotation;
        self.right.palm_velocity = self.right_hand.velocity;
    }

    /// Advance the full system by `dt` seconds.
    pub fn step(&mut self, dt: f32) {
        self.ingest_legacy_writes();
        self.left.step(dt);
        self.right.step(dt);
        self.left_skeleton_arm.step(dt, self.left.hand_fatigue);
        self.right_skeleton_arm.step(dt, self.right.hand_fatigue);
        self.sync_legacy_views();
        self.current_time += dt;
    }

    /// Keep the legacy `update(time)` entry point used by `hand.rs`.
    pub fn update(&mut self, time: f32) {
        let dt = (time - self.current_time).max(0.0);
        self.step(dt.max(1e-3));
    }

    /// Phigros-note outcome prediction via the new skeletal model.
    ///
    pub fn predict_outcome_for_note(
        &self,
        world_target: Vector2,
        note_kind: &NoteKind,
        note_time: f32,
        current_time: f32,
        hand: Hand,
    ) -> NotePrediction {
        let sh = match hand {
            Hand::Left => &self.left,
            Hand::Right => &self.right,
        };
        sh.predict_action_outcome(world_target, note_kind, note_time, current_time)
    }

    /// Backward-compatible alias used by `hand.rs::calculate_reward`.
    pub fn predict_outcome_from_position(
        &self,
        world_position: Vector2,
        note_time: f32,
        note_kind: &NoteKind,
        hand: Hand,
    ) -> NotePrediction {
        self.predict_outcome_for_note(world_position, note_kind, note_time, note_time, hand)
    }

    /// Backward-compatible wrapper around the skeletal model.
    pub fn predict_note_outcome(&self, note: &Note, hand: Hand) -> NotePrediction {
        let world = Vector2::new(note.object.translation.0.now(), 0.0);
        self.predict_outcome_for_note(world, &note.kind, note.time, note.time, hand)
    }

    pub fn choose_best_hand_for_note(&self, note: &Note) -> (Hand, NotePrediction) {
        SkeletalHand::choose_best_hand(&self.left, &self.right, note)
    }

    /// Backward-compatible note-success predicate. Now implemented in terms
    /// of `predict_outcome_for_note` so all legacy call sites get the new
    /// biomechanical signal for free.
    pub fn evaluate_note_success(
        &self,
        hand: Hand,
        target_position: &Vector2,
        note_time: f32,
        current_time: f32,
        note_kind: &NoteKind,
    ) -> (bool, f32, f32, f32) {
        let p = self.predict_outcome_for_note(*target_position, note_kind, note_time, current_time, hand);
        let success = !matches!(p.judgement, Judgement::Miss) && p.feasible;
        (success, p.position_error, p.dt, p.confidence)
    }

    /// Legacy API: pick a hand for a note. Now uses loss-driven selection.
    pub fn assign_note_hand(
        &mut self,
        note_position: Vector2,
        note_kind: &NoteKind,
        time: f32,
    ) -> (Hand, usize, f32) {
        let lp = self.left.predict_action_outcome(note_position, note_kind, time, time);
        let rp = self.right.predict_action_outcome(note_position, note_kind, time, time);
        let (hand, pred) = if lp.loss <= rp.loss { (Hand::Left, lp) } else { (Hand::Right, rp) };
        // Pick an active finger on the chosen hand based on game mode.
        let finger_idx = match self.game_mode {
            GameMode::TwoFinger => FingerType::Index.index(),
            GameMode::FourFinger => FingerType::Index.index(),
        };
        (hand, finger_idx, pred.confidence)
    }

    /// Legacy API: per-finger bookkeeping used by `hand.rs`.
    pub fn update_finger_state(
        &mut self,
        hand: Hand,
        finger_type: FingerType,
        new_position: Vector2,
        time: f32,
        success: bool,
        _note_kind: &NoteKind,
    ) {
        let fingers = match hand {
            Hand::Left => &mut self.left_fingers,
            Hand::Right => &mut self.right_fingers,
        };
        if let Some(f) = fingers.iter_mut().find(|f| f.finger_type == finger_type) {
            f.position = new_position;
            f.last_time = time;
            f.total_actions += 1;
            if success { f.success_streak += 1; } else { f.success_streak = 0; }
        }
    }

    pub fn clean_all_finger_states(&mut self) {
        for f in self.left_fingers.iter_mut().chain(self.right_fingers.iter_mut()) {
            if !f.last_time.is_finite() { f.last_time = -1.0; }
            if !f.confidence.is_finite() { f.confidence = 1.0; }
            if !f.performance_score.is_finite() { f.performance_score = 1.0; }
            if !f.busy_until.is_finite() { f.busy_until = -1.0; }
        }
    }

    pub fn set_game_mode(&mut self, mode: GameMode) {
        self.game_mode = mode;
    }

    pub fn is_finger_available(&self, hand: Hand, finger: FingerType, t: f32) -> bool {
        let fingers = match hand {
            Hand::Left => &self.left_fingers,
            Hand::Right => &self.right_fingers,
        };
        let is_active = match self.game_mode {
            GameMode::TwoFinger => matches!(finger, FingerType::Index),
            GameMode::FourFinger => matches!(finger, FingerType::Index | FingerType::Middle),
        };
        fingers
            .iter()
            .find(|f| f.finger_type == finger)
            .map(|f| is_active && !f.is_busy && t >= f.busy_until)
            .unwrap_or(false)
    }

    pub fn get_active_fingers(&self, _hand: Hand) -> Vec<FingerType> {
        match self.game_mode {
            GameMode::TwoFinger => vec![FingerType::Index],
            GameMode::FourFinger => vec![FingerType::Index, FingerType::Middle],
        }
    }

    pub fn detect_finger_collisions(&self, hand: Hand) -> CollisionResult {
        let sh = match hand { Hand::Left => &self.left, Hand::Right => &self.right };
        let pairs = sh.detect_internal_collisions();
        let mut colliding: Vec<(usize, usize, f32)> = Vec::new();
        let mut min_sep = f32::MAX;
        for (a, b, pen) in &pairs {
            colliding.push((a.index(), b.index(), *pen));
        }
        // Inter-hand separation
        if !colliding.is_empty() { min_sep = 0.0; }
        CollisionResult {
            has_collision: !colliding.is_empty(),
            colliding_pairs: colliding,
            min_separation_distance: if min_sep == f32::MAX { 0.0 } else { min_sep },
        }
    }

    pub fn resolve_finger_collisions(&mut self, hand: Hand) {
        // Push each overlapping fingertip away from its neighbour along the
        // line between them. Simple, stable, good enough for 5 fingers.
        let sh = match hand { Hand::Left => &mut self.left, Hand::Right => &mut self.right };
        for _ in 0..3 {
            let pairs = sh.detect_internal_collisions();
            if pairs.is_empty() { break; }
            for (fa, fb, penetration) in pairs {
                let a = sh.fingers[fa.index()].fingertip_in_palm;
                let b = sh.fingers[fb.index()].fingertip_in_palm;
                let dir = (b - a).normalize();
                let push = dir * (penetration * 0.5 + 0.05);
                sh.fingers[fa.index()].fingertip_in_palm = a - push;
                sh.fingers[fb.index()].fingertip_in_palm = b + push;
            }
        }
        self.sync_legacy_views();
    }

    pub fn resolve_all_collisions(&mut self) {
        self.resolve_finger_collisions(Hand::Left);
        self.resolve_finger_collisions(Hand::Right);
    }

    pub fn can_execute_note(&self, hand: Hand, t: f32) -> bool {
        match self.game_mode {
            GameMode::TwoFinger => self.is_finger_available(hand, FingerType::Index, t),
            GameMode::FourFinger => {
                self.is_finger_available(hand, FingerType::Index, t)
                    || self.is_finger_available(hand, FingerType::Middle, t)
            }
        }
    }

    // ── Legacy API stubs used by `hand.rs` ──────────────────────────────

    /// Backward-compat: mark one finger as pressed at time `t`.
    pub fn apply_finger_press(&mut self, hand: Hand, finger_index: usize, t: f32) {
        let fingers = match hand {
            Hand::Left => &mut self.left_fingers,
            Hand::Right => &mut self.right_fingers,
        };
        if let Some(f) = fingers.get_mut(finger_index) {
            f.is_pressed = true;
            f.press_time = t;
            f.is_busy = true;
            // Busy duration based on the active finger type.
            f.busy_until = t + match f.finger_type {
                FingerType::Index => 0.12,
                FingerType::Middle => 0.14,
                _ => 0.18,
            };
        }
    }

    /// Backward-compat: release a previously pressed finger.
    pub fn reset_finger_state(&mut self, hand: Hand, finger_index: usize) {
        let fingers = match hand {
            Hand::Left => &mut self.left_fingers,
            Hand::Right => &mut self.right_fingers,
        };
        if let Some(f) = fingers.get_mut(finger_index) {
            f.is_pressed = false;
        }
    }

    /// Backward-compat: coarse difficulty used by `hand.rs` for feature
    /// construction. The new signal goes through `predict_action_outcome`.
    pub fn calculate_hand_difficulty(
        &self,
        hand_model: &HandModel,
        target: &Vector2,
        note_kind: &NoteKind,
    ) -> f32 {
        let movement = hand_model.calculate_movement_difficulty(target);
        let arm_comfort = match hand_model.hand_type {
            Hand::Left => self.left_arm.calculate_comfort(),
            Hand::Right => self.right_arm.calculate_comfort(),
        };
        let note_factor = match note_kind {
            NoteKind::Click => 1.0,
            NoteKind::Drag => 1.2,
            NoteKind::Flick => 1.3,
            NoteKind::Hold { .. } => 1.5,
        };
        ((movement * 0.6 + (1.0 - arm_comfort) * 0.4) * note_factor * self.difficulty_factor)
            .clamp(0.0, 2.0)
    }

    /// Backward-compat: pick the best finger on `hand` for `target`.
    pub fn select_best_finger(&self, hand: Hand, target: &Vector2) -> (usize, f32) {
        let fingers = match hand {
            Hand::Left => &self.left_fingers,
            Hand::Right => &self.right_fingers,
        };
        let valid_indices: Vec<usize> = match self.game_mode {
            GameMode::TwoFinger => vec![FingerType::Index.index()],
            GameMode::FourFinger => vec![FingerType::Index.index(), FingerType::Middle.index()],
        };
        let mut best_idx = valid_indices[0];
        let mut best_score = -1.0_f32;
        for &i in &valid_indices {
            if let Some(f) = fingers.get(i) {
                let s = f.calculate_suitability(target);
                if s > best_score {
                    best_score = s;
                    best_idx = i;
                }
            }
        }
        (best_idx, best_score.max(0.0))
    }
}

impl Default for ErgonomicHandSystem {
    fn default() -> Self { Self::new() }
}

// ─────────────────────────────────────────────────────────────────────────────
// §8  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_math_basics() {
        let a = Vector2::new(3.0, 4.0);
        assert!((a.magnitude() - 5.0).abs() < 1e-6);
        let b = Vector2::new(1.0, 2.0);
        assert!((a.distance_to(&b) - (8.0_f32).sqrt()).abs() < 1e-5);
        assert!((a.dot(&b) - 11.0).abs() < 1e-6);
        assert!((a.cross(&b) - 2.0).abs() < 1e-6);
        let r = Vector2::new(1.0, 0.0).rotate(PI / 2.0);
        assert!(r.x.abs() < 1e-5 && (r.y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn finger_fk_gives_finite_tip() {
        let mut idx = FingerSkeleton::from_anatomy(FingerType::Index);
        let tip = idx.forward_kinematics();
        assert!(tip.x.is_finite() && tip.y.is_finite());
        // Tip must lie within the full-reach circle.
        let reach = idx.max_reach_from_mcp() + idx.metacarpal.length;
        let d = (tip - idx.mount_offset).magnitude();
        assert!(d <= reach * 1.01);
    }

    #[test]
    fn ik_roundtrip() {
        // The analytical 2-link IK treats the (middle + distal) subchain as a
        // rigid link, so its joint-angle solution won't equal the original
        // posture in general. What must roundtrip is the FINGERTIP POSITION:
        // FK(IK(target)) ≈ target, within ~5 mm (physiological noise floor).
        let mut idx = FingerSkeleton::from_anatomy(FingerType::Middle);

        // Pick a reachable target: FK of a modest posture.
        idx.mcp_flexion.angle = 0.5;
        idx.pip.angle = 0.9;
        idx.dip.angle = 0.6;
        let _ = idx.forward_kinematics();
        let target_in_mcp = (idx.fingertip_in_palm - idx.mcp_position_in_palm)
            .rotate(-idx.mount_angle);

        let (mcp, pip, dip, reached) = idx.solve_ik(target_in_mcp);
        assert!(reached, "target inside envelope must be reachable");

        // Apply the IK solution and re-run FK.
        idx.apply_ik(mcp, pip, dip);
        let reconstructed_local = (idx.fingertip_in_palm - idx.mcp_position_in_palm)
            .rotate(-idx.mount_angle);

        let err = (reconstructed_local - target_in_mcp).magnitude();
        assert!(err < 0.5, "fingertip roundtrip error = {} cm (want < 0.5)", err);
    }

    #[test]
    fn fitts_law_monotone() {
        // Farther targets → longer MT.
        let mt1 = fitts_movement_time(5.0, 2.0);
        let mt2 = fitts_movement_time(50.0, 2.0);
        assert!(mt2 > mt1);
        // Wider targets → shorter MT.
        let mt3 = fitts_movement_time(20.0, 1.0);
        let mt4 = fitts_movement_time(20.0, 5.0);
        assert!(mt3 > mt4);
    }

    #[test]
    fn skeletal_hand_predict_perfect_for_close_target() {
        let hand = SkeletalHand::new(Hand::Left);
        // A target right where the index fingertip already is: should be Perfect.
        let tip = hand.fingers[FingerType::Index.index()]
            .fingertip_world(hand.palm_position, hand.palm_rotation);
        let p = hand.predict_action_outcome(tip, &NoteKind::Click, 0.0, 0.0);
        assert!(matches!(p.judgement, Judgement::Perfect));
        assert!(p.feasible);
    }

    #[test]
    fn skeletal_hand_miss_for_unreachable() {
        let hand = SkeletalHand::new(Hand::Left);
        // A target 2 meters away is well outside human reach.
        let far = Vector2::new(200.0, 0.0);
        let p = hand.predict_action_outcome(far, &NoteKind::Click, 0.0, 0.0);
        assert!(matches!(p.judgement, Judgement::Miss));
        assert!(!p.feasible);
    }

    #[test]
    fn ergonomic_system_api_unchanged() {
        // Spot-check the backward-compatible public surface.
        let mut sys = ErgonomicHandSystem::new();
        let (hand, _finger, _conf) =
            sys.assign_note_hand(Vector2::new(0.0, 0.0), &NoteKind::Click, 0.0);
        assert!(matches!(hand, Hand::Left | Hand::Right));
        sys.set_game_mode(GameMode::FourFinger);
        assert_eq!(sys.game_mode, GameMode::FourFinger);
        let active = sys.get_active_fingers(Hand::Left);
        assert!(active.contains(&FingerType::Index));
        assert!(active.contains(&FingerType::Middle));
        sys.update(0.016);
    }
}
