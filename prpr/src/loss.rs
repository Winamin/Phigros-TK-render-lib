use crate::core::note::{Hand, Note, NoteKind};
use crate::core::Chart;
use crate::judge::{Judgement, PlayResult, LIMIT_BAD, LIMIT_GOOD, LIMIT_PERFECT};

#[inline]
pub fn judgement_penalty(j: Judgement) -> f32 {
    match j {
        Judgement::Perfect => 0.0,
        Judgement::Good    => 0.35,
        Judgement::Bad     => 1.0,
        Judgement::Miss    => 2.0,
    }
}

#[inline]
pub fn note_loss(judgement: Judgement, dt: f32, hand_feasible: bool) -> f32 {
    let penalty   = judgement_penalty(judgement);
    let timing    = 0.5 * dt * dt;
    let phys      = if hand_feasible { 0.0 } else { 5.0 };
    penalty + timing + phys
}

pub fn chart_loss(result: &PlayResult, feasible_flags: &[bool]) -> f32 {
    let n = result.num_of_notes.max(1) as f32;
    let accuracy_loss = (1.0 - result.accuracy as f32).max(0.0);
    let miss_rate = result.counts[Judgement::Miss as usize] as f32 / n;

    let early = result.early as f32;
    let late  = result.late  as f32;
    let timing_variance = if result.num_of_notes > 0 {
        let imbalance = ((early - late) / n).powi(2);
        imbalance
    } else { 0.0 };

    let combo_loss = if result.num_of_notes > 0 {
        1.0 - (result.max_combo as f32 / n).min(1.0)
    } else { 0.0 };

    let infeasible_rate = if feasible_flags.is_empty() {
        0.0
    } else {
        let bad = feasible_flags.iter().filter(|&&ok| !ok).count() as f32;
        bad / feasible_flags.len() as f32
    };

    const KAPPA: f32 = 1.5;
    const TAU:   f32 = 0.3;
    const GAMMA: f32 = 0.4;
    const ETA:   f32 = 2.0;

    accuracy_loss
        + KAPPA * miss_rate
        + TAU   * timing_variance
        + GAMMA * combo_loss
        + ETA   * infeasible_rate
}

pub fn chart_loss_from_counts(counts: [u32; 4], max_combo: u32, total: u32) -> f32 {
    if total == 0 { return 0.0; }
    let n = total as f32;
    let perfect = counts[Judgement::Perfect as usize] as f32;
    let good    = counts[Judgement::Good    as usize] as f32;
    let bad     = counts[Judgement::Bad     as usize] as f32;
    let miss    = counts[Judgement::Miss    as usize] as f32;

    let accuracy = (perfect + 0.65 * good) / n;
    let miss_rate = miss / n;
    let combo_loss = 1.0 - (max_combo as f32 / n).min(1.0);
    let bad_rate = bad / n;

    (1.0 - accuracy) + 1.5 * miss_rate + 0.4 * combo_loss + 0.8 * bad_rate
}

pub fn simulate_note_outcome(
    note: &Note,
    hand: Hand,
    hand_system: &crate::hand_model::ErgonomicHandSystem,
    note_world_x: f32,
    current_time: f32,
) -> (Judgement, f32, bool) {
    let pos = crate::hand_model::Vector2::new(note_world_x, 0.0);
    let (success, pos_err, time_err, _conf) =
        hand_system.evaluate_note_success(hand, &pos, note.time, current_time, &note.kind);

    let dt = time_err; // 秒

    let judgement = if !success {
        // 手模型认为根本打不到 → Miss
        Judgement::Miss
    } else if dt <= LIMIT_PERFECT && pos_err <= 0.1 {
        Judgement::Perfect
    } else if dt <= LIMIT_GOOD {
        Judgement::Good
    } else if dt <= LIMIT_BAD {
        Judgement::Bad
    } else {
        Judgement::Miss
    };

    let judgement = match note.kind {
        NoteKind::Flick | NoteKind::Drag if matches!(judgement, Judgement::Bad) => {
            Judgement::Good
        }
        _ => judgement,
    };

    let feasible = success || dt <= LIMIT_BAD;
    (judgement, dt, feasible)
}

pub fn evaluate_assignments(
    chart: &Chart,
    assignments: &[Hand],
    hand_system: &crate::hand_model::ErgonomicHandSystem,
) -> (Vec<f32>, f32) {
    let mut per_note = Vec::with_capacity(assignments.len());
    let mut counts = [0u32; 4];
    let mut max_combo = 0u32;
    let mut combo = 0u32;
    let mut total = 0u32;
    let mut feasible_flags = Vec::with_capacity(assignments.len());

    let mut idx = 0;
    for line in &chart.lines {
        for note in &line.notes {
            if note.fake { continue; }
            if idx >= assignments.len() { break; }

            let hand = assignments[idx];
            // 用 note 的 translation.x 作为世界坐标近似
            let world_x = note.object.translation.0.now_opt().unwrap_or(0.0);
            let (j, dt, feasible) =
                simulate_note_outcome(note, hand, hand_system, world_x, note.time);

            per_note.push(note_loss(j, dt, feasible));
            feasible_flags.push(feasible);
            counts[j as usize] += 1;
            total += 1;
            match j {
                Judgement::Perfect | Judgement::Good => {
                    combo += 1;
                    if combo > max_combo { max_combo = combo; }
                }
                _ => combo = 0,
            }
            idx += 1;
        }
    }

    let aggregated = chart_loss_from_counts(counts, max_combo, total)
        + if feasible_flags.is_empty() { 0.0 }
          else {
              let bad = feasible_flags.iter().filter(|&&ok| !ok).count() as f32;
              2.0 * bad / feasible_flags.len() as f32
          };

    (per_note, aggregated)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_is_zero_loss() {
        assert_eq!(note_loss(Judgement::Perfect, 0.0, true), 0.0);
    }

    #[test]
    fn miss_is_heaviest() {
        let p = note_loss(Judgement::Perfect, 0.0, true);
        let g = note_loss(Judgement::Good,    0.0, true);
        let b = note_loss(Judgement::Bad,     0.0, true);
        let m = note_loss(Judgement::Miss,    0.0, true);
        assert!(p < g && g < b && b < m);
    }

    #[test]
    fn infeasible_hand_heavily_penalized() {
        let ok  = note_loss(Judgement::Perfect, 0.0, true);
        let bad = note_loss(Judgement::Perfect, 0.0, false);
        assert!(bad - ok >= 4.9);
    }

    #[test]
    fn chart_loss_full_perfect_is_zero() {
        let r = PlayResult {
            score: 1_000_000,
            accuracy: 1.0,
            max_combo: 100,
            num_of_notes: 100,
            counts: [100, 0, 0, 0],
            early: 50, late: 50,
            std: 0.0,
        };
        let flags = vec![true; 100];
        let l = chart_loss(&r, &flags);
        assert!(l.abs() < 1e-6, "full perfect should be 0 loss, got {}", l);
    }

    #[test]
    fn chart_loss_counts_api() {
        let l = chart_loss_from_counts([100, 0, 0, 0], 100, 100);
        assert_eq!(l, 0.0);
        let l2 = chart_loss_from_counts([0, 0, 0, 100], 0, 100);
        assert!(l2 > 2.0);
    }
}
