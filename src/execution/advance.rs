use crate::bytes::RuntimeStateByteCount;
use crate::error::RuleAttemptStepError;
use crate::limits::RuleAttemptCount;
use crate::policy::{ExecutionPolicy, RuleAttemptPolicy};
use crate::program::ExecutableProgram;
use crate::runtime::action::{AppliedRule, PreparedRuleStep, prepare_matched_rule};
use crate::runtime::budget::{RuleAttemptBudgetState, RuleAttemptReservation, RuntimeBudgetState};
use crate::runtime::matcher::{EvaluatedRuleMiss, MatchedRuleApplication, RuleAttemptEvaluation};
use crate::runtime::once::{ContinuingRuntimeRulePass, FinalRuntimeRulePass};
use crate::runtime::rewrite::RewriteScratch;
use crate::runtime::state::State;

use super::engine::{
    AttemptRunCoreParts, ContinuingRuleAttemptCore, ContinuingRuleAttemptRun, FinalRuleAttemptCore,
    FinalRuleAttemptRun, TerminalRunCore,
};
use super::session::BorrowedRuleAttemptCursor;
use super::transition::{
    BorrowedAlwaysReturnStateMismatchRuleAttempt, BorrowedAlwaysRewriteStateMismatchRuleAttempt,
    BorrowedContinuingRuleAttemptTransition, BorrowedFinalRuleAttemptTransition,
    BorrowedOnceReturnStateMismatchRuleAttempt, BorrowedOnceRewriteConsumedRuleAttempt,
    BorrowedOnceRewriteStateMismatchRuleAttempt, BorrowedRuleAttemptAlwaysReturnRun,
    BorrowedRuleAttemptAlwaysRewriteStep, BorrowedRuleAttemptFailedRun,
    BorrowedRuleAttemptOnceReturnRun, BorrowedRuleAttemptOnceRewriteStep,
    BorrowedRuleAttemptStableAfterAlwaysReturnStateMismatch,
    BorrowedRuleAttemptStableAfterAlwaysRewriteStateMismatch,
    BorrowedRuleAttemptStableAfterOnceReturnStateMismatch,
    BorrowedRuleAttemptStableAfterOnceRewriteConsumed,
    BorrowedRuleAttemptStableAfterOnceRewriteStateMismatch,
};

/// Advances a borrowed rule-attempt session whose current rule has successors.
pub(super) fn advance_continuing_borrowed_rule_attempt<'program, E, A>(
    session: ContinuingRuleAttemptRun<'program, E, A>,
) -> BorrowedContinuingRuleAttemptTransition<'program, E, A>
where
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
{
    let ContinuingRuleAttemptRun { program, core } = session;
    let ContinuingRuleAttemptCore {
        parts,
        runtime_rules: pass,
    } = core;
    advance_continuing_rule_attempt(program, parts, pass)
}

/// Advances a borrowed rule-attempt session whose current rule exhausts the pass.
pub(super) fn advance_final_borrowed_rule_attempt<'program, E, A>(
    session: FinalRuleAttemptRun<'program, E, A>,
) -> BorrowedFinalRuleAttemptTransition<'program, E, A>
where
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
{
    let FinalRuleAttemptRun { program, core } = session;
    let FinalRuleAttemptCore {
        parts,
        runtime_rules: pass,
    } = core;
    advance_final_rule_attempt(program, parts, pass)
}

/// Rule-attempt miss after the attempt reservation has committed.
struct CommittedRuleAttemptMiss<'program> {
    /// Rule-attempt count committed for this miss.
    attempt: RuleAttemptCount,
    /// Exact non-applying rule result.
    miss: EvaluatedRuleMiss<'program>,
}

/// Prepared rule-attempt application that still owns the attempt reservation.
struct PreparedRuleAttemptApplication<'program, 'once, 'attempt_budget, 'step_budget, E, A>
where
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
{
    /// Rule-attempt reservation that must commit before the prepared rule step.
    attempt: RuleAttemptReservation<'attempt_budget, A>,
    /// Prepared runtime step side effects.
    step: PreparedRuleStep<'program, 'once, 'step_budget, E>,
}

/// Rule-attempt application after both attempt and step side effects commit.
struct CommittedRuleAttemptApplication<'program> {
    /// Rule-attempt count committed for this application.
    attempt: RuleAttemptCount,
    /// Exact committed runtime action.
    applied: AppliedRule<'program>,
}

impl<'program> CommittedRuleAttemptMiss<'program> {
    /// Commits the rule-attempt reservation for one non-applying rule.
    fn commit<A>(
        reservation: RuleAttemptReservation<'_, A>,
        miss: EvaluatedRuleMiss<'program>,
    ) -> Self
    where
        A: RuleAttemptPolicy,
    {
        Self {
            attempt: reservation.commit(),
            miss,
        }
    }

    /// Projects a committed continuing miss into its exact public transition.
    fn into_continuing_transition<E, A>(
        self,
        cursor: BorrowedRuleAttemptCursor<'program, E, A>,
    ) -> BorrowedContinuingRuleAttemptTransition<'program, E, A>
    where
        E: ExecutionPolicy,
        A: RuleAttemptPolicy,
    {
        let Self { attempt, miss } = self;
        match miss {
            EvaluatedRuleMiss::AlwaysRewriteStateMismatch(rule) => {
                BorrowedContinuingRuleAttemptTransition::AlwaysRewriteStateMismatch(
                    BorrowedAlwaysRewriteStateMismatchRuleAttempt {
                        attempt,
                        rule,
                        cursor,
                    },
                )
            }
            EvaluatedRuleMiss::OnceRewriteStateMismatch(rule) => {
                BorrowedContinuingRuleAttemptTransition::OnceRewriteStateMismatch(
                    BorrowedOnceRewriteStateMismatchRuleAttempt {
                        attempt,
                        rule,
                        cursor,
                    },
                )
            }
            EvaluatedRuleMiss::AlwaysReturnStateMismatch(rule) => {
                BorrowedContinuingRuleAttemptTransition::AlwaysReturnStateMismatch(
                    BorrowedAlwaysReturnStateMismatchRuleAttempt {
                        attempt,
                        rule,
                        cursor,
                    },
                )
            }
            EvaluatedRuleMiss::OnceReturnStateMismatch(rule) => {
                BorrowedContinuingRuleAttemptTransition::OnceReturnStateMismatch(
                    BorrowedOnceReturnStateMismatchRuleAttempt {
                        attempt,
                        rule,
                        cursor,
                    },
                )
            }
            EvaluatedRuleMiss::OnceRewriteConsumed(rule) => {
                BorrowedContinuingRuleAttemptTransition::OnceRewriteConsumed(
                    BorrowedOnceRewriteConsumedRuleAttempt {
                        attempt,
                        rule,
                        cursor,
                    },
                )
            }
        }
    }

    /// Projects a committed final miss into its exact stable terminal transition.
    fn into_final_transition<E, A>(
        self,
        program: &'program ExecutableProgram,
        core: TerminalRunCore,
    ) -> BorrowedFinalRuleAttemptTransition<'program, E, A>
    where
        E: ExecutionPolicy,
        A: RuleAttemptPolicy,
    {
        let Self { attempt, miss } = self;
        match miss {
            EvaluatedRuleMiss::AlwaysRewriteStateMismatch(rule) => {
                BorrowedFinalRuleAttemptTransition::StableAfterAlwaysRewriteStateMismatch(
                    BorrowedRuleAttemptStableAfterAlwaysRewriteStateMismatch {
                        attempts: attempt,
                        rule,
                        program,
                        core,
                    },
                )
            }
            EvaluatedRuleMiss::OnceRewriteStateMismatch(rule) => {
                BorrowedFinalRuleAttemptTransition::StableAfterOnceRewriteStateMismatch(
                    BorrowedRuleAttemptStableAfterOnceRewriteStateMismatch {
                        attempts: attempt,
                        rule,
                        program,
                        core,
                    },
                )
            }
            EvaluatedRuleMiss::AlwaysReturnStateMismatch(rule) => {
                BorrowedFinalRuleAttemptTransition::StableAfterAlwaysReturnStateMismatch(
                    BorrowedRuleAttemptStableAfterAlwaysReturnStateMismatch {
                        attempts: attempt,
                        rule,
                        program,
                        core,
                    },
                )
            }
            EvaluatedRuleMiss::OnceReturnStateMismatch(rule) => {
                BorrowedFinalRuleAttemptTransition::StableAfterOnceReturnStateMismatch(
                    BorrowedRuleAttemptStableAfterOnceReturnStateMismatch {
                        attempts: attempt,
                        rule,
                        program,
                        core,
                    },
                )
            }
            EvaluatedRuleMiss::OnceRewriteConsumed(rule) => {
                BorrowedFinalRuleAttemptTransition::StableAfterOnceRewriteConsumed(
                    BorrowedRuleAttemptStableAfterOnceRewriteConsumed {
                        attempts: attempt,
                        rule,
                        program,
                        core,
                    },
                )
            }
        }
    }
}

impl<'program, 'once, 'attempt_budget, 'step_budget, E, A>
    PreparedRuleAttemptApplication<'program, 'once, 'attempt_budget, 'step_budget, E, A>
where
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
{
    /// Couples a reserved attempt with a prepared runtime step.
    fn new(
        attempt: RuleAttemptReservation<'attempt_budget, A>,
        step: PreparedRuleStep<'program, 'once, 'step_budget, E>,
    ) -> Self {
        Self { attempt, step }
    }

    /// Commits the attempt before committing runtime step side effects.
    fn commit(
        self,
        state: &mut State,
        scratch: &mut RewriteScratch,
    ) -> CommittedRuleAttemptApplication<'program> {
        CommittedRuleAttemptApplication {
            attempt: self.attempt.commit(),
            applied: self.step.commit(state, scratch),
        }
    }
}

impl<'program> CommittedRuleAttemptApplication<'program> {
    /// Projects a committed continuing application into its exact public transition.
    fn into_continuing_transition<E, A>(
        self,
        program: &'program ExecutableProgram,
        parts: AttemptRunCoreParts<E, A>,
        pass: ContinuingRuntimeRulePass<'program>,
    ) -> BorrowedContinuingRuleAttemptTransition<'program, E, A>
    where
        E: ExecutionPolicy,
        A: RuleAttemptPolicy,
    {
        let Self { attempt, applied } = self;
        match applied {
            AppliedRule::AlwaysRewritten(committed) => {
                let step = committed.step();
                let rule = committed.rule();
                let cursor = BorrowedRuleAttemptCursor::from_runtime_pass(
                    program,
                    parts,
                    pass.reset_after_rewrite(),
                );
                BorrowedContinuingRuleAttemptTransition::AlwaysRewritten(
                    BorrowedRuleAttemptAlwaysRewriteStep {
                        attempt,
                        step,
                        rule,
                        cursor,
                    },
                )
            }
            AppliedRule::OnceRewritten(committed) => {
                let step = committed.step();
                let rule = committed.rule();
                let cursor = BorrowedRuleAttemptCursor::from_runtime_pass(
                    program,
                    parts,
                    pass.reset_after_rewrite(),
                );
                BorrowedContinuingRuleAttemptTransition::OnceRewritten(
                    BorrowedRuleAttemptOnceRewriteStep {
                        attempt,
                        step,
                        rule,
                        cursor,
                    },
                )
            }
            AppliedRule::AlwaysReturned(committed) => {
                let step = committed.step();
                let rule = committed.rule();
                let output = committed.into_output();
                BorrowedContinuingRuleAttemptTransition::AlwaysReturned(
                    BorrowedRuleAttemptAlwaysReturnRun {
                        attempt,
                        step,
                        rule,
                        program,
                        output,
                    },
                )
            }
            AppliedRule::OnceReturned(committed) => {
                let step = committed.step();
                let rule = committed.rule();
                let output = committed.into_output();
                BorrowedContinuingRuleAttemptTransition::OnceReturned(
                    BorrowedRuleAttemptOnceReturnRun {
                        attempt,
                        step,
                        rule,
                        program,
                        output,
                    },
                )
            }
        }
    }

    /// Projects a committed final application into its exact public transition.
    fn into_final_transition<E, A>(
        self,
        program: &'program ExecutableProgram,
        parts: AttemptRunCoreParts<E, A>,
        pass: FinalRuntimeRulePass<'program>,
    ) -> BorrowedFinalRuleAttemptTransition<'program, E, A>
    where
        E: ExecutionPolicy,
        A: RuleAttemptPolicy,
    {
        let Self { attempt, applied } = self;
        match applied {
            AppliedRule::AlwaysRewritten(committed) => {
                let step = committed.step();
                let rule = committed.rule();
                let cursor = BorrowedRuleAttemptCursor::from_runtime_pass(
                    program,
                    parts,
                    pass.reset_after_rewrite(),
                );
                BorrowedFinalRuleAttemptTransition::AlwaysRewritten(
                    BorrowedRuleAttemptAlwaysRewriteStep {
                        attempt,
                        step,
                        rule,
                        cursor,
                    },
                )
            }
            AppliedRule::OnceRewritten(committed) => {
                let step = committed.step();
                let rule = committed.rule();
                let cursor = BorrowedRuleAttemptCursor::from_runtime_pass(
                    program,
                    parts,
                    pass.reset_after_rewrite(),
                );
                BorrowedFinalRuleAttemptTransition::OnceRewritten(
                    BorrowedRuleAttemptOnceRewriteStep {
                        attempt,
                        step,
                        rule,
                        cursor,
                    },
                )
            }
            AppliedRule::AlwaysReturned(committed) => {
                let step = committed.step();
                let rule = committed.rule();
                let output = committed.into_output();
                BorrowedFinalRuleAttemptTransition::AlwaysReturned(
                    BorrowedRuleAttemptAlwaysReturnRun {
                        attempt,
                        step,
                        rule,
                        program,
                        output,
                    },
                )
            }
            AppliedRule::OnceReturned(committed) => {
                let step = committed.step();
                let rule = committed.rule();
                let output = committed.into_output();
                BorrowedFinalRuleAttemptTransition::OnceReturned(BorrowedRuleAttemptOnceReturnRun {
                    attempt,
                    step,
                    rule,
                    program,
                    output,
                })
            }
        }
    }
}

/// Advances one continuing rule-attempt pass without erasing its destination shape.
fn advance_continuing_rule_attempt<'program, E, A>(
    program: &'program ExecutableProgram,
    mut parts: AttemptRunCoreParts<E, A>,
    mut pass: ContinuingRuntimeRulePass<'program>,
) -> BorrowedContinuingRuleAttemptTransition<'program, E, A>
where
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
{
    let reservation =
        match reserve_next_rule_attempt(&mut parts.attempt_budget, parts.state.byte_count()) {
            Ok(reservation) => reservation,
            Err(error) => return failed_continuing_rule_attempt(program, parts, error),
        };

    match pass.attempt_current_rule(&parts.state) {
        RuleAttemptEvaluation::Miss(miss) => {
            let committed = CommittedRuleAttemptMiss::commit(reservation, miss);
            let cursor =
                BorrowedRuleAttemptCursor::from_runtime_pass(program, parts, pass.commit_miss());
            committed.into_continuing_transition(cursor)
        }
        RuleAttemptEvaluation::Matched(matched) => {
            let state_len = parts.state.byte_count();
            let prepared = match prepare_rule_attempt_application(
                &mut parts.scratch,
                &mut parts.budget,
                state_len,
                matched,
            ) {
                Ok(prepared) => prepared,
                Err(error) => return failed_continuing_rule_attempt(program, parts, error),
            };
            let prepared = PreparedRuleAttemptApplication::new(reservation, prepared);
            let committed = prepared.commit(&mut parts.state, &mut parts.scratch);
            committed.into_continuing_transition(program, parts, pass)
        }
    }
}

/// Advances one final rule-attempt pass without erasing its destination shape.
fn advance_final_rule_attempt<'program, E, A>(
    program: &'program ExecutableProgram,
    mut parts: AttemptRunCoreParts<E, A>,
    mut pass: FinalRuntimeRulePass<'program>,
) -> BorrowedFinalRuleAttemptTransition<'program, E, A>
where
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
{
    let reservation =
        match reserve_next_rule_attempt(&mut parts.attempt_budget, parts.state.byte_count()) {
            Ok(reservation) => reservation,
            Err(error) => return failed_final_rule_attempt(program, parts, error),
        };

    match pass.attempt_current_rule(&parts.state) {
        RuleAttemptEvaluation::Miss(miss) => {
            let committed = CommittedRuleAttemptMiss::commit(reservation, miss);
            let core = parts.into_terminal();
            committed.into_final_transition(program, core)
        }
        RuleAttemptEvaluation::Matched(matched) => {
            let state_len = parts.state.byte_count();
            let prepared = match prepare_rule_attempt_application(
                &mut parts.scratch,
                &mut parts.budget,
                state_len,
                matched,
            ) {
                Ok(prepared) => prepared,
                Err(error) => return failed_final_rule_attempt(program, parts, error),
            };
            let prepared = PreparedRuleAttemptApplication::new(reservation, prepared);
            let committed = prepared.commit(&mut parts.state, &mut parts.scratch);
            committed.into_final_transition(program, parts, pass)
        }
    }
}

/// Reserves the next rule-attempt count without touching transition projection.
///
/// # Errors
///
/// Returns `RuleAttemptStepError` if the rule-attempt budget is exhausted or
/// the next attempt count cannot be represented.
fn reserve_next_rule_attempt<A>(
    attempt_budget: &mut RuleAttemptBudgetState<A>,
    state_len: RuntimeStateByteCount,
) -> Result<RuleAttemptReservation<'_, A>, RuleAttemptStepError>
where
    A: RuleAttemptPolicy,
{
    attempt_budget.reserve_next_attempt(state_len)
}

/// Prepares a matched rule-attempt application without committing progress.
///
/// # Errors
///
/// Returns `RuleAttemptStepError` if step reservation, rewrite preparation,
/// return-output materialization, or allocation fails.
fn prepare_rule_attempt_application<'program, 'once, 'budget, E>(
    scratch: &mut RewriteScratch,
    budget: &'budget mut RuntimeBudgetState<E>,
    state_len: RuntimeStateByteCount,
    matched: MatchedRuleApplication<'program, '_, 'once>,
) -> Result<PreparedRuleStep<'program, 'once, 'budget, E>, RuleAttemptStepError>
where
    E: ExecutionPolicy,
{
    prepare_matched_rule(scratch, budget, state_len, matched).map_err(Into::into)
}

/// Projects an uncommitted continuing-pass failure.
fn failed_continuing_rule_attempt<'program, E, A>(
    program: &'program ExecutableProgram,
    parts: AttemptRunCoreParts<E, A>,
    error: RuleAttemptStepError,
) -> BorrowedContinuingRuleAttemptTransition<'program, E, A>
where
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
{
    BorrowedContinuingRuleAttemptTransition::Failed(BorrowedRuleAttemptFailedRun::new(
        error,
        program,
        parts.into_failed_rule_attempt_terminal(),
    ))
}

/// Projects an uncommitted final-pass failure.
fn failed_final_rule_attempt<'program, E, A>(
    program: &'program ExecutableProgram,
    parts: AttemptRunCoreParts<E, A>,
    error: RuleAttemptStepError,
) -> BorrowedFinalRuleAttemptTransition<'program, E, A>
where
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
{
    BorrowedFinalRuleAttemptTransition::Failed(BorrowedRuleAttemptFailedRun::new(
        error,
        program,
        parts.into_failed_rule_attempt_terminal(),
    ))
}
