use crate::bytes::RuntimeStateByteCount;
use crate::error::RuleAttemptStepError;
use crate::policy::{ExecutionPolicy, RuleAttemptPolicy};
use crate::program::ExecutableProgram;
use crate::runtime::action::{AppliedRule, PreparedRuleStep, prepare_matched_rule};
use crate::runtime::budget::{RuleAttemptBudgetState, RuleAttemptReservation, RuntimeBudgetState};
use crate::runtime::matcher::{EvaluatedRuleMiss, MatchedRuleApplication, RuleAttemptEvaluation};
use crate::runtime::once::{ContinuingRuntimeRulePass, FinalRuntimeRulePass};
use crate::runtime::rewrite::RewriteScratch;

use super::engine::{
    AttemptRunCoreParts, ContinuingRuleAttemptCore, ContinuingRuleAttemptRun, FinalRuleAttemptCore,
    FinalRuleAttemptRun,
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
            match committed.into_miss() {
                EvaluatedRuleMiss::AlwaysRewriteStateMismatch(rule) => {
                    BorrowedContinuingRuleAttemptTransition::AlwaysRewriteStateMismatch(
                        BorrowedAlwaysRewriteStateMismatchRuleAttempt {
                            attempt: committed.attempt(),
                            rule,
                            cursor,
                        },
                    )
                }
                EvaluatedRuleMiss::OnceRewriteStateMismatch(rule) => {
                    BorrowedContinuingRuleAttemptTransition::OnceRewriteStateMismatch(
                        BorrowedOnceRewriteStateMismatchRuleAttempt {
                            attempt: committed.attempt(),
                            rule,
                            cursor,
                        },
                    )
                }
                EvaluatedRuleMiss::AlwaysReturnStateMismatch(rule) => {
                    BorrowedContinuingRuleAttemptTransition::AlwaysReturnStateMismatch(
                        BorrowedAlwaysReturnStateMismatchRuleAttempt {
                            attempt: committed.attempt(),
                            rule,
                            cursor,
                        },
                    )
                }
                EvaluatedRuleMiss::OnceReturnStateMismatch(rule) => {
                    BorrowedContinuingRuleAttemptTransition::OnceReturnStateMismatch(
                        BorrowedOnceReturnStateMismatchRuleAttempt {
                            attempt: committed.attempt(),
                            rule,
                            cursor,
                        },
                    )
                }
                EvaluatedRuleMiss::OnceRewriteConsumed(rule) => {
                    BorrowedContinuingRuleAttemptTransition::OnceRewriteConsumed(
                        BorrowedOnceRewriteConsumedRuleAttempt {
                            attempt: committed.attempt(),
                            rule,
                            cursor,
                        },
                    )
                }
            }
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
            match committed.into_applied() {
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
                            attempt: committed.attempt(),
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
                            attempt: committed.attempt(),
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
                            attempt: committed.attempt(),
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
                            attempt: committed.attempt(),
                            step,
                            rule,
                            program,
                            output,
                        },
                    )
                }
            }
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
            match committed.into_miss() {
                EvaluatedRuleMiss::AlwaysRewriteStateMismatch(rule) => {
                    BorrowedFinalRuleAttemptTransition::StableAfterAlwaysRewriteStateMismatch(
                        BorrowedRuleAttemptStableAfterAlwaysRewriteStateMismatch {
                            attempts: committed.attempt(),
                            rule,
                            program,
                            core,
                        },
                    )
                }
                EvaluatedRuleMiss::OnceRewriteStateMismatch(rule) => {
                    BorrowedFinalRuleAttemptTransition::StableAfterOnceRewriteStateMismatch(
                        BorrowedRuleAttemptStableAfterOnceRewriteStateMismatch {
                            attempts: committed.attempt(),
                            rule,
                            program,
                            core,
                        },
                    )
                }
                EvaluatedRuleMiss::AlwaysReturnStateMismatch(rule) => {
                    BorrowedFinalRuleAttemptTransition::StableAfterAlwaysReturnStateMismatch(
                        BorrowedRuleAttemptStableAfterAlwaysReturnStateMismatch {
                            attempts: committed.attempt(),
                            rule,
                            program,
                            core,
                        },
                    )
                }
                EvaluatedRuleMiss::OnceReturnStateMismatch(rule) => {
                    BorrowedFinalRuleAttemptTransition::StableAfterOnceReturnStateMismatch(
                        BorrowedRuleAttemptStableAfterOnceReturnStateMismatch {
                            attempts: committed.attempt(),
                            rule,
                            program,
                            core,
                        },
                    )
                }
                EvaluatedRuleMiss::OnceRewriteConsumed(rule) => {
                    BorrowedFinalRuleAttemptTransition::StableAfterOnceRewriteConsumed(
                        BorrowedRuleAttemptStableAfterOnceRewriteConsumed {
                            attempts: committed.attempt(),
                            rule,
                            program,
                            core,
                        },
                    )
                }
            }
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
            match committed.into_applied() {
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
                            attempt: committed.attempt(),
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
                            attempt: committed.attempt(),
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
                            attempt: committed.attempt(),
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
                    BorrowedFinalRuleAttemptTransition::OnceReturned(
                        BorrowedRuleAttemptOnceReturnRun {
                            attempt: committed.attempt(),
                            step,
                            rule,
                            program,
                            output,
                        },
                    )
                }
            }
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
    let attempts = parts.completed_attempts();
    BorrowedContinuingRuleAttemptTransition::Failed(BorrowedRuleAttemptFailedRun::new(
        error,
        attempts,
        program,
        parts.into_terminal(),
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
    let attempts = parts.completed_attempts();
    BorrowedFinalRuleAttemptTransition::Failed(BorrowedRuleAttemptFailedRun::new(
        error,
        attempts,
        program,
        parts.into_terminal(),
    ))
}
