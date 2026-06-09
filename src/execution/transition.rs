use crate::error::{RuleAttemptStepError, RunFinishError, RunStepError};
use crate::inspect::{
    AlwaysReturnRuleView, AlwaysRewriteRuleView, OnceReturnRuleView, OnceRewriteRuleView,
};
use crate::limits::{RuleAttemptCount, StepCount};
use crate::policy::{ExecutionPolicy, RuleAttemptPolicy};
use crate::program::{ExecutableProgram, ReturnOutput, ReturnOutputView, RunResult};
use crate::trace::RuntimeStateView;

use super::engine::{FailedRuleAttemptTerminal, TerminalRunCore};
use super::session::{BorrowedRuleAttemptCursor, BorrowedRunSession};

/// Result of advancing a borrowed run session once.
///
/// Only rewritten transitions carry a continuation session. Stable, returned,
/// and failed transitions are terminal.
pub enum BorrowedStepTransition<'program, E: ExecutionPolicy> {
    /// One reusable rewrite rule was applied and execution can continue.
    AlwaysRewritten(BorrowedAlwaysRewriteStep<'program, E>),
    /// One once-only rewrite rule was applied and execution can continue.
    OnceRewritten(BorrowedOnceRewriteStep<'program, E>),
    /// No rule matched the final runtime state.
    Stable(BorrowedStableRun<'program>),
    /// A matched reusable rule executed `(return)`.
    AlwaysReturned(BorrowedAlwaysReturnRun<'program>),
    /// A matched once-only rule executed `(return)`.
    OnceReturned(BorrowedOnceReturnRun<'program>),
    /// A matching rule failed before committing.
    Failed(BorrowedFailedRun<'program>),
}

/// One committed reusable rewrite in a borrowed session.
pub struct BorrowedAlwaysRewriteStep<'program, E: ExecutionPolicy> {
    /// Step number committed by this transition.
    pub(super) step: StepCount,
    /// Borrowed rewrite rule committed by this transition.
    pub(super) rule: AlwaysRewriteRuleView<'program>,
    /// Continuation session after the committed rule application.
    pub(super) session: BorrowedRunSession<'program, E>,
}

/// One committed once-only rewrite in a borrowed session.
pub struct BorrowedOnceRewriteStep<'program, E: ExecutionPolicy> {
    /// Step number committed by this transition.
    pub(super) step: StepCount,
    /// Borrowed rewrite rule committed by this transition.
    pub(super) rule: OnceRewriteRuleView<'program>,
    /// Continuation session after the committed rule application.
    pub(super) session: BorrowedRunSession<'program, E>,
}

/// Terminal borrowed run state reached by no matching rule.
pub struct BorrowedStableRun<'program> {
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Terminal runtime core containing the stable state.
    pub(super) core: TerminalRunCore,
}

/// Terminal borrowed run state reached by a reusable `(return)` rule.
pub struct BorrowedAlwaysReturnRun<'program> {
    /// Step number that executed the return action.
    pub(super) step: StepCount,
    /// Borrowed return rule committed by this transition.
    pub(super) rule: AlwaysReturnRuleView<'program>,
    /// Borrowed return output retained for trace event emission.
    pub(super) output_view: ReturnOutputView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Materialized return output produced by the committed return rule.
    pub(super) output: ReturnOutput,
}

/// Terminal borrowed run state reached by a once-only `(return)` rule.
pub struct BorrowedOnceReturnRun<'program> {
    /// Step number that executed the return action.
    pub(super) step: StepCount,
    /// Borrowed return rule committed by this transition.
    pub(super) rule: OnceReturnRuleView<'program>,
    /// Borrowed return output retained for trace event emission.
    pub(super) output_view: ReturnOutputView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Materialized return output produced by the committed return rule.
    pub(super) output: ReturnOutput,
}

/// Runtime failure that preserves uncommitted borrowed state for inspection.
pub struct BorrowedFailedRun<'program> {
    /// Runtime error that stopped the candidate step before commit.
    pub(super) error: RunStepError,
    /// Parsed program borrowed by the failed terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Uncommitted runtime core retained for diagnostic inspection.
    pub(super) core: TerminalRunCore,
}

/// Result of advancing a continuing borrowed rule-attempt session once.
///
/// This transition type has no stable variant because the current rule has a
/// successor. A non-applying rule must return a cursor pointing at the next
/// typed pass state.
pub enum BorrowedContinuingRuleAttemptTransition<'program, E: ExecutionPolicy, A: RuleAttemptPolicy>
{
    /// A reusable rewrite rule was available but did not match the current state.
    AlwaysRewriteStateMismatch(BorrowedAlwaysRewriteStateMismatchRuleAttempt<'program, E, A>),
    /// A once-only rewrite rule was available but did not match the current state.
    OnceRewriteStateMismatch(BorrowedOnceRewriteStateMismatchRuleAttempt<'program, E, A>),
    /// A reusable return rule was available but did not match the current state.
    AlwaysReturnStateMismatch(BorrowedAlwaysReturnStateMismatchRuleAttempt<'program, E, A>),
    /// A once-only return rule was available but did not match the current state.
    OnceReturnStateMismatch(BorrowedOnceReturnStateMismatchRuleAttempt<'program, E, A>),
    /// A once-only rewrite rule had already committed in this run.
    OnceRewriteConsumed(BorrowedOnceRewriteConsumedRuleAttempt<'program, E, A>),
    /// One reusable rewrite rule was applied and execution can continue.
    AlwaysRewritten(BorrowedRuleAttemptAlwaysRewriteStep<'program, E, A>),
    /// One once-only rewrite rule was applied and execution can continue.
    OnceRewritten(BorrowedRuleAttemptOnceRewriteStep<'program, E, A>),
    /// A matched reusable rule executed `(return)`.
    AlwaysReturned(BorrowedRuleAttemptAlwaysReturnRun<'program>),
    /// A matched once-only rule executed `(return)`.
    OnceReturned(BorrowedRuleAttemptOnceReturnRun<'program>),
    /// A matching rule failed before committing runtime state.
    Failed(BorrowedRuleAttemptFailedRun<'program>),
}

/// Result of advancing a final borrowed rule-attempt session once.
///
/// This transition type has no missed-continuation variant because a
/// non-applying final rule exhausts the pass and terminates as stable.
pub enum BorrowedFinalRuleAttemptTransition<'program, E: ExecutionPolicy, A: RuleAttemptPolicy> {
    /// The pass ended after a reusable rewrite rule did not match the current state.
    StableAfterAlwaysRewriteStateMismatch(
        BorrowedRuleAttemptStableAfterAlwaysRewriteStateMismatch<'program>,
    ),
    /// The pass ended after a once-only rewrite rule did not match the current state.
    StableAfterOnceRewriteStateMismatch(
        BorrowedRuleAttemptStableAfterOnceRewriteStateMismatch<'program>,
    ),
    /// The pass ended after a reusable return rule did not match the current state.
    StableAfterAlwaysReturnStateMismatch(
        BorrowedRuleAttemptStableAfterAlwaysReturnStateMismatch<'program>,
    ),
    /// The pass ended after a once-only return rule did not match the current state.
    StableAfterOnceReturnStateMismatch(
        BorrowedRuleAttemptStableAfterOnceReturnStateMismatch<'program>,
    ),
    /// The pass ended after a consumed once-only rewrite rule.
    StableAfterOnceRewriteConsumed(BorrowedRuleAttemptStableAfterOnceRewriteConsumed<'program>),
    /// One reusable rewrite rule was applied and execution can continue.
    AlwaysRewritten(BorrowedRuleAttemptAlwaysRewriteStep<'program, E, A>),
    /// One once-only rewrite rule was applied and execution can continue.
    OnceRewritten(BorrowedRuleAttemptOnceRewriteStep<'program, E, A>),
    /// A matched reusable rule executed `(return)`.
    AlwaysReturned(BorrowedRuleAttemptAlwaysReturnRun<'program>),
    /// A matched once-only rule executed `(return)`.
    OnceReturned(BorrowedRuleAttemptOnceReturnRun<'program>),
    /// A matching rule failed before committing runtime state.
    Failed(BorrowedRuleAttemptFailedRun<'program>),
}

/// Continuing rule-attempt miss for a reusable rewrite state mismatch.
pub struct BorrowedAlwaysRewriteStateMismatchRuleAttempt<
    'program,
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
> {
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Non-applying rule.
    pub(super) rule: AlwaysRewriteRuleView<'program>,
    /// Cursor after consuming the rule line.
    pub(super) cursor: BorrowedRuleAttemptCursor<'program, E, A>,
}

/// Continuing rule-attempt miss for a once-only rewrite state mismatch.
pub struct BorrowedOnceRewriteStateMismatchRuleAttempt<
    'program,
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
> {
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Non-applying rule.
    pub(super) rule: OnceRewriteRuleView<'program>,
    /// Cursor after consuming the rule line.
    pub(super) cursor: BorrowedRuleAttemptCursor<'program, E, A>,
}

/// Continuing rule-attempt miss for a reusable return state mismatch.
pub struct BorrowedAlwaysReturnStateMismatchRuleAttempt<
    'program,
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
> {
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Non-applying rule.
    pub(super) rule: AlwaysReturnRuleView<'program>,
    /// Cursor after consuming the rule line.
    pub(super) cursor: BorrowedRuleAttemptCursor<'program, E, A>,
}

/// Continuing rule-attempt miss for a once-only return state mismatch.
pub struct BorrowedOnceReturnStateMismatchRuleAttempt<
    'program,
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
> {
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Non-applying rule.
    pub(super) rule: OnceReturnRuleView<'program>,
    /// Cursor after consuming the rule line.
    pub(super) cursor: BorrowedRuleAttemptCursor<'program, E, A>,
}

/// Continuing rule-attempt miss for a consumed once-only rewrite rule.
pub struct BorrowedOnceRewriteConsumedRuleAttempt<
    'program,
    E: ExecutionPolicy,
    A: RuleAttemptPolicy,
> {
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Consumed once-only rule.
    pub(super) rule: OnceRewriteRuleView<'program>,
    /// Cursor after consuming the rule line.
    pub(super) cursor: BorrowedRuleAttemptCursor<'program, E, A>,
}

/// Terminal rule-attempt state after a reusable rewrite state mismatch.
pub struct BorrowedRuleAttemptStableAfterAlwaysRewriteStateMismatch<'program> {
    /// Rule attempts consumed before stability.
    pub(super) attempts: RuleAttemptCount,
    /// Final non-applying rule.
    pub(super) rule: AlwaysRewriteRuleView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Terminal runtime core containing the stable state.
    pub(super) core: TerminalRunCore,
}

/// Terminal rule-attempt state after a once-only rewrite state mismatch.
pub struct BorrowedRuleAttemptStableAfterOnceRewriteStateMismatch<'program> {
    /// Rule attempts consumed before stability.
    pub(super) attempts: RuleAttemptCount,
    /// Final non-applying rule.
    pub(super) rule: OnceRewriteRuleView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Terminal runtime core containing the stable state.
    pub(super) core: TerminalRunCore,
}

/// Terminal rule-attempt state after a reusable return state mismatch.
pub struct BorrowedRuleAttemptStableAfterAlwaysReturnStateMismatch<'program> {
    /// Rule attempts consumed before stability.
    pub(super) attempts: RuleAttemptCount,
    /// Final non-applying rule.
    pub(super) rule: AlwaysReturnRuleView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Terminal runtime core containing the stable state.
    pub(super) core: TerminalRunCore,
}

/// Terminal rule-attempt state after a once-only return state mismatch.
pub struct BorrowedRuleAttemptStableAfterOnceReturnStateMismatch<'program> {
    /// Rule attempts consumed before stability.
    pub(super) attempts: RuleAttemptCount,
    /// Final non-applying rule.
    pub(super) rule: OnceReturnRuleView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Terminal runtime core containing the stable state.
    pub(super) core: TerminalRunCore,
}

/// Terminal rule-attempt state after a consumed once-only rewrite rule.
pub struct BorrowedRuleAttemptStableAfterOnceRewriteConsumed<'program> {
    /// Rule attempts consumed before stability.
    pub(super) attempts: RuleAttemptCount,
    /// Final consumed once-only rule.
    pub(super) rule: OnceRewriteRuleView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Terminal runtime core containing the stable state.
    pub(super) core: TerminalRunCore,
}

/// One committed reusable rewrite in a borrowed rule-attempt session.
pub struct BorrowedRuleAttemptAlwaysRewriteStep<'program, E: ExecutionPolicy, A: RuleAttemptPolicy>
{
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Step number committed by this transition.
    pub(super) step: StepCount,
    /// Borrowed rewrite rule committed by this transition.
    pub(super) rule: AlwaysRewriteRuleView<'program>,
    /// Cursor after the committed rule application.
    pub(super) cursor: BorrowedRuleAttemptCursor<'program, E, A>,
}

/// One committed once-only rewrite in a borrowed rule-attempt session.
pub struct BorrowedRuleAttemptOnceRewriteStep<'program, E: ExecutionPolicy, A: RuleAttemptPolicy> {
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Step number committed by this transition.
    pub(super) step: StepCount,
    /// Borrowed rewrite rule committed by this transition.
    pub(super) rule: OnceRewriteRuleView<'program>,
    /// Cursor after the committed rule application.
    pub(super) cursor: BorrowedRuleAttemptCursor<'program, E, A>,
}

/// Terminal borrowed rule-attempt run state reached by a reusable `(return)` rule.
pub struct BorrowedRuleAttemptAlwaysReturnRun<'program> {
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Step number that executed the return action.
    pub(super) step: StepCount,
    /// Borrowed return rule committed by this transition.
    pub(super) rule: AlwaysReturnRuleView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Materialized return output produced by the committed return rule.
    pub(super) output: ReturnOutput,
}

/// Terminal borrowed rule-attempt run state reached by a once-only `(return)` rule.
pub struct BorrowedRuleAttemptOnceReturnRun<'program> {
    /// Rule-attempt count committed by this transition.
    pub(super) attempt: RuleAttemptCount,
    /// Step number that executed the return action.
    pub(super) step: StepCount,
    /// Borrowed return rule committed by this transition.
    pub(super) rule: OnceReturnRuleView<'program>,
    /// Parsed program borrowed by the terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Materialized return output produced by the committed return rule.
    pub(super) output: ReturnOutput,
}

/// Runtime failure that preserves uncommitted borrowed rule-attempt state for inspection.
pub struct BorrowedRuleAttemptFailedRun<'program> {
    /// Runtime error that stopped the candidate attempt before commit.
    pub(super) error: RuleAttemptStepError,
    /// Number of rule attempts consumed before the failure was reported.
    pub(super) attempts: RuleAttemptCount,
    /// Parsed program borrowed by the failed terminal state.
    pub(super) program: &'program ExecutableProgram,
    /// Uncommitted runtime core retained for diagnostic inspection.
    pub(super) core: TerminalRunCore,
}

/// Implements shared accessors for borrowed rewrite step witnesses.
macro_rules! impl_borrowed_rewrite_step {
    ($step:ident, $rule:ident) => {
        impl<'program, E: ExecutionPolicy> $step<'program, E> {
            /// One-based applied step count.
            #[must_use]
            pub const fn step(&self) -> StepCount {
                self.step
            }

            /// Borrowed rule committed by this transition.
            #[must_use]
            pub const fn rule(&self) -> $rule<'program> {
                self.rule
            }

            /// Runtime state after the applied step.
            #[must_use]
            pub fn state(&self) -> RuntimeStateView<'_> {
                self.session.state()
            }

            /// Continue running after observing this applied step.
            ///
            /// This is the only borrowed transition that can resume execution.
            #[must_use]
            pub fn into_session(self) -> BorrowedRunSession<'program, E> {
                self.session
            }
        }
    };
}

impl_borrowed_rewrite_step!(BorrowedAlwaysRewriteStep, AlwaysRewriteRuleView);
impl_borrowed_rewrite_step!(BorrowedOnceRewriteStep, OnceRewriteRuleView);

/// Implements shared accessors for continuing rule-attempt misses.
macro_rules! impl_borrowed_rule_attempt_miss {
    ($miss:ident, $rule:ident) => {
        impl<'program, E: ExecutionPolicy, A: RuleAttemptPolicy> $miss<'program, E, A> {
            /// One-based consumed rule-attempt count.
            #[must_use]
            pub const fn attempt(&self) -> RuleAttemptCount {
                self.attempt
            }

            /// Non-applying rule consumed by this rule attempt.
            #[must_use]
            pub const fn rule(&self) -> $rule<'program> {
                self.rule
            }

            /// Runtime state after the non-applying rule attempt.
            #[must_use]
            pub fn state(&self) -> RuntimeStateView<'_> {
                self.cursor.state()
            }

            /// Continue running after observing this non-applying rule attempt.
            #[must_use]
            pub fn into_cursor(self) -> BorrowedRuleAttemptCursor<'program, E, A> {
                self.cursor
            }
        }
    };
}

impl_borrowed_rule_attempt_miss!(
    BorrowedAlwaysRewriteStateMismatchRuleAttempt,
    AlwaysRewriteRuleView
);
impl_borrowed_rule_attempt_miss!(
    BorrowedOnceRewriteStateMismatchRuleAttempt,
    OnceRewriteRuleView
);
impl_borrowed_rule_attempt_miss!(
    BorrowedAlwaysReturnStateMismatchRuleAttempt,
    AlwaysReturnRuleView
);
impl_borrowed_rule_attempt_miss!(
    BorrowedOnceReturnStateMismatchRuleAttempt,
    OnceReturnRuleView
);
impl_borrowed_rule_attempt_miss!(BorrowedOnceRewriteConsumedRuleAttempt, OnceRewriteRuleView);

/// Implements shared accessors for borrowed rule-attempt rewrite witnesses.
macro_rules! impl_borrowed_rule_attempt_rewrite_step {
    ($step:ident, $rule:ident) => {
        impl<'program, E: ExecutionPolicy, A: RuleAttemptPolicy> $step<'program, E, A> {
            /// One-based consumed rule-attempt count.
            #[must_use]
            pub const fn attempt(&self) -> RuleAttemptCount {
                self.attempt
            }

            /// One-based applied step count.
            #[must_use]
            pub const fn step(&self) -> StepCount {
                self.step
            }

            /// Borrowed rule committed by this rule-attempt transition.
            #[must_use]
            pub const fn rule(&self) -> $rule<'program> {
                self.rule
            }

            /// Runtime state after the applied step.
            #[must_use]
            pub fn state(&self) -> RuntimeStateView<'_> {
                self.cursor.state()
            }

            /// Continue running after observing this applied rule attempt.
            #[must_use]
            pub fn into_cursor(self) -> BorrowedRuleAttemptCursor<'program, E, A> {
                self.cursor
            }
        }
    };
}

impl_borrowed_rule_attempt_rewrite_step!(
    BorrowedRuleAttemptAlwaysRewriteStep,
    AlwaysRewriteRuleView
);
impl_borrowed_rule_attempt_rewrite_step!(BorrowedRuleAttemptOnceRewriteStep, OnceRewriteRuleView);

impl<'program> BorrowedStableRun<'program> {
    /// Number of execution steps committed before reaching the stable state.
    #[must_use]
    pub const fn steps(&self) -> StepCount {
        self.core.completed_steps()
    }

    /// Borrow the parsed program used by this terminal state.
    #[must_use]
    pub const fn program(&self) -> &'program ExecutableProgram {
        self.program
    }

    /// Borrowed final runtime state.
    #[must_use]
    pub fn state(&self) -> RuntimeStateView<'_> {
        self.core.state()
    }

    /// Materializes this stable run as a run result.
    ///
    /// # Errors
    ///
    /// Returns `RunFinishError` if final state materialization cannot allocate.
    pub fn into_result(self) -> Result<RunResult, RunFinishError> {
        self.core.into_stable_result()
    }
}

/// Implements shared accessors for exact stable rule-attempt terminals.
macro_rules! impl_borrowed_rule_attempt_stable_miss {
    ($run:ident, $rule:ident) => {
        impl<'program> $run<'program> {
            /// Number of rule attempts consumed before reaching the stable state.
            #[must_use]
            pub const fn attempts(&self) -> RuleAttemptCount {
                self.attempts
            }

            /// Number of execution steps committed before reaching the stable state.
            #[must_use]
            pub const fn steps(&self) -> StepCount {
                self.core.completed_steps()
            }

            /// Final non-applying rule that exhausted this rule-attempt pass.
            #[must_use]
            pub const fn rule(&self) -> $rule<'program> {
                self.rule
            }

            /// Borrow the parsed program used by this terminal state.
            #[must_use]
            pub const fn program(&self) -> &'program ExecutableProgram {
                self.program
            }

            /// Borrowed final runtime state.
            #[must_use]
            pub fn state(&self) -> RuntimeStateView<'_> {
                self.core.state()
            }

            /// Materializes this stable run as a run result.
            ///
            /// # Errors
            ///
            /// Returns `RunFinishError` if final state materialization cannot allocate.
            pub fn into_result(self) -> Result<RunResult, RunFinishError> {
                self.core.into_stable_result()
            }
        }
    };
}

impl_borrowed_rule_attempt_stable_miss!(
    BorrowedRuleAttemptStableAfterAlwaysRewriteStateMismatch,
    AlwaysRewriteRuleView
);
impl_borrowed_rule_attempt_stable_miss!(
    BorrowedRuleAttemptStableAfterOnceRewriteStateMismatch,
    OnceRewriteRuleView
);
impl_borrowed_rule_attempt_stable_miss!(
    BorrowedRuleAttemptStableAfterAlwaysReturnStateMismatch,
    AlwaysReturnRuleView
);
impl_borrowed_rule_attempt_stable_miss!(
    BorrowedRuleAttemptStableAfterOnceReturnStateMismatch,
    OnceReturnRuleView
);
impl_borrowed_rule_attempt_stable_miss!(
    BorrowedRuleAttemptStableAfterOnceRewriteConsumed,
    OnceRewriteRuleView
);

/// Implements shared accessors for borrowed return terminal witnesses.
macro_rules! impl_borrowed_return_run {
    ($run:ident, $rule:ident) => {
        impl<'program> $run<'program> {
            /// One-based applied step count for the return rule.
            #[must_use]
            pub const fn step(&self) -> StepCount {
                self.step
            }

            /// Borrow the parsed program used by this terminal state.
            #[must_use]
            pub const fn program(&self) -> &'program ExecutableProgram {
                self.program
            }

            /// Borrowed return rule committed by this terminal state.
            #[must_use]
            pub const fn rule(&self) -> $rule<'program> {
                self.rule
            }

            /// Materialized return output from runtime execution.
            #[must_use]
            pub const fn output(&self) -> &ReturnOutput {
                &self.output
            }

            /// Materializes this returned run as a run result.
            #[must_use]
            pub fn into_result(self) -> RunResult {
                RunResult::from_return(self.output, self.step)
            }
        }
    };
}

/// Implements shared accessors for borrowed rule-attempt return terminal witnesses.
macro_rules! impl_borrowed_rule_attempt_return_run {
    ($run:ident, $rule:ident) => {
        impl<'program> $run<'program> {
            /// One-based consumed rule-attempt count.
            #[must_use]
            pub const fn attempt(&self) -> RuleAttemptCount {
                self.attempt
            }

            /// One-based applied step count for the return rule.
            #[must_use]
            pub const fn step(&self) -> StepCount {
                self.step
            }

            /// Borrow the parsed program used by this terminal state.
            #[must_use]
            pub const fn program(&self) -> &'program ExecutableProgram {
                self.program
            }

            /// Borrowed return rule committed by this terminal state.
            #[must_use]
            pub const fn rule(&self) -> $rule<'program> {
                self.rule
            }

            /// Materialized return output from runtime execution.
            #[must_use]
            pub const fn output(&self) -> &ReturnOutput {
                &self.output
            }

            /// Materializes this returned run as a run result.
            #[must_use]
            pub fn into_result(self) -> RunResult {
                RunResult::from_return(self.output, self.step)
            }
        }
    };
}

impl_borrowed_return_run!(BorrowedAlwaysReturnRun, AlwaysReturnRuleView);
impl_borrowed_return_run!(BorrowedOnceReturnRun, OnceReturnRuleView);
impl_borrowed_rule_attempt_return_run!(BorrowedRuleAttemptAlwaysReturnRun, AlwaysReturnRuleView);
impl_borrowed_rule_attempt_return_run!(BorrowedRuleAttemptOnceReturnRun, OnceReturnRuleView);

impl<'program> BorrowedFailedRun<'program> {
    /// Captures a failed borrowed session without committing the attempted step.
    pub(super) fn new(
        error: RunStepError,
        program: &'program ExecutableProgram,
        core: TerminalRunCore,
    ) -> Self {
        Self {
            error,
            program,
            core,
        }
    }

    /// Runtime error that prevented the step from committing.
    #[must_use]
    pub const fn error(&self) -> &RunStepError {
        &self.error
    }

    /// Number of execution steps that completed before the failed step attempt.
    #[must_use]
    pub const fn completed_steps(&self) -> StepCount {
        self.core.completed_steps()
    }

    /// Borrow the parsed program used by this failed session.
    #[must_use]
    pub fn program(&self) -> &'program ExecutableProgram {
        self.program
    }

    /// Borrow the uncommitted runtime state preserved by this error.
    #[must_use]
    pub fn state(&self) -> RuntimeStateView<'_> {
        self.core.state()
    }

    /// Discard the uncommitted run session and return the runtime error.
    ///
    /// Borrowed failed runs are terminal; there is no retryable borrowed
    /// continuation after an uncommitted failure.
    #[must_use]
    pub fn into_error(self) -> RunStepError {
        self.error
    }
}

impl core::fmt::Display for BorrowedFailedRun<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.error.fmt(formatter)
    }
}

impl core::error::Error for BorrowedFailedRun<'_> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.error)
    }
}

impl<'program> BorrowedRuleAttemptFailedRun<'program> {
    /// Captures a failed borrowed rule-attempt session without committing runtime state.
    pub(super) fn new(
        error: RuleAttemptStepError,
        program: &'program ExecutableProgram,
        terminal: FailedRuleAttemptTerminal,
    ) -> Self {
        Self {
            error,
            attempts: terminal.completed_attempts(),
            program,
            core: terminal.into_terminal_core(),
        }
    }

    /// Runtime error that prevented the rule attempt from completing.
    #[must_use]
    pub const fn error(&self) -> &RuleAttemptStepError {
        &self.error
    }

    /// Number of rule attempts consumed before the failure was reported.
    #[must_use]
    pub const fn completed_attempts(&self) -> RuleAttemptCount {
        self.attempts
    }

    /// Number of execution steps that completed before the failed rule attempt.
    #[must_use]
    pub const fn completed_steps(&self) -> StepCount {
        self.core.completed_steps()
    }

    /// Borrow the parsed program used by this failed session.
    #[must_use]
    pub fn program(&self) -> &'program ExecutableProgram {
        self.program
    }

    /// Borrow the uncommitted runtime state preserved by this error.
    #[must_use]
    pub fn state(&self) -> RuntimeStateView<'_> {
        self.core.state()
    }

    /// Discard the uncommitted run session and return the runtime error.
    #[must_use]
    pub fn into_error(self) -> RuleAttemptStepError {
        self.error
    }
}

impl core::fmt::Display for BorrowedRuleAttemptFailedRun<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.error.fmt(formatter)
    }
}

impl core::error::Error for BorrowedRuleAttemptFailedRun<'_> {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        Some(&self.error)
    }
}
