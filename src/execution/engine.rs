use crate::error::{RunError, RunFinishError, RunStartError, TracedRunError};
use crate::input::AdmittedRun;
use crate::limits::{RuleAttemptCount, StepCount};
use crate::policy::{ExecutionPolicy, RuleAttemptPolicy};
use crate::program::{ExecutableProgram, RunResult};
use crate::runtime::action::{AppliedRule, prepare_matched_rule};
use crate::runtime::budget::{RuleAttemptBudgetState, RuntimeBudgetState};
use crate::runtime::once::{
    ContinuingRuntimeRulePass, FinalRuntimeRulePass, RuntimeRuleScan, RuntimeRuleTable,
};
use crate::runtime::rewrite::RewriteScratch;
use crate::runtime::state::State;
use crate::trace::{BorrowedTraceEvent, RuntimeStateView};

use super::session::BorrowedRunSession;
use super::transition::{
    BorrowedAlwaysReturnRun, BorrowedAlwaysRewriteStep, BorrowedFailedRun, BorrowedOnceReturnRun,
    BorrowedOnceRewriteStep, BorrowedStableRun, BorrowedStepTransition,
};

/// Active mutable runtime state tied to one borrowed executable program.
#[derive(Debug)]
pub(super) struct ActiveRunCore<'program, E: ExecutionPolicy> {
    /// Current runtime byte state.
    pub(super) state: State,
    /// Reusable buffer for candidate rewrites.
    pub(super) scratch: RewriteScratch,
    /// Runtime limits and completed-step count.
    pub(super) budget: RuntimeBudgetState<E>,
    /// Per-run executable rules with fresh/consumed once-cell variants.
    runtime_rules: RuntimeRuleTable<'program>,
}

/// Active mutable rule-attempt runtime state tied to a continuing pass.
#[derive(Debug)]
pub(super) struct ContinuingRuleAttemptCore<'program, E: ExecutionPolicy, A: RuleAttemptPolicy> {
    /// Pass-independent active runtime state.
    pub(super) parts: AttemptRunCoreParts<E, A>,
    /// Rule-attempt pass that owns current target and remaining scan state.
    pub(super) runtime_rules: ContinuingRuntimeRulePass<'program>,
}

/// Active mutable rule-attempt runtime state tied to a final pass.
#[derive(Debug)]
pub(super) struct FinalRuleAttemptCore<'program, E: ExecutionPolicy, A: RuleAttemptPolicy> {
    /// Pass-independent active runtime state.
    pub(super) parts: AttemptRunCoreParts<E, A>,
    /// Rule-attempt pass that owns the final current target.
    pub(super) runtime_rules: FinalRuntimeRulePass<'program>,
}

/// Active rule-attempt runtime state without a selected pass.
#[derive(Debug)]
pub(super) struct AttemptRunCoreParts<E: ExecutionPolicy, A: RuleAttemptPolicy> {
    /// Current runtime byte state.
    pub(super) state: State,
    /// Reusable buffer for candidate rewrites.
    pub(super) scratch: RewriteScratch,
    /// Runtime limits and completed-step count.
    pub(super) budget: RuntimeBudgetState<E>,
    /// Rule-attempt limits and completed-attempt count.
    pub(super) attempt_budget: RuleAttemptBudgetState<A>,
}

/// Terminal runtime state after active execution can no longer resume.
#[derive(Debug)]
pub(super) struct TerminalRunCore {
    /// Runtime byte state preserved for terminal observation.
    state: State,
    /// Steps committed before the terminal boundary.
    steps: StepCount,
}

/// Terminal rule-attempt failure state before the current attempt can commit.
#[derive(Debug)]
pub(super) struct FailedRuleAttemptTerminal {
    /// Rule attempts committed before the failed candidate.
    attempts: RuleAttemptCount,
    /// Terminal runtime core preserving uncommitted runtime state.
    core: TerminalRunCore,
}

/// Runtime session that borrows one executable program.
pub(super) struct Session<'program, E: ExecutionPolicy> {
    /// Borrowed parsed program.
    pub(super) program: &'program ExecutableProgram,
    /// Mutable execution state.
    pub(super) core: ActiveRunCore<'program, E>,
}

/// Runtime rule-attempt session whose current target has a successor.
pub(super) struct ContinuingRuleAttemptRun<'program, E: ExecutionPolicy, A: RuleAttemptPolicy> {
    /// Borrowed parsed program.
    pub(super) program: &'program ExecutableProgram,
    /// Mutable execution state.
    pub(super) core: ContinuingRuleAttemptCore<'program, E, A>,
}

/// Runtime rule-attempt session whose current target exhausts the pass.
pub(super) struct FinalRuleAttemptRun<'program, E: ExecutionPolicy, A: RuleAttemptPolicy> {
    /// Borrowed parsed program.
    pub(super) program: &'program ExecutableProgram,
    /// Mutable execution state.
    pub(super) core: FinalRuleAttemptCore<'program, E, A>,
}

impl<'program, E: ExecutionPolicy> ActiveRunCore<'program, E> {
    /// Builds the mutable runtime core for one execution.
    ///
    /// # Errors
    ///
    /// Returns `RunStartError` if per-run rule state allocation fails.
    fn new(
        program: &'program ExecutableProgram,
        admitted: AdmittedRun<E>,
    ) -> Result<Self, RunStartError> {
        let (input, budget) = admitted.into_runtime_parts();
        let state = State::from_input(input);
        let runtime_rules = RuntimeRuleTable::from_program(program)?;
        Ok(Self {
            state,
            scratch: RewriteScratch::new(),
            budget,
            runtime_rules,
        })
    }

    /// Number of steps already committed in this core.
    pub(super) const fn completed_steps(&self) -> StepCount {
        self.budget.completed_steps()
    }

    /// Borrows the current runtime state.
    pub(super) fn state(&self) -> RuntimeStateView<'_> {
        self.state.view()
    }
}

impl<E: ExecutionPolicy, A: RuleAttemptPolicy> AttemptRunCoreParts<E, A> {
    /// Builds pass-independent rule-attempt runtime state from admitted input.
    pub(super) fn new(admitted: AdmittedRun<E>) -> Self {
        let (input, budget) = admitted.into_runtime_parts();
        Self {
            state: State::from_input(input),
            scratch: RewriteScratch::new(),
            budget,
            attempt_budget: RuleAttemptBudgetState::new(),
        }
    }

    /// Number of steps already committed in these runtime parts.
    pub(super) const fn completed_steps(&self) -> StepCount {
        self.budget.completed_steps()
    }

    /// Number of rule attempts already consumed in these runtime parts.
    pub(super) const fn completed_attempts(&self) -> RuleAttemptCount {
        self.attempt_budget.completed_attempts()
    }

    /// Borrows the current runtime state.
    pub(super) fn state(&self) -> RuntimeStateView<'_> {
        self.state.view()
    }

    /// Converts active runtime state into terminal runtime state.
    pub(super) fn into_terminal(self) -> TerminalRunCore {
        let steps = self.completed_steps();
        TerminalRunCore {
            state: self.state,
            steps,
        }
    }

    /// Converts active runtime state into a terminal rule-attempt failure.
    pub(super) fn into_failed_rule_attempt_terminal(self) -> FailedRuleAttemptTerminal {
        let attempts = self.completed_attempts();
        FailedRuleAttemptTerminal {
            attempts,
            core: self.into_terminal(),
        }
    }
}

impl TerminalRunCore {
    /// Number of steps completed before the terminal boundary.
    pub(super) const fn completed_steps(&self) -> StepCount {
        self.steps
    }

    /// Borrows the terminal runtime state.
    pub(super) fn state(&self) -> RuntimeStateView<'_> {
        self.state.view()
    }

    /// Materializes this terminal core as a stable result.
    ///
    /// # Errors
    ///
    /// Returns `RunFinishError` if final state materialization cannot allocate.
    pub(super) fn into_stable_result(self) -> Result<RunResult, RunFinishError> {
        let output = self
            .state
            .into_snapshot()
            .map_err(RunFinishError::FinalOutput)?;
        Ok(RunResult::stable(output, self.steps))
    }
}

impl FailedRuleAttemptTerminal {
    /// Number of attempts committed before the failed candidate.
    pub(super) const fn completed_attempts(&self) -> RuleAttemptCount {
        self.attempts
    }

    /// Consumes the failure state into its terminal runtime core.
    pub(super) fn into_terminal_core(self) -> TerminalRunCore {
        self.core
    }
}

impl<'program, E: ExecutionPolicy> Session<'program, E> {
    /// Starts a new run session for a parsed program and admitted run witness.
    ///
    /// # Errors
    ///
    /// Returns `RunStartError` if allocating per-run rule state fails.
    pub(super) fn new(
        program: &'program ExecutableProgram,
        admitted: AdmittedRun<E>,
    ) -> Result<Self, RunStartError> {
        let core = ActiveRunCore::new(program, admitted)?;
        Ok(Self { program, core })
    }

    /// Borrows the parsed program.
    pub(super) const fn program(&self) -> &'program ExecutableProgram {
        self.program
    }

    /// Number of execution steps that have already completed in this run.
    pub(super) const fn completed_steps(&self) -> StepCount {
        self.core.completed_steps()
    }

    /// Borrow the current runtime state.
    pub(super) fn state(&self) -> RuntimeStateView<'_> {
        self.core.state()
    }

    /// Consumes this session and advances it by one ordinary execution step.
    ///
    /// This is the only ordinary execution path that combines the parsed
    /// program scan with run-local once-cell state.
    ///
    /// # Errors
    ///
    /// Failed preparation returns a terminal transition that preserves
    /// uncommitted runtime state.
    pub(super) fn advance_run_step(self) -> BorrowedStepTransition<'program, E> {
        let Session { program, core } = self;
        let ActiveRunCore {
            mut state,
            mut scratch,
            mut budget,
            mut runtime_rules,
        } = core;

        let matched = match runtime_rules.scan_for_match(&state) {
            RuntimeRuleScan::Matched(matched) => matched,
            RuntimeRuleScan::Unmatched => {
                return BorrowedStepTransition::Stable(BorrowedStableRun {
                    program,
                    core: TerminalRunCore {
                        state,
                        steps: budget.completed_steps(),
                    },
                });
            }
        };
        let state_len = state.byte_count();
        let prepared = match prepare_matched_rule(&mut scratch, &mut budget, state_len, matched) {
            Ok(prepared) => prepared,
            Err(error) => {
                return BorrowedStepTransition::Failed(BorrowedFailedRun::new(
                    error,
                    program,
                    TerminalRunCore {
                        state,
                        steps: budget.completed_steps(),
                    },
                ));
            }
        };
        let applied = prepared.commit(&mut state, &mut scratch);
        let core = ActiveRunCore {
            state,
            scratch,
            budget,
            runtime_rules,
        };
        committed_run_transition(program, core, applied)
    }

    /// Runs this session to completion.
    ///
    /// # Errors
    ///
    /// Returns `RunFinishError` when a later matching rule would exceed configured
    /// limits.
    pub(super) fn finish(self) -> Result<RunResult, RunFinishError> {
        let mut session = self;
        loop {
            match session.advance_run_step() {
                BorrowedStepTransition::AlwaysRewritten(rewrite) => {
                    session = rewrite.into_session().session;
                }
                BorrowedStepTransition::OnceRewritten(rewrite) => {
                    session = rewrite.into_session().session;
                }
                BorrowedStepTransition::AlwaysReturned(returned) => {
                    return Ok(returned.into_result());
                }
                BorrowedStepTransition::OnceReturned(returned) => return Ok(returned.into_result()),
                BorrowedStepTransition::Stable(stable) => return stable.into_result(),
                BorrowedStepTransition::Failed(failure) => {
                    return Err(RunFinishError::from(failure.into_error()));
                }
            }
        }
    }
}

/// Builds the canonical ordinary transition for one committed runtime action.
fn committed_run_transition<'program, E>(
    program: &'program ExecutableProgram,
    core: ActiveRunCore<'program, E>,
    applied: AppliedRule<'program>,
) -> BorrowedStepTransition<'program, E>
where
    E: ExecutionPolicy,
{
    match applied {
        AppliedRule::AlwaysRewritten(committed) => {
            BorrowedStepTransition::AlwaysRewritten(BorrowedAlwaysRewriteStep {
                step: committed.step(),
                rule: committed.rule(),
                session: BorrowedRunSession {
                    session: Session { program, core },
                },
            })
        }
        AppliedRule::OnceRewritten(committed) => {
            BorrowedStepTransition::OnceRewritten(BorrowedOnceRewriteStep {
                step: committed.step(),
                rule: committed.rule(),
                session: BorrowedRunSession {
                    session: Session { program, core },
                },
            })
        }
        AppliedRule::AlwaysReturned(committed) => {
            let step = committed.step();
            let rule = committed.rule();
            let output_view = committed.output_view();
            let output = committed.into_output();
            BorrowedStepTransition::AlwaysReturned(BorrowedAlwaysReturnRun {
                step,
                rule,
                output_view,
                program,
                output,
            })
        }
        AppliedRule::OnceReturned(committed) => {
            let step = committed.step();
            let rule = committed.rule();
            let output_view = committed.output_view();
            let output = committed.into_output();
            BorrowedStepTransition::OnceReturned(BorrowedOnceReturnRun {
                step,
                rule,
                output_view,
                program,
                output,
            })
        }
    }
}

impl<'program, E: ExecutionPolicy> Session<'program, E> {
    /// Runs to completion while emitting borrowed trace events.
    ///
    /// # Errors
    ///
    /// Returns `TracedRunError::Trace` if the trace sink fails. Returns
    /// `TracedRunError::Run` if runtime execution fails.
    pub(super) fn trace_events<F, TraceError>(
        self,
        mut trace: F,
    ) -> Result<RunResult, TracedRunError<TraceError>>
    where
        F: for<'run> FnMut(BorrowedTraceEvent<'program, 'run>) -> Result<(), TraceError>,
    {
        trace(BorrowedTraceEvent::Initial {
            state: self.state(),
        })
        .map_err(TracedRunError::Trace)?;

        let mut session = self;
        loop {
            match session.advance_run_step() {
                BorrowedStepTransition::AlwaysRewritten(rewrite) => {
                    trace(BorrowedTraceEvent::AlwaysRewritten {
                        step: rewrite.step(),
                        rule: rewrite.rule(),
                        state: rewrite.state(),
                    })
                    .map_err(TracedRunError::Trace)?;
                    session = rewrite.into_session().session;
                }
                BorrowedStepTransition::OnceRewritten(rewrite) => {
                    trace(BorrowedTraceEvent::OnceRewritten {
                        step: rewrite.step(),
                        rule: rewrite.rule(),
                        state: rewrite.state(),
                    })
                    .map_err(TracedRunError::Trace)?;
                    session = rewrite.into_session().session;
                }
                BorrowedStepTransition::AlwaysReturned(returned) => {
                    trace(BorrowedTraceEvent::AlwaysReturned {
                        step: returned.step,
                        rule: returned.rule,
                        output: returned.output_view,
                    })
                    .map_err(TracedRunError::Trace)?;
                    return Ok(returned.into_result());
                }
                BorrowedStepTransition::OnceReturned(returned) => {
                    trace(BorrowedTraceEvent::OnceReturned {
                        step: returned.step,
                        rule: returned.rule,
                        output: returned.output_view,
                    })
                    .map_err(TracedRunError::Trace)?;
                    return Ok(returned.into_result());
                }
                BorrowedStepTransition::Stable(stable) => {
                    return stable
                        .into_result()
                        .map_err(RunError::from)
                        .map_err(TracedRunError::Run);
                }
                BorrowedStepTransition::Failed(failure) => {
                    return Err(TracedRunError::Run(RunError::from(RunFinishError::from(
                        failure.into_error(),
                    ))));
                }
            }
        }
    }
}
