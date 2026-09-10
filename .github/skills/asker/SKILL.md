---
name: asker
description: 'Clarify vague coding requests before implementation by asking focused questions about technology stack, coding style, requirements, and UI or frontend design; enforce a three-retry error limit; provide precise post-coding change explanations.'
argument-hint: 'Describe the coding task and any constraints.'
user-invocable: true
disable-model-invocation: false
---

# Asker

## Purpose

Use this skill for coding, debugging, refactoring, and UI or frontend development tasks when the request may be underspecified. Its goal is to make assumptions explicit before implementation, keep debugging bounded, and leave the user with an accurate account of the changes.

## Workflow

1. Inspect the user's request and identify the concrete implementation surface, expected outcome, and available project context.
2. Decide whether clarification is needed. Ask questions when the request leaves a meaningful choice unresolved, especially about:
   - technology stack, runtime, framework, or dependency constraints;
   - coding style, naming, architecture, compatibility, or testing expectations;
   - functional requirements, inputs, outputs, edge cases, and acceptance criteria;
   - UI or frontend appearance, including layout, interaction, responsive behavior, accessibility, and visual direction.
3. Ask only the smallest sufficient set of focused questions. Group related questions and offer clear options when practical. Do not ask about UI details for a non-UI task unless they affect the implementation.
4. Summarize the answers and the resulting implementation assumptions. Ask the user to confirm the requirements before making code changes.
5. After confirmation, inspect the nearest owning code path, make the smallest coherent implementation, and preserve existing project conventions.
6. Validate the change with the narrowest relevant test, type check, lint, build, or runtime check available.
7. If validation or implementation reports an error, diagnose the current failure and retry a repair. Count each repair attempt. After the third retry, stop making changes and ask the user how to proceed. Include a short list of plausible methods, such as reverting the local approach, changing the dependency or API usage, adjusting the test expectation, or providing additional error context. Never silently continue with a fourth retry.
8. After coding is complete, explain precisely what was written or modified. Identify each affected file, its role, the important behavior or API changes, validation performed, and any remaining limitations or assumptions. Do not claim that a check passed unless it was actually run.

## Clarification Rules

- If the request is already specific and the repository establishes the relevant conventions, state the assumptions briefly and ask for confirmation rather than conducting an unnecessary interview.
- If the user answers partially, ask only for the unresolved decisions that block safe implementation.
- For visual work, confirm the target audience and required states or workflows when appearance requirements are vague; do not substitute a generic layout without confirmation.
- Treat an explicit user correction as the newest source of truth and update the requirement summary before proceeding.

## Completion Checklist

- Requirements and implementation assumptions were summarized and confirmed before editing.
- Changes stayed within the requested scope and followed local conventions.
- A focused executable validation was run after editing when available.
- Error repair attempts were counted and no fourth retry was made.
- The final explanation names affected files and describes behavior, validation, and residual risks precisely.
