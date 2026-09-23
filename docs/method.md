# Method

WebKD distills a text-only teacher into a web agent at the states visited by the
student. The student checkpoint used for scoring and anchoring is frozen before
the distillation update.

## State-aligned, student-style editing

1. Collect trajectories from the student and identify unsuccessful episodes.
2. Ask the teacher to locate the first erroneous step.
3. Replay the original student actions. From the localized step onward, ask the
   teacher to write a corrected structured response at each observed state.
4. Condition the teacher on rationale exemplars from the student, preferring
   solved episodes and excluding the task being edited. Exemplars contain
   reasoning, plans, step indices, and memory; they exclude actions.
5. Keep the original actions and original response history in the replay. The
   teacher's corrections become training targets and do not determine subsequent
   states in this editing mode.

Each response contains `<think>`, `<plan>`, `<step>`, `<memory>`, and `<action>`.
State-consistency filtering checks claims in the corrected thinking and memory
against the goal and observations available up to that step. The implementation
uses rule-based claim extraction and text matching, not a semantic proof of
correctness. Filtering removes individual corrected steps; replayed prefix steps
remain unchanged.

Implementation: [edit](../src/webkd/edit.py), [style](../src/webkd/style.py),
[grounding](../src/webkd/grounding.py), and [filter](../src/webkd/filter.py).

## Action-informativeness weights

Split `<think>`, `<plan>`, and `<memory>` into rationale segments. For each segment
`s`, score the *same corrected action call* under the frozen student with and
without that segment:

```text
contribution(s) = log p_ref(action | context, full rationale)
                − log p_ref(action | context, rationale without s)
```

The action score sums log probabilities over the bare call inside `<action>`.
The scorer verifies that the action tokenization matches between the two inputs.
Within a response, average ranks map contributions to `[floor, 1]`. A dead-zone
shrinkage rule moves near-constant contributions toward uniform weights. Defaults
are `floor=0.1` and `tau=0.05` nats. Actions, step indices, section delimiters, and
response-envelope tokens retain unit cross-entropy weight.

Weights are stored as character spans, keyed by the exact assistant text. The
training tokenizer maps spans to tokens using offsets. Boundary-straddling tokens
receive a character-coverage-weighted average. Hard surgical masks, used by a
separate baseline, remove a token only when its entire span is masked.

Implementation: [segmentation and weighting](../src/webkd/action_informativeness.py),
[scoring](../src/webkd/score.py), [sidecar construction](../src/webkd/build_weights.py),
and [token alignment](../src/webkd/training/token_weights.py).

## Training objective

For supervised target tokens `y_i`, let `w_i` be action-informativeness weights,
`gamma_i` the role mask, and `p_ref` the frozen initial student:

```text
CE = sum_i w_i · (−log p_theta(y_i)) / sum_i w_i
d_i = clamp(log p_ref(y_i) − log p_theta(y_i), −20, 8)
A = sum_i gamma_i · (exp(d_i) − d_i − 1) / sum_i gamma_i
L_WebKD = CE + beta · A
```

This is the **implemented target-token k3 anchor**. Because it is evaluated on
teacher-written corpus tokens, it is not a full-vocabulary KL divergence or an
unbiased estimate obtained from student-policy samples. The reference is the
student before distillation, not the black-box teacher.

The rationale mask gives unit weight to thinking, plan, and memory sections.
Other response characters receive `1e-6` to remain compatible with the shared
positive-weight label alignment contract. Response-envelope tokens retain unit
anchor weight. The corrected action is therefore effectively unanchored while
remaining fully supervised by cross-entropy. CE weights and anchor masks travel
in separate columns and cannot silently substitute for one another.

The combined objective averages each term over its own support per microbatch.
Gradient accumulation and distributed training average those microbatch losses.
With variable-length examples, this differs from one global token-weighted mean.
The frozen reference is replicated per training process, including when the
trainable model uses ZeRO sharding; account for that memory separately.

Implementation: [anchor mask](../src/webkd/build_anchor.py) and
[trainer](../integrations/llamafactory/train/sft/trainer.py).
