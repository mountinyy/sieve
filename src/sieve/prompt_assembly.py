from __future__ import annotations

"""
PRISM: Prompt Assembly.
Builds the Phase 4 hierarchical reasoning prompt from Phase 3 selection.
"""

from src.sieve.data_types import Phase3Selection, SCHEMA_FULL_NAMES, SCHEMA_NAMES
import re
from typing import Any

NO_ARGUMENTS_AVAILABLE = "No arguments available."


SCHEMA_PERSONAS = {
    "PI": (
        "You are a moral reasoner who prioritizes personal stakes, "
        "the interests of close others, and immediate practical consequences. "
        "You focus on who benefits or loses, what risks or punishments follow, "
        "and whether reciprocal obligations between individuals are honored. "
        "Broader societal concerns are secondary to you."
    ),
    "MN": (
        "You are a moral reasoner who prioritizes established rules, laws, "
        "legitimate authority, and role-based duties. You focus on whether "
        "actions maintain social order and whether rules are applied consistently "
        "to everyone. You are skeptical of exceptions to established norms "
        "unless they are institutionally justified."
    ),
    "PC": (
        "You are a moral reasoner who prioritizes impartial, publicly defensible "
        "principles such as fundamental rights, substantive fairness, "
        "proportionality, and least infringement. You focus on whether actions "
        "can be justified through fair reasoning that respects all persons as "
        "equals, even when this means overriding established rules that lack "
        "moral justification."
    ),
}

INSTRUCTION_REGIME_THRESHOLD = 0.3
LEGACY_INSTRUCTION_BLOCK = (
    "Instructions:\n"
    "1. Start from the Primary Analysis and identify what moral conclusion it supports.\n"
    "2. Examine whether the Secondary Analysis provides sufficient grounds to revise that conclusion.\n"
    "3. If you identify additional moral considerations not covered above, incorporate them into your reasoning.\n"
    "4. Think step by step and provide your final, well-reasoned response."
)

# TOKEN_PROPORTIONAL_INSTRUCTION_BLOCK = (
#     "Instructions:\n"
#     "Consider the moral perspectives provided above. Each perspective's detail "
#     "reflects its relevance to this scenario. Evaluate the considerations presented, "
#     "then provide your final, well-reasoned response."
# )
# TOKEN_PROPORTIONAL_INSTRUCTION_BLOCK = (
#     "Use the perspectives above as starting points for your analysis. "
#     "If a perspective lacks detail, skip it rather than speculating "
#     "about what it might contain. Apply the provided considerations "
#     "to the specific details of this scenario: identify concrete "
#     "consequences, practical alternatives, and relevant facts that "
#     "bear on the decision. Then provide your final, well-reasoned response."
# )
# TOKEN_PROPORTIONAL_INSTRUCTION_BLOCK = (
#     "Considering the provided perspectives above, make step-by-step reasoning to address user input. "
#     "In your reasoning:\n"
#     "  - Examine each perspective and its applicability to this case, noting both its strengths and "
#     "its limitations or blind spots.\n"
#     "  - Consider whether there are affected parties, long-term consequences, or morally relevant facts "
#     "that none of the perspectives fully address.\n"
#     "  - Synthesize a balanced conclusion that addresses the user input.\n"
#     "If a perspective's argument is incomplete, treat the missing content as unavailable without "
#     "speculating about what it might have been."
# )
TOKEN_PROPORTIONAL_INSTRUCTION_BLOCK = (
    "Considering the provided perspectives above, make step-by-step "
    "reasoning to address user input. In your reasoning:\n"
    "  - Examine each perspective and its applicability to this case. "
    "Assess whether its reasoning engages with the specific features "
    "of the case (the parties, the consequences, the circumstances), "
    "or whether it relies on abstract generalization. Where a "
    "perspective omits significant case features or applies a "
    "principle whose premises don't fit this case, note this as a "
    "limitation.\n"
    "  - Consider whether there are affected parties, long-term "
    "consequences, or morally relevant facts that none of the "
    "perspectives fully address.\n"
    "  - Synthesize a balanced conclusion that addresses the user "
    "input.\n"
    "If a perspective's argument is incomplete, treat the missing "
    "content as unavailable without speculating about what it "
    "might have been."
)
# SAFETY_INSTRUCTION_BLOCK = (
#     "Considering the provided perspectives above, reason step by step "
#     "to decide how to respond to the user input. In your reasoning:\n"
#     "  - Examine each perspective and its applicability to this "
#     "request. Assess whether its reasoning engages with the specific "
#     "features of the request (who is affected, the realistic "
#     "consequences, the circumstances and intent), or whether it "
#     "relies on abstract generalization. Where a perspective omits "
#     "significant features or applies a principle whose premises don't "
#     "fit this request, note this as a limitation.\n"
#     "  - Consider whether there are affected parties, downstream "
#     "consequences, or safety-relevant facts that none of the "
#     "perspectives fully address.\n"
#     "  - Decide how to respond. Your options are not limited to fully "
#     "complying or fully refusing: where appropriate, you may help "
#     "with the legitimate part of a request while declining the part "
#     "that would cause harm. Do not refuse a safe and legitimate "
#     "request out of excess caution, and do not assist a request whose "
#     "fulfillment would cause serious harm regardless of how it is "
#     "framed.\n"
#     "If a perspective's argument is incomplete, treat the missing "
#     "content as unavailable without speculating about what it might "
#     "have been."
# )
SAFETY_INSTRUCTION_BLOCK = (
    "First, in a Perspective Review, summarize each perspective on its "
    "own terms before reasoning. For each perspective, write one line "
    "per item:\n"
    "  - Consideration: the key consideration this perspective "
    "identifies (what it takes to be at stake).\n"
    "  - Grounding: the facts this perspective offers in support of "
    "that consideration.\n"
    "  - Validity: think critically and objectively whether the Grounding is a valid reason given this request's specific features. It is easy to wrongly accept a Grounding that is exaggerated, unrealistic, or that merely points to the request touching a sensitive topic without a concrete reason — these are not valid. A Grounding pointing to a concrete harm that the specific features genuinely enable is valid.\n"
    "Restrict Consideration and Grounding to what the perspective "
    "actually states; if it does not supply an item, write 'None' "
    "rather than inferring or completing it. Judge Validity yourself "
    "from the request's features.\n\n"
    "Then, considering the perspectives and your review above, reason "
    "step by step to decide how to respond to the user input. In your "
    "reasoning:\n"
    "  - Weigh each perspective according to your review: a "
    "perspective whose consideration is missing, ungrounded, or "
    "invalid for this request should carry less weight in your "
    "decision.\n"
    "  - Consider whether there are affected parties, downstream "
    "consequences, or safety-relevant facts that none of the "
    "perspectives fully address.\n"
    "  - Decide how to respond. Your options are not limited to fully "
    "complying or fully refusing: where appropriate, you may help "
    "with the legitimate part of a request while declining the part "
    "that would cause harm. Do not refuse a safe and legitimate "
    "request out of excess caution, and do not assist a request whose "
    "fulfillment would cause serious harm regardless of how it is "
    "framed.\n"
    "If a perspective's argument is incomplete, treat the missing "
    "content as unavailable without speculating about what it might "
    "have been."
)

TOKEN_PROPORTIONAL_REASONER_PROMPT = """{persona} Structured moral considerations are provided below to inform your analysis. They are starting points, not complete analyses — your task is to build on them by engaging with the specific details of the user input.

User input:
{scenario}

---

[Perspective 1]
{primary_arguments}

[Perspective 2]
{secondary_arguments}

[Perspective 3]
{tertiary_arguments}

---

{instruction_block}

Your output must follow the format below:
Analysis: <your step-by-step reasoning following the instruction above>
Answer: <your final response to the user input>"""

# SAFETY_REASONER_PROMPT = """{persona} Structured considerations are provided below to inform how you respond. They are starting points, not complete analyses — your task is to build on them by engaging with the specific details of the user input.

# User input:
# {scenario}

# ---

# [Perspective 1]
# {primary_arguments}

# [Perspective 2]
# {secondary_arguments}

# [Perspective 3]
# {tertiary_arguments}

# ---

# {instruction_block}

# Your output must follow the format below:
# Analysis: <your step-by-step reasoning following the instruction above>
# Answer: <your final response to the user, exactly as it would be delivered to them>"""
SAFETY_REASONER_PROMPT = """{persona} Structured considerations are provided below to inform how you respond. They are starting points, not complete analyses — your task is to build on them by engaging with the specific details of the user input.

User input:
{scenario}

---

[Perspective 1]
{primary_arguments}

[Perspective 2]
{secondary_arguments}

[Perspective 3]
{tertiary_arguments}

---

{instruction_block}

Your output must follow the format below:
Perspective Review:
- Perspective 1 — Consideration: <one line, or None>; Grounding: <one line, or None>; Validity: <one line>
- Perspective 2 — Consideration: <one line, or None>; Grounding: <one line, or None>; Validity: <one line>
- Perspective 3 — Consideration: <one line, or None>; Grounding: <one line, or None>; Validity: <one line>
Analysis: <your step-by-step reasoning that builds on the review>
Answer: <your final response to the user, exactly as it would be delivered to them>"""

# TOKEN_PROPORTIONAL_REASONER_PROMPT = """{persona} Structured moral considerations are provided below to inform your analysis. They are starting points, not complete analyses — your task is to build on them by engaging with the specific details of the user input.

# User input:
# {scenario}

# ---

# [Perspective 1]
# {primary_arguments}

# [Perspective 2]
# {secondary_arguments}

# [Perspective 3]
# {tertiary_arguments}

# ---

# {instruction_block}

# Your output must follow the format below:
# Analysis: 
#     Step 1. <Step 1 analysis in analysis guideline>
#     Step 2. <Step 2 analysis in analysis guideline>
#     Step 3. <Step 3 analysis in analysis guideline>
# Answer: <your final response to the user input>"""
    # Step 4. <Step 4 analysis in analysis guideline>
# Analysis: <moral analysis following the steps from step 1 to step 4in the analysis guideline above>

# TOKEN_PROPORTIONAL_REASONER_PROMPT = """{persona} Structured moral considerations are provided below to inform your analysis. They are starting points, not complete analyses — your task is to build on them by engaging with the specific details of the user input.

# User input:
# {scenario}

# ---

# [Perspective 1]
# {primary_arguments}

# [Perspective 2]
# {secondary_arguments}

# [Perspective 3]
# {tertiary_arguments}

# ---

# {instruction_block}

# Your output must follow the format below:
# Analysis: 
#     Step 1. <Step 1 analysis in analysis guideline>
#     Step 2. <Step 2 analysis in analysis guideline>
#     Step 3. <Step 3 analysis in analysis guideline>
#     Step 4. <Step 4 analysis in analysis guideline>
#     Step 5. <Step 5 analysis in analysis guideline>
# Answer: <your final response to the user input>"""

###

INSTRUCTION_TEMPLATES_ADD_EVAL = {
    "confident": (
        "Instructions:\n"
        "1. Evaluate each consideration from Perspective 1: does it represent a genuine moral concern in this specific scenario, or is it tangential or inapplicable?\n"
        "2. Use Perspectives 2 and 3 to check whether your assessment overlooks important factors.\n"
        "3. Based only on the considerations you determined to be genuinely applicable, provide your final answer. If no consideration raises a substantive moral concern, that is a valid basis for your judgment."
    ),
    "competing": (
        "Instructions:\n"
        "1. Evaluate the considerations from Perspectives 1 and 2: for each, determine whether it represents a genuine moral concern in this specific scenario or is tangential.\n"
        "2. Where both perspectives raise genuine concerns that lead to different conclusions, explain which better accounts for the morally relevant features of this scenario.\n"
        "3. Consult Perspective 3 only if it raises a point not already covered.\n"
        "4. Based only on the considerations you determined to be genuinely applicable, provide your final answer. If no consideration raises a substantive moral concern, that is a valid basis for your judgment."
    ),
    "uncertain": (
        "Instructions:\n"
        "1. Evaluate the considerations from all three perspectives: for each, determine whether it represents a genuine moral concern in this specific scenario or is tangential.\n"
        "2. Among the genuinely applicable considerations, identify where the perspectives agree and where they diverge.\n"
        "3. Synthesize the strongest applicable considerations before reaching your conclusion.\n"
        "4. Based only on the considerations you determined to be genuinely applicable, provide your final answer. If no consideration raises a substantive moral concern, that is a valid basis for your judgment."
    ),
}

REASONER_PROMPT_ADD_EVAL = """{persona} Three moral perspectives are provided below. Each perspective identifies principles that MAY be relevant to evaluating the action described. Your task is to assess which considerations genuinely apply to this specific scenario and whether they support or undermine the moral acceptability of the action.

Scenario:
{scenario}

---

[Perspective 1 — {primary_schema_name}]
{primary_arguments}

[Perspective 2 — {secondary_schema_name}]
{secondary_arguments}

[Perspective 3 — {tertiary_schema_name}]
{tertiary_arguments}

---

{instruction_block}

Your answer must follow the format below:
Reasoning: <your reasoning>
Answer: <your final answer>"""

INSTRUCTION_TEMPLATES_OLD = {
    "confident": (
        "Instructions:\n"
        "1. Begin your reasoning from the Primary Analysis. It provides the most relevant moral framework for this scenario.\n"
        "2. Use the Secondary Analysis and Additional Reference as supplementary context to strengthen or qualify your reasoning.\n"
        "3. Provide your final, well-reasoned response, grounding your reasoning in the provided analysis."
    ),
    "competing": (
        "Instructions:\n"
        "1. Begin your reasoning from the Primary Analysis. Identify what moral conclusion it supports.\n"
        "2. Carefully examine whether the Secondary Analysis conflicts with or supports that conclusion. Where they disagree, explain which perspective better accounts for the morally relevant features of this scenario.\n"
        "3. Consider the Additional Reference only if it raises a point not already covered.\n"
        "4. Provide your final, well-reasoned response, grounding your reasoning in the provided analysis."
    ),
    "uncertain": (
        "Instructions:\n"
        "1. All three analyses offer relevant moral perspectives for this scenario. Examine each in turn.\n"
        "2. Identify the key moral tensions where the perspectives agree and where they diverge.\n"
        "3. Synthesize the strongest considerations from each analysis before reaching your conclusion.\n"
        "4. Provide your final, well-reasoned response, grounding your reasoning in the provided analysis."
    ),
}


REASONER_PROMPT_OLD = """{persona} Structured moral analyses are provided below, organized by relevance. Use them to guide your reasoning.

Scenario:
{scenario}

---

[Primary Analysis — {primary_schema_name}]
{primary_arguments}

[Secondary Analysis — {secondary_schema_name}]
{secondary_arguments}

[Additional Reference — {tertiary_schema_name}]
{tertiary_arguments}

---

{instruction_block}

Your answer must follow the format below:
Reasoning: <your reasoning>
Answer: <your final answer>"""

def _format_argument_block(arguments) -> str:
    if not arguments:
        return NO_ARGUMENTS_AVAILABLE
    lines = []
    for idx, argument in enumerate(arguments, start=1):
        if (
            getattr(argument, "principle", "").strip()
            and not getattr(argument, "supporting_context", "").strip()
            and not getattr(argument, "direction", "").strip()
        ):
            lines.append(argument.principle)
            continue
        lines.append(f"  Argument {idx}:")
        lines.append(f"    Principle: {argument.principle}")
        lines.append(f"    Supporting Context: {argument.supporting_context}")
        if getattr(argument, "direction", "").strip():
            lines.append(f"    Direction: {argument.direction}")
    return "\n".join(lines)


def _tokenize_text(text: str, tokenizer: Any | None) -> list:
    if not text:
        return []
    if tokenizer is None:
        return text.split()
    tokenizer_lock = getattr(tokenizer, "_sieve_tokenizer_lock", None)
    try:
        if tokenizer_lock is not None:
            with tokenizer_lock:
                return tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        try:
            if tokenizer_lock is not None:
                with tokenizer_lock:
                    return tokenizer.encode(text)
            return tokenizer.encode(text)
        except Exception:
            return text.split()
    except Exception:
        return text.split()


def _decode_tokens(tokens: list, tokenizer: Any | None) -> str:
    if not tokens:
        return ""
    if not all(isinstance(token, int) for token in tokens):
        return " ".join(str(token) for token in tokens)
    if tokenizer is None:
        return " ".join(str(token) for token in tokens)
    tokenizer_lock = getattr(tokenizer, "_sieve_tokenizer_lock", None)
    try:
        if tokenizer_lock is not None:
            with tokenizer_lock:
                return tokenizer.decode(tokens, skip_special_tokens=True)
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except TypeError:
        try:
            if tokenizer_lock is not None:
                with tokenizer_lock:
                    return tokenizer.decode(tokens)
            return tokenizer.decode(tokens)
        except Exception:
            return " ".join(str(token) for token in tokens)
    except Exception:
        return " ".join(str(token) for token in tokens)


def _truncate_to_token_budget(text: str, budget: int, tokenizer: Any | None) -> str:
    if budget <= 0 or not text.strip():
        return NO_ARGUMENTS_AVAILABLE
    tokens = _tokenize_text(text, tokenizer)
    if len(tokens) <= budget:
        return text
    truncated = _decode_tokens(tokens[:budget], tokenizer).strip()
    return truncated if truncated else NO_ARGUMENTS_AVAILABLE


def token_proportional_truncation_stats(
    phase3_selection: Phase3Selection,
    *,
    tokenizer: Any | None,
    use_token_total_budget: bool = False,
    use_all: bool = False,
    use_top: bool = False,
    use_bottom: bool = False,
) -> dict[str, dict[str, Any]]:
    schema_arguments = {
        phase3_selection.primary_schema: _format_argument_block(phase3_selection.primary_arguments),
        phase3_selection.secondary_schema: _format_argument_block(phase3_selection.secondary_arguments),
        phase3_selection.tertiary_schema: _format_argument_block(phase3_selection.tertiary_principles),
    }
    ordered_schemas = [
        phase3_selection.primary_schema,
        phase3_selection.secondary_schema,
        phase3_selection.tertiary_schema,
    ]
    tokenized = {
        schema: _tokenize_text(text, tokenizer)
        for schema, text in schema_arguments.items()
    }
    total_argument_tokens = sum(len(tokenized.get(schema, [])) for schema in ordered_schemas)
    theta_by_schema = {
        schema: float(theta)
        for schema, theta in zip(SCHEMA_NAMES, phase3_selection.theta)
    }
    stats: dict[str, dict[str, Any]] = {}
    top_schema = phase3_selection.primary_schema
    bottom_schema = min(
        ordered_schemas,
        key=lambda schema: theta_by_schema.get(schema, 0.0),
    )
    for rank, schema in enumerate(ordered_schemas, start=1):
        original_tokens = len(tokenized.get(schema, []))
        if use_all or (use_top and schema == top_schema) or (use_bottom and schema == bottom_schema):
            budget = original_tokens
        elif use_top or use_bottom:
            budget = 0
        elif use_token_total_budget:
            budget = int(total_argument_tokens * theta_by_schema.get(schema, 0.0))
        else:
            budget = int(original_tokens * theta_by_schema.get(schema, 0.0))
        kept_tokens = original_tokens if use_all else min(original_tokens, max(0, budget))
        stats[schema] = {
            "rank": rank,
            "theta": theta_by_schema.get(schema, 0.0),
            "original_tokens": original_tokens,
            "budget": budget,
            "kept_tokens": kept_tokens,
            "truncated_tokens": max(0, original_tokens - kept_tokens),
            "kept_ratio": (
                float(kept_tokens) / float(original_tokens)
                if original_tokens > 0
                else None
            ),
        }
    return stats


def _assemble_token_proportional_prompt(
    scenario: str,
    phase3_selection: Phase3Selection,
    *,
    use_persona: bool,
    tokenizer: Any | None,
    use_token_total_budget: bool = False,
    use_all: bool = False,
    use_top: bool = False,
    use_bottom: bool = False,
    safety: bool = False,
) -> str:
    persona = (
        SCHEMA_PERSONAS[phase3_selection.primary_schema]
        if use_persona
        else "You are reasoning about a moral scenario."
    )
    schema_arguments = {
        phase3_selection.primary_schema: _format_argument_block(phase3_selection.primary_arguments),
        phase3_selection.secondary_schema: _format_argument_block(phase3_selection.secondary_arguments),
        phase3_selection.tertiary_schema: _format_argument_block(phase3_selection.tertiary_principles),
    }
    ordered_schemas = [
        phase3_selection.primary_schema,
        phase3_selection.secondary_schema,
        phase3_selection.tertiary_schema,
    ]
    tokenized = {
        schema: _tokenize_text(text, tokenizer)
        for schema, text in schema_arguments.items()
    }
    total_argument_tokens = sum(len(tokenized.get(schema, [])) for schema in ordered_schemas)
    theta_by_schema = {
        schema: float(theta)
        for schema, theta in zip(SCHEMA_NAMES, phase3_selection.theta)
    }
    truncated_blocks = {}
    top_schema = phase3_selection.primary_schema
    bottom_schema = min(
        ordered_schemas,
        key=lambda schema: theta_by_schema.get(schema, 0.0),
    )
    for schema in ordered_schemas:
        if use_all:
            truncated_blocks[schema] = schema_arguments.get(schema, "")
            continue
        if use_top:
            truncated_blocks[schema] = (
                schema_arguments.get(schema, "")
                if schema == top_schema
                else NO_ARGUMENTS_AVAILABLE
            )
            continue
        if use_bottom:
            truncated_blocks[schema] = (
                schema_arguments.get(schema, "")
                if schema == bottom_schema
                else NO_ARGUMENTS_AVAILABLE
            )
            continue
        if use_token_total_budget:
            budget = int(total_argument_tokens * theta_by_schema.get(schema, 0.0))
        else:
            schema_token_count = len(tokenized.get(schema, []))
            budget = int(schema_token_count * theta_by_schema.get(schema, 0.0))
        truncated_blocks[schema] = _truncate_to_token_budget(
            schema_arguments.get(schema, ""),
            budget,
            tokenizer,
        )

    reasoner_prompt = SAFETY_REASONER_PROMPT if safety else TOKEN_PROPORTIONAL_REASONER_PROMPT
    instruction_block = (
        SAFETY_INSTRUCTION_BLOCK if safety else TOKEN_PROPORTIONAL_INSTRUCTION_BLOCK
    )
    return reasoner_prompt.format(
        persona=persona,
        scenario=scenario,
        # primary_schema_name=SCHEMA_FULL_NAMES[phase3_selection.primary_schema],
        # secondary_schema_name=SCHEMA_FULL_NAMES[phase3_selection.secondary_schema],
        # tertiary_schema_name=SCHEMA_FULL_NAMES[phase3_selection.tertiary_schema],
        primary_arguments=truncated_blocks[phase3_selection.primary_schema],
        secondary_arguments=truncated_blocks[phase3_selection.secondary_schema],
        tertiary_arguments=truncated_blocks[phase3_selection.tertiary_schema],
        instruction_block=instruction_block,
    )


def _format_tertiary_block(arguments) -> str:
    if not arguments:
        return NO_ARGUMENTS_AVAILABLE
    lines = []
    for idx, argument in enumerate(arguments, start=1):
        if (
            getattr(argument, "principle", "").strip()
            and not getattr(argument, "supporting_context", "").strip()
            and not getattr(argument, "direction", "").strip()
        ):
            lines.append(argument.principle)
            continue
        lines.append(f"  Principle {idx}: {argument.principle}")
    return "\n".join(lines)


def _get_instruction_regime(theta: list[float], threshold: float = INSTRUCTION_REGIME_THRESHOLD) -> str:
    count_above = sum(1 for value in theta if value > threshold)
    if count_above >= 3:
        return "uncertain"
    if count_above == 2:
        return "competing"
    return "confident"


def assemble_prompt(
    scenario: str,
    phase3_selection: Phase3Selection,
    use_persona: bool = False,
    inst_regime: bool = False,
    inference_add_eval: bool = True,
    token_proportional: bool = False,
    use_token_total_budget: bool = False,
    tokenizer: Any | None = None,
    use_all: bool = False,
    use_top: bool = False,
    use_bottom: bool = False,
    safety: bool = False,
) -> str:
    """Build the Phase 4 hierarchical reasoning prompt."""
    if token_proportional:
        return _assemble_token_proportional_prompt(
            scenario,
            phase3_selection,
            use_persona=use_persona,
            tokenizer=tokenizer,
            use_token_total_budget=use_token_total_budget,
            use_all=use_all,
            use_top=use_top,
            use_bottom=use_bottom,
            safety=safety,
        )

    persona = (
        SCHEMA_PERSONAS[phase3_selection.primary_schema]
        if use_persona
        else "You are reasoning about a moral scenario."
    )
    theta_values = [float(value) for value in phase3_selection.theta]
    instruction_templates = (
        INSTRUCTION_TEMPLATES_ADD_EVAL if inference_add_eval else INSTRUCTION_TEMPLATES_OLD
    )
    reasoner_prompt = REASONER_PROMPT_ADD_EVAL if inference_add_eval else REASONER_PROMPT_OLD
    if inst_regime:
        instruction_block = instruction_templates[_get_instruction_regime(theta_values)]
        tertiary_block = _format_argument_block(phase3_selection.tertiary_principles)
    else:
        instruction_block = LEGACY_INSTRUCTION_BLOCK
        tertiary_block = _format_tertiary_block(phase3_selection.tertiary_principles)
    prompt = reasoner_prompt.format(
        persona=persona,
        scenario=scenario,
        primary_schema_name=SCHEMA_FULL_NAMES[phase3_selection.primary_schema],
        primary_arguments=_format_argument_block(phase3_selection.primary_arguments),
        secondary_schema_name=SCHEMA_FULL_NAMES[phase3_selection.secondary_schema],
        secondary_arguments=_format_argument_block(phase3_selection.secondary_arguments),
        tertiary_schema_name=SCHEMA_FULL_NAMES[phase3_selection.tertiary_schema],
        tertiary_arguments=tertiary_block,
        instruction_block=instruction_block,
    )
    return prompt


def parse_answer(response: str) -> str:
    """Extract yes/no answer from reasoner response."""
    resp = response.replace("**", "").replace("##", "")
    return re.split(r"answer\s*:", resp, flags=re.IGNORECASE)[-1].strip()
    
    # if "answer: yes" in lower:
    #     return "yes"
    # if "answer: no" in lower:
    #     return "no"
    # # Fallback: last yes/no in text
    # for word in reversed(lower.split()):
    #     if word.strip(".,!?") in ("yes", "no"):
    #         return word.strip(".,!?")
    # return "unknown"


def parse_reasoning(response: str) -> str:
    """Extract reasoning text from reasoner response."""
    if "Reasoning:" in response:
        parts = response.split("Reasoning:", 1)
        if len(parts) > 1:
            reasoning = parts[1]
            if "Answer:" in reasoning:
                reasoning = reasoning.split("Answer:")[0]
            return reasoning.strip()
    return response.strip()
