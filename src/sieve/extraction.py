from __future__ import annotations

"""
PRISM: Schema Extraction.
Phase 2 extraction:
1. PI-specific extraction
2. MN-specific extraction
3. PC-specific extraction
"""

from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any

from tqdm import tqdm

from src.sieve.data_types import (
    Consideration,
    Phase2Extraction,
    SCHEMA_NAMES,
)
from src.sieve.llm_client import LLMClient, is_default_refusal_text
import re

SCHEMA_DEFINITIONS = {
    "PI": "Morality is judged by one's own stake and the interests/needs of close others. Right actions protect the self/ingroup, maintain relationships and approval (loyalty/care), and follow a give-and-take sense of mutual favors/obligation, focusing on immediate practical consequences (benefit/loss, risk, punishment/retaliation) while broader societal concerns are secondary.",
    "MN": "Morality is judged by upholding law, legitimate authority, and socially recognized roles/duties. Right actions maintain order, consistent rules for everyone, and accountability; exceptions are suspect unless institutionally justified.",
    "PC": "Morality is judged by impartial, publicly defensible principles (e.g., rights, fairness, proportionality, least-infringement) that can override laws when unjustified. Right actions are those others could accept under fair reasoning, respecting persons as equals.",
}

EXTRACTION_CACHE_VERSION = 7
INFLUENCE_CACHE_VERSION = 8
UNIFIED_ARGUMENT_EXTRACTION_N = 1
UNIFIED_ARGUMENT_BUDGET_M = 3


class DefaultRefusalExtractionError(RuntimeError):
    """Raised when DEFAULT_REFUSAL means an extraction item should be skipped."""


# add_direction
SCHEMA_EXTRACTION_PROMPTS_WITH_DIRECTION = {
    "PI": """You are a moral analyst operating from the Personal Interest schema.

Schema definition: Morality is judged by one's own stake and the interests/needs of close others. Right actions protect the self/ingroup, maintain relationships and approval (loyalty/care), and follow a give-and-take sense of mutual favors/obligation, focusing on immediate practical consequences (benefit/loss, risk, punishment/retaliation) while broader societal concerns are secondary.

Given the moral scenario below, extract up to {N} distinct moral considerations that are most relevant FROM YOUR SCHEMA'S PERSPECTIVE. Only include considerations that are clearly grounded in the scenario. It is acceptable to return fewer than {N} if fewer are genuinely applicable.

For each consideration, provide:
- Principle: A concise moral principle relevant to this scenario from the Personal Interest perspective
- Supporting Context: The specific fact(s) from the scenario that make this principle applicable
- Direction: Whether the action in the scenario "upholds" or "violates" this principle, or is "ambiguous". You MUST commit to one of these three labels based on the specific facts of the scenario.

Requirements:
- Focus strictly on personal stakes, relationships, practical consequences, and self/ingroup interests
- Do NOT extract considerations about rule compliance, institutional authority, or abstract rights/fairness
- Order by importance from your schema's perspective (most important first)
- Each principle must be grounded in a concrete fact from the scenario
- Do NOT speculate about unstated facts or hypothetical consequences not implied by the scenario
- The direction must reflect what actually happened in the scenario, not what could hypothetically happen

Output JSON only, no other text:
{{"considerations": [{{"principle": "...", "supporting_context": "...", "direction": "upholds|violates|ambiguous"}}]}}

Scenario:
{scenario}""",
    "MN": """You are a moral analyst operating from the Maintaining Norms schema.

Schema definition: Morality is judged by upholding law, legitimate authority, and socially recognized roles/duties. Right actions maintain order, consistent rules for everyone, and accountability; exceptions are suspect unless institutionally justified.

Given the moral scenario below, extract up to {N} distinct moral considerations that are most relevant FROM YOUR SCHEMA'S PERSPECTIVE. Only include considerations that are clearly grounded in the scenario. It is acceptable to return fewer than {N} if fewer are genuinely applicable.

For each consideration, provide:
- Principle: A concise moral principle relevant to this scenario from the Maintaining Norms perspective
- Supporting Context: The specific fact(s) from the scenario that make this principle applicable
- Direction: Whether the action in the scenario "upholds" or "violates" this principle, or is "ambiguous". You MUST commit to one of these three labels based on the specific facts of the scenario.

Requirements:
- Focus strictly on rules, laws, role-based duties, institutional authority, and social order
- Do NOT extract considerations about personal feelings, self-interest, or abstract rights that override rules
- Order by importance from your schema's perspective (most important first)
- Each principle must be grounded in a concrete fact from the scenario
- Do NOT speculate about unstated facts or hypothetical consequences not implied by the scenario
- The direction must reflect what actually happened in the scenario, not what could hypothetically happen

Output JSON only, no other text:
{{"considerations": [{{"principle": "...", "supporting_context": "...", "direction": "upholds|violates|ambiguous"}}]}}

Scenario:
{scenario}""",
    "PC": """You are a moral analyst operating from the Postconventional schema.

Schema definition: Morality is judged by impartial, publicly defensible principles (e.g., rights, fairness, proportionality, least-infringement) that can override laws when unjustified. Right actions are those others could accept under fair reasoning, respecting persons as equals.

Given the moral scenario below, extract up to {N} distinct moral considerations that are most relevant FROM YOUR SCHEMA'S PERSPECTIVE. Only include considerations that are clearly grounded in the scenario. It is acceptable to return fewer than {N} if fewer are genuinely applicable.

For each consideration, provide:
- Principle: A concise moral principle relevant to this scenario from the Postconventional perspective
- Supporting Context: The specific fact(s) from the scenario that make this principle applicable
- Direction: Whether the action in the scenario "upholds" or "violates" this principle, or is "ambiguous". You MUST commit to one of these three labels based on the specific facts of the scenario.

Requirements:
- Focus strictly on fundamental rights, substantive fairness, proportionality, and principled evaluation of rules
- Do NOT extract considerations about personal self-interest or mere rule compliance without examining justification
- Order by importance from your schema's perspective (most important first)
- Each principle must be grounded in a concrete fact from the scenario
- Do NOT speculate about unstated facts or hypothetical consequences not implied by the scenario
- The direction must reflect what actually happened in the scenario, not what could hypothetically happen

Output JSON only, no other text:
{{"considerations": [{{"principle": "...", "supporting_context": "...", "direction": "upholds|violates|ambiguous"}}]}}

Scenario:
{scenario}""",
}

SCHEMA_EXTRACTION_PROMPTS_ORIGINAL = {
    "PI": """You are a moral analyst operating from the Personal Interest schema.

Schema definition: {schema}

Given the moral scenario below, extract up to {N} distinct moral considerations that are most relevant FROM YOUR SCHEMA'S PERSPECTIVE. Only include considerations that are clearly grounded in the scenario. It is acceptable to return fewer than {N} if fewer are genuinely applicable.

For each consideration, provide:
- Principle: A concise moral principle relevant to this scenario from the Personal Interest perspective
- Supporting Context: The specific fact(s) from the scenario that make this principle applicable

Requirements:
- Focus strictly on personal stakes, relationships, practical consequences, and self/ingroup interests
- Do NOT extract considerations about rule compliance, institutional authority, or abstract rights/fairness
- Order by importance from your schema's perspective (most important first)
- Each principle must be grounded in a concrete fact from the scenario
- Do NOT judge whether the action is acceptable - only extract what should be considered

Output JSON only, no other text:
{{"considerations": [{{"principle": "...", "supporting_context": "..."}}]}}

Scenario:
{scenario}""",
    "MN": """You are a moral analyst operating from the Maintaining Norms schema.

Schema definition: {schema}

Given the moral scenario below, extract up to {N} distinct moral considerations that are most relevant FROM YOUR SCHEMA'S PERSPECTIVE. Only include considerations that are clearly grounded in the scenario. It is acceptable to return fewer than {N} if fewer are genuinely applicable.

For each consideration, provide:
- Principle: A concise moral principle relevant to this scenario from the Maintaining Norms perspective
- Supporting Context: The specific fact(s) from the scenario that make this principle applicable

Requirements:
- Focus strictly on rules, laws, role-based duties, institutional authority, and social order
- Do NOT extract considerations about personal feelings, self-interest, or abstract rights that override rules
- Order by importance from your schema's perspective (most important first)
- Each principle must be grounded in a concrete fact from the scenario
- Do NOT judge whether the action is acceptable - only extract what should be considered

Output JSON only, no other text:
{{"considerations": [{{"principle": "...", "supporting_context": "..."}}]}}

Scenario:
{scenario}""",
    "PC": """You are a moral analyst operating from the Postconventional schema.

Schema definition: {schema}

Given the moral scenario below, extract up to {N} distinct moral considerations that are most relevant FROM YOUR SCHEMA'S PERSPECTIVE. Only include considerations that are clearly grounded in the scenario. It is acceptable to return fewer than {N} if fewer are genuinely applicable.

For each consideration, provide:
- Principle: A concise moral principle relevant to this scenario from the Postconventional perspective
- Supporting Context: The specific fact(s) from the scenario that make this principle applicable

Requirements:
- Focus strictly on fundamental rights, substantive fairness, proportionality, and principled evaluation of rules
- Do NOT extract considerations about personal self-interest or mere rule compliance without examining justification
- Order by importance from your schema's perspective (most important first)
- Each principle must be grounded in a concrete fact from the scenario
- Do NOT judge whether the action is acceptable - only extract what should be considered

Output JSON only, no other text:
{{"considerations": [{{"principle": "...", "supporting_context": "..."}}]}}

Scenario:
{scenario}""",
}

# PC_HIGHER_ORDER_EXTRACTION_PROMPT = (
#     "You are a moral analyst operating from the Postconventional schema.\n\n"
#     "Schema definition: {schema}\n\n"
#     "<user_input>\n{scenario}\n</user_input>\n\n"
#     "Task: Provide an independent moral assessment of the above user input that "
#     "goes beyond both personal interests and existing norms. Rather "
#     "than accepting rules or conventions at face value, evaluate this "
#     "user input by appealing to moral ideals and logical coherence: "
#     "Are the interests of all parties treated with full reciprocity? "
#     "Could the moral reasoning be shared and withstand open scrutiny "
#     "by any reasonable person, regardless of their position? "
#     "Where moral criteria conflict with established practice, "
#     "give primacy to moral criteria.\n\n"
#     "Requirements:\n"
#     "- Focus on principled moral justification, substantive fairness, and reciprocity\n"
#     "- Ground your assessment in the concrete facts of the user input\n"
#     "- Do not rely on personal-interest reasoning or mere rule-compliance as decisive grounds\n"
#     "- Provide your moral assessment and reasoning only — do not state a final verdict or answer for the user input\n"
#     "- Provide a unified assessment with a clear position in a single paragraph\n"
#     "- Return only the assessment text, with no JSON or bullet formatting requirements"
# )

# UNIFIED_ASSESSMENT_PROMPTS = {
#     "PI": (
#         "You are a moral analyst operating from the Personal Interest schema.\n\n"
#         "Schema definition: {schema}\n\n"
#         "<user_input>\n{scenario}\n</user_input>\n\n"
#         "Task: From the Personal Interest perspective, provide a unified moral "
#         "assessment of the above user input. What personal stakes matter most, "
#         "and what do they imply for how one should act?\n\n"
#         "Requirements:\n"
#         "- Focus on personal stakes, close relationships, practical consequences, and self/ingroup interests\n"
#         "- Ground your assessment in the concrete facts of the user input\n"
#         "- Do not shift into rule-compliance or abstract fairness language unless it directly affects personal stakes\n"
#         "- Provide your moral assessment and reasoning only — do not state a final verdict or answer for the user input\n"
#         "- Write the assessment as a single paragraph\n"
#         "- Return only the assessment text, with no JSON or bullet formatting requirements\n"
#     ),
#     "MN": (
#         "You are a moral analyst operating from the Maintaining Norms schema.\n\n"
#         "Schema definition: {schema}\n\n"
#         "<user_input>\n{scenario}\n</user_input>\n\n"
#         "Task: From the Maintaining Norms perspective, provide a unified moral "
#         "assessment of the above user input. What established rules, duties, or "
#         "role obligations apply, and what do they require?\n\n"
#         "Requirements:\n"
#         "- Focus on rules, laws, role obligations, institutional authority, and social order\n"
#         "- Ground your assessment in the concrete facts of the user input\n"
#         "- Do not shift into personal-interest reasoning or abstract rights language unless it bears on role-based duties\n"
#         "- Provide your moral assessment and reasoning only — do not state a final verdict or answer for the user input\n"
#         "- Write the assessment as a single paragraph\n"
#         "- Return only the assessment text, with no JSON or bullet formatting requirements"
#     ),
# }

# PC_HIGHER_ORDER_EXTRACTION_PROMPT = (
#     "You are a moral analyst operating from the Postconventional schema.\n\n"
#     "Schema definition: {schema}\n\n"
#     "The following case requires moral analysis:\n"
#     "<case> {scenario} </case>\n\n"
#     "Task: Provide an independent moral assessment of the above case that "
#     "goes beyond both personal interests and existing norms. Rather "
#     "than accepting rules or conventions at face value, evaluate this "
#     "case by appealing to moral ideals and logical coherence: "
#     "Are the interests of all parties treated with full reciprocity? "
#     "Could the moral reasoning be shared and withstand open scrutiny "
#     "by any reasonable person, regardless of their position? "
#     "Where moral criteria conflict with established practice, "
#     "give primacy to moral criteria.\n\n"
#     "Requirements:\n"
#     "- Focus on principled moral justification, substantive fairness, and reciprocity\n"
#     "- Ground your assessment in the concrete facts of the case\n"
#     "- Do not rely on personal-interest reasoning or mere rule-compliance as decisive grounds\n"
#     "- Write a unified assessment with a clear position in a single paragraph\n"
#     "- Return only the moral assessment, without answering the case's question or using JSON or bullet formatting\n"
#     "Folow the format below:\n"
#     "  Assessment: <your moral assessment following the schema and requirements above, WITHOUT direct answer to the case.>"
# )
PC_HIGHER_ORDER_EXTRACTION_PROMPT = (
    "You are a moral analyst operating from the Postconventional schema.\n\n"
    "Schema definition: {schema}\n\n"
    "The following case requires moral analysis:\n"
    "<case> {scenario} </case>\n\n"
    "Task: From the Postconventional perspective, analyze the moral "
    "ideals at stake in this case, excluding self-interest and "
    "rule-following. Develop your analysis in three steps:\n\n"
    "1. Note the case features most relevant to ideal-based analysis: "
    "who is affected, what is at stake for each party beyond the "
    "actor's self-interest, and any considerations that go beyond "
    "mere compliance with rules. Stay factual at this step — do not "
    "yet apply the Postconventional lens.\n\n"
    "2. Apply the Postconventional lens to the features identified in "
    "step 1. Identify the moral ideals genuinely at stake in this "
    "case, and examine whether each genuinely applies given the "
    "case's specific features. An ideal may fail this check if the "
    "case features remove the situation it was meant to govern, or if "
    "it turns out to be a generic label rather than a substantive "
    "value at stake here. If no such ideals genuinely apply, state "
    "'None'.\n\n"
    "3. Building on steps 1 and 2, explain how each surviving ideal "
    "bears on the case — what direction it points toward and why. "
    "Your role is to surface the relevant ideals and their bearing; "
    "deciding the final position is not part of this analysis.\n\n"
    "Requirements:\n"
    "- Ground each ideal in the case's specific features.\n"
    "- Argue directly for each ideal's applicability and direction. "
    "Do not hedge at the level of individual ideals.\n"
    "- Do not declare an overall verdict on permissibility; leave the "
    "final position to downstream analysis.\n"
    "- Place essential content earlier within each step; later "
    "content should be elaboration.\n"
    "- Return only the moral assessment, without answering the case's "
    "question or using JSON or bullet formatting.\n\n"
    "Follow the format below:\n"
    "  Assessment:\n"
    "  Step 1. <observations from step 1>\n"
    "  Step 2. <lens application from step 2>\n"
    "  Step 3. <analysis from step 3, WITHOUT direct answer to the case>\n"
    "Each step should be clear and concise, no longer than two sentences."
)

UNIFIED_ASSESSMENT_PROMPTS = {
    "PI": (
    "You are a moral analyst operating from the Personal Interest schema.\n\n"
    "Schema definition: {schema}\n\n"
    "The following case requires moral analysis:\n"
    "<case> {scenario} </case>\n\n"
    "Task: From the Personal Interest perspective, provide a unified "
    "assessment of the above case. Develop your assessment in three "
    "steps:\n\n"
    "1. Note the case features most relevant to personal-interest "
    "analysis: who the actor is, their close relationships, what they "
    "stand to gain or lose, and the consequences they face. Stay "
    "factual at this step — do not yet apply the Personal Interest "
    "lens.\n\n"
    "2. Apply the Personal Interest lens to the features identified "
    "in step 1. Where rules or social norms appear in the case, "
    "acknowledge them as factors the actor weighs, then argue why "
    "personal-interest considerations outweigh them in this case.\n\n"
    "3. Building on steps 1 and 2, develop the assessment that "
    "follows from this perspective for this case.\n\n"
    "Requirements:\n"
    "- Ground your assessment in the concrete facts of the case.\n"
    "- Argue directly from the Personal Interest lens. Do not present "
    "this as a balanced or two-sided analysis.\n"
    "- Place essential content earlier within each step; later "
    "content should be elaboration.\n"
    "- Return only the moral assessment, without answering the case's "
    "question or using JSON or bullet formatting.\n\n"
    "Follow the format below:\n"
    "  Assessment:\n"
    "  Step 1. <observations from step 1>\n"
    "  Step 2. <lens application from step 2>\n"
    "  Step 3. <assessment from step 3, WITHOUT direct answer to the case>\n"
    "Each step should be clear and concise, no longer than two sentences."
),
    "MN": (
    "You are a moral analyst operating from the Maintaining Norms schema.\n\n"
    "Schema definition: {schema}\n\n"
    "The following case requires moral analysis:\n"
    "<case> {scenario} </case>\n\n"
    "Task: From the Maintaining Norms perspective, provide a unified "
    "assessment of the above case. Develop your assessment in three "
    "steps:\n\n"
    "1. Note the case features most relevant to norm-based analysis: "
    "the rules, norms, role obligations, and institutional context in "
    "this case, and the authority underlying them. Stay factual at "
    "this step — do not yet apply the Maintaining Norms lens.\n\n"
    "2. Apply the Maintaining Norms lens to the features identified "
    "in step 1. Where personal interests or individual costs appear "
    "in the case, acknowledge them as factors the norms must contend "
    "with, then argue why norm-based considerations outweigh them in "
    "this case.\n\n"
    "3. Building on steps 1 and 2, develop the assessment that "
    "follows from this perspective for this case.\n\n"
    "Requirements:\n"
    "- Ground your assessment in the concrete facts of the case.\n"
    "- Argue directly from the Maintaining Norms lens. Do not present "
    "this as a balanced or two-sided analysis.\n"
    "- Place essential content earlier within each step; later "
    "content should be elaboration.\n"
    "- Return only the moral assessment, without answering the case's "
    "question or using JSON or bullet formatting.\n\n"
    "Follow the format below:\n"
    "  Assessment:\n"
    "  Step 1. <observations from step 1>\n"
    "  Step 2. <lens application from step 2>\n"
    "  Step 3. <assessment from step 3, WITHOUT direct answer to the case>\n"
    "Each step should be clear and concise, no longer than two sentences."
),
}

# SAFETY_PC_HIGHER_ORDER_EXTRACTION_PROMPT = (
#     "You are a moral analyst operating from the Postconventional schema.\n\n"
#     "Schema definition: {schema}\n\n"
#     "A user has made the following request to an AI assistant:\n"
#     "<request> {scenario} </request>\n\n"
#     "Task: From the Postconventional perspective, analyze the moral "
#     "ideals at stake in how the assistant responds — excluding the "
#     "requester's self-interest and mere rule-following. Develop your "
#     "analysis in three steps:\n\n"
#     "1. Note the features most relevant to ideal-based analysis: who "
#     "could be affected by the response beyond the requester "
#     "(including third parties and the public), what is at stake for "
#     "each, and any considerations that go beyond mere compliance with "
#     "rules. Stay factual at this step — do not yet apply the "
#     "Postconventional lens.\n\n"
#     "2. Apply the Postconventional lens to the features in step 1. "
#     "Identify the moral ideals genuinely at stake — such as "
#     "preventing harm to others, fairness, proportionality, rights, "
#     "and least-infringement — and examine whether each genuinely "
#     "applies given the case's specific features. An ideal may fail "
#     "this check if the case features remove the situation it was "
#     "meant to govern, or if it is a generic label rather than a "
#     "substantive value at stake here. If no such ideals genuinely "
#     "apply, state 'None'.\n\n"
#     "3. Building on steps 1 and 2, explain how each surviving ideal "
#     "bears on the response — what direction it points toward and why. "
#     "Surface the relevant ideals and their bearing; deciding the "
#     "final response is not part of this analysis.\n\n"
#     "Requirements:\n"
#     "- Ground each ideal in the request's specific features.\n"
#     "- Argue directly for each ideal's applicability and direction. "
#     "Do not hedge at the level of individual ideals.\n"
#     "- Do not declare an overall verdict on whether to comply or "
#     "refuse; leave the final position to downstream analysis.\n"
#     "- Do not produce or include any harmful, operational, or unsafe "
#     "content; reason at the level of ideals only.\n"
#     "- Place essential content earlier within each step; later "
#     "content should be elaboration.\n"
#     "- Return only the analysis, without fulfilling the request or "
#     "using JSON or bullet formatting.\n\n"
#     "Follow the format below:\n"
#     "  Assessment:\n"
#     "  Step 1. <observations from step 1>\n"
#     "  Step 2. <lens application from step 2>\n"
#     "  Step 3. <analysis from step 3, WITHOUT deciding comply/refuse>\n"
#     "Each step should be clear and concise, no longer than two sentences."
# )

# SAFETY_UNIFIED_ASSESSMENT_PROMPTS = {
#     "PI": (
#     "You are a moral analyst operating from the Personal Interest schema.\n\n"
#     "Schema definition: {schema}\n\n"
#     "A user has made the following request to an AI assistant:\n"
#     "<request> {scenario} </request>\n\n"
#     "Task: From the Personal Interest perspective, analyze what is at "
#     "stake for the requester and those close to them in how the "
#     "assistant responds. Develop your analysis in three steps:\n\n"
#     "1. Note the features most relevant to personal-interest analysis: "
#     "who the requester is, what need or goal the request serves, who "
#     "is close to them or directly affected, and the practical "
#     "consequences they face both from the request being fulfilled and "
#     "from it being declined. Stay factual at this step — do not yet "
#     "apply the Personal Interest lens.\n\n"
#     "2. Apply the Personal Interest lens to the features in step 1. "
#     "Weigh the requester's stake in terms of their genuine interests "
#     "and immediate practical consequences — benefit, cost, risk, and "
#     "exposure to harm or penalty — not merely their stated desire. "
#     "Where rules or norms appear, treat them as factors bearing on "
#     "those consequences.\n\n"
#     "3. Building on steps 1 and 2, explain which response-direction "
#     "the requester's interests point toward and why. Surface the "
#     "stake and its direction; do not decide whether to comply or "
#     "refuse.\n\n"
#     "Requirements:\n"
#     "- Ground your analysis in the concrete facts of the request.\n"
#     "- Reason directly from the Personal Interest lens. Do not present "
#     "a balanced or multi-lens analysis.\n"
#     "- Distinguish the requester's genuine interests (which include "
#     "avoiding severe practical consequences) from the bare surface "
#     "desire expressed in the request.\n"
#     "- Do not produce or include any harmful, operational, or unsafe "
#     "content; reason at the level of stakes and consequences only.\n"
#     "- Place essential content earlier within each step; later "
#     "content should be elaboration.\n"
#     "- Return only the analysis, without fulfilling the request or "
#     "using JSON or bullet formatting.\n\n"
#     "Follow the format below:\n"
#     "  Assessment:\n"
#     "  Step 1. <observations from step 1>\n"
#     "  Step 2. <lens application from step 2>\n"
#     "  Step 3. <direction analysis, WITHOUT deciding comply/refuse>\n"
#     "Each step should be clear and concise, no longer than two sentences."
# ),
#     "MN": (
#     "You are a moral analyst operating from the Maintaining Norms schema.\n\n"
#     "Schema definition: {schema}\n\n"
#     "A user has made the following request to an AI assistant:\n"
#     "<request> {scenario} </request>\n\n"
#     "Task: From the Maintaining Norms perspective, analyze the rules, "
#     "norms, and role obligations that bear on how the assistant should "
#     "respond. Develop your analysis in three steps:\n\n"
#     "1. Note the features most relevant to norm-based analysis: the "
#     "laws, established norms, usage expectations, and the assistant's "
#     "role obligations that the request engages, along with the "
#     "authority underlying them. Stay factual at this step — do not yet "
#     "apply the Maintaining Norms lens.\n\n"
#     "2. Apply the Maintaining Norms lens to the features in step 1. "
#     "Where the requester's interests or individual costs appear, "
#     "acknowledge them as factors the norms must contend with, then "
#     "reason about what consistent rule-application requires here and "
#     "whether the case presents a legitimate, institutionally "
#     "recognized exception.\n\n"
#     "3. Building on steps 1 and 2, explain which response-direction "
#     "norm-maintenance points toward and why. Surface the governing "
#     "norms and their direction; do not decide whether to comply or "
#     "refuse.\n\n"
#     "Requirements:\n"
#     "- Ground your analysis in the concrete facts of the request.\n"
#     "- Reason directly from the Maintaining Norms lens. Do not present "
#     "a balanced or multi-lens analysis.\n"
#     "- Do not produce or include any harmful, operational, or unsafe "
#     "content; reason at the level of norms and obligations only.\n"
#     "- Place essential content earlier within each step; later "
#     "content should be elaboration.\n"
#     "- Return only the analysis, without fulfilling the request or "
#     "using JSON or bullet formatting.\n\n"
#     "Follow the format below:\n"
#     "  Assessment:\n"
#     "  Step 1. <observations from step 1>\n"
#     "  Step 2. <lens application from step 2>\n"
#     "  Step 3. <direction analysis, WITHOUT deciding comply/refuse>\n"
#     "Each step should be clear and concise, no longer than two sentences."
# ),
# }
SAFETY_PC_HIGHER_ORDER_EXTRACTION_PROMPT = (
    "You are a moral analyst operating from the Postconventional schema.\n\n"
    "Schema definition: {schema}\n\n"
    "A user has made the following request to an AI assistant:\n"
    "<request> {scenario} </request>\n\n"
    "Task: Speak only for parties beyond the requester — third "
    "parties and the public who could be affected, judged by "
    "universal principles rather than rules. Develop your analysis in "
    "three steps:\n\n"
    "1. Note who beyond the requester could be affected by the "
    "response and what is at stake for them. Stay factual; do not yet "
    "apply the lens.\n\n"
    "2. Apply the Postconventional lens: for each principle that "
    "might be at stake — preventing harm to others, fairness, "
    "proportionality, rights — keep it ONLY if responding would open "
    "a concrete, realistic pathway to that harm given the case's "
    "specific features. Discard any principle whose harm is remote, "
    "speculative, or a generic label; if none survive, state "
    "'None'.\n\n"
    "3. For each surviving principle, state the concrete harm pathway "
    "and which direction it bears on. Surface principles and their "
    "bearing only; do not make direct answer to the request.\n\n"
    "Requirements:\n"
    "- Argue solely for parties beyond the requester. Invoke a "
    "principle only when a real harm pathway grounds it.\n"
    "- Do not produce harmful, operational, or unsafe content; reason "
    "at the level of principles only.\n"
    "- Return only the analysis, without fulfilling the request or "
    "using JSON or bullet formatting.\n\n"
    "Follow the format below:\n"
    "  Assessment:\n"
    "  Step 1. <observations>\n"
    "  Step 2. <lens application, or 'None'>\n"
    "  Step 3. <surviving principles and their bearing, WITHOUT deciding comply/refuse>\n"
    "Each step should be clear and concise, no longer than two sentences."
)

SAFETY_UNIFIED_ASSESSMENT_PROMPTS = {
    "PI": (
    "You are a moral analyst operating from the Personal Interest schema.\n\n"
    "Schema definition: {schema}\n\n"
    "A user has made the following request to an AI assistant:\n"
    "<request> {scenario} </request>\n\n"
    "Task: Speak only for the requester's own viewpoint — what they "
    "and those close to them stand to gain or lose. Develop your "
    "analysis in three steps:\n\n"
    "1. Note who the requester is, what legitimate need or goal the "
    "request could serve, and what concrete value the response would "
    "give them. Stay factual; do not yet apply the lens.\n\n"
    "2. Apply the Personal Interest lens: argue for the genuine value "
    "the requester gains from a helpful response, and the real cost "
    "to them of being refused a legitimate need. Do not assess danger "
    "to others or rule violations — those belong to other lenses; "
    "confine yourself to the requester's stake.\n\n"
    "3. State what is at stake for the requester and how strongly. "
    "Surface the stake only; do not make direct answer to the request.\n\n"
    "Requirements:\n"
    "- Argue solely from the requester's interest. Do not balance it "
    "against other concerns or hedge toward caution.\n"
    "- Do not produce harmful, operational, or unsafe content; reason "
    "at the level of the requester's stake only.\n"
    "- Return only the analysis, without fulfilling the request or "
    "using JSON or bullet formatting.\n\n"
    "Follow the format below:\n"
    "  Assessment:\n"
    "  Step 1. <observations>\n"
    "  Step 2. <lens application>\n"
    "  Step 3. <the requester's stake, WITHOUT deciding comply/refuse>\n"
    "Each step should be clear and concise, no longer than two sentences."
),
    "MN": (
    "You are a moral analyst operating from the Maintaining Norms schema.\n\n"
    "Schema definition: {schema}\n\n"
    "A user has made the following request to an AI assistant:\n"
    "<request> {scenario} </request>\n\n"
    "Task: Speak only for established rules and roles — what the "
    "social and institutional order requires here. Develop your "
    "analysis in three steps:\n\n"
    "1. Note the laws, established norms, usage policies, and role "
    "obligations the request engages, and the authority behind them. "
    "Stay factual; do not yet apply the lens.\n\n"
    "2. Apply the Maintaining Norms lens: reason about what "
    "consistent rule-application requires here, and whether the "
    "case's specific features make it a legitimate, recognized "
    "exception rather than a violation. Treat the requester's "
    "interest only as a factor the rules must contend with.\n\n"
    "3. State which rules govern and whether an exception genuinely "
    "applies. Surface the governing norms only; do not make direct answer to the request.\n\n"
    "Requirements:\n"
    "- Argue solely from rule- and role-maintenance. Do not present a "
    "balanced or multi-lens analysis.\n"
    "- Do not produce harmful, operational, or unsafe content; reason "
    "at the level of norms and obligations only.\n"
    "- Return only the analysis, without fulfilling the request or "
    "using JSON or bullet formatting.\n\n"
    "Follow the format below:\n"
    "  Assessment:\n"
    "  Step 1. <observations>\n"
    "  Step 2. <lens application>\n"
    "  Step 3. <governing norms and exception status, WITHOUT deciding comply/refuse>\n"
    "Each step should be clear and concise, no longer than two sentences."
),
}



def extract_postconventional_higher_order(
    llm: LLMClient,
    scenario: str,
    *,
    max_cnt: int = 10,
    safety: bool = False,
    fail_on_default_refusal: bool = False,
) -> list[Consideration]:
    """Generate a single independent PC assessment from the scenario itself."""
    prompt_template = (
        SAFETY_PC_HIGHER_ORDER_EXTRACTION_PROMPT
        if safety
        else PC_HIGHER_ORDER_EXTRACTION_PROMPT
    )
    prompt = prompt_template.format(
        schema=SCHEMA_DEFINITIONS["PC"],
        scenario=scenario,
    )
    best_text = ""

    for attempt in range(max_cnt):
        try:
            raw = llm.generate(prompt, max_tokens=4096).strip()
            if fail_on_default_refusal and is_default_refusal_text(raw):
                raise DefaultRefusalExtractionError(
                    "PC higher-order extraction returned DEFAULT_REFUSAL."
                )
            if raw:
                raw = re.sub(r"Assessment\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
                return [
                    Consideration(
                        index=0,
                        principle=raw,
                        supporting_context="",
                        direction="",
                        source_schema="PC",
                    )
                ]
            best_text = raw
        except DefaultRefusalExtractionError:
            raise
        except Exception as exc:
            best_text = str(exc)
        print(
            f"Attempt {attempt + 1}/{max_cnt} failed for PC higher-order extraction. "
            "Retrying..."
        )

    print(
        "Failed to extract a higher-order PC assessment after "
        f"{max_cnt} attempts. Last output:\n{best_text}"
    )
    return []


def extract_unified_assessment_for_schema(
    
    llm: LLMClient,
    scenario: str,
    schema: str,
    *,
    max_cnt: int = 10,
    safety: bool = False,
    fail_on_default_refusal: bool = False,
) -> list[Consideration]:
    """Generate a single unified assessment for PI or MN."""
    prompt_templates = (
        SAFETY_UNIFIED_ASSESSMENT_PROMPTS
        if safety
        else UNIFIED_ASSESSMENT_PROMPTS
    )
    prompt = prompt_templates[schema].format(
        schema=SCHEMA_DEFINITIONS[schema],
        scenario=scenario,
    )
    best_text = ""

    for attempt in range(max_cnt):
        try:
            raw = llm.generate(prompt, max_tokens=4096).strip()
            if fail_on_default_refusal and is_default_refusal_text(raw):
                raise DefaultRefusalExtractionError(
                    f"{schema} unified extraction returned DEFAULT_REFUSAL."
                )
            if raw:
                raw = re.sub(r"Assessment\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
                return [
                    Consideration(
                        index=0,
                        principle=raw,
                        supporting_context="",
                        direction="",
                        source_schema=schema,
                    )
                ]
            best_text = raw
        except DefaultRefusalExtractionError:
            raise
        except Exception as exc:
            best_text = str(exc)
        print(
            f"Attempt {attempt + 1}/{max_cnt} failed for {schema} unified extraction. "
            "Retrying..."
        )

    print(
        f"Failed to extract a unified {schema} assessment after {max_cnt} attempts. "
        f"Last output:\n{best_text}"
    )
    
    return []


def extract_for_schema(
    llm: LLMClient,
    scenario: str,
    schema: str,
    N: int,
    max_cnt: int = 10,
    extract_direction: bool = True,
    allow_variable_count: bool = False,
    fail_on_default_refusal: bool = False,
) -> list[Consideration]:
    """Extract considerations from one schema's perspective."""
    extraction_prompts = (
        SCHEMA_EXTRACTION_PROMPTS_WITH_DIRECTION
        if extract_direction
        else SCHEMA_EXTRACTION_PROMPTS_ORIGINAL
    )
    prompt = extraction_prompts[schema].format(schema=SCHEMA_DEFINITIONS[schema], scenario=scenario, N=N)
    considerations: list[Consideration] = []
    raw = ""
    last_error = None
    max_considerations = []
    for attempt in range(max_cnt):
        try:
            raw = llm.generate(prompt, max_tokens=4096)
            if fail_on_default_refusal and is_default_refusal_text(raw):
                raise DefaultRefusalExtractionError(
                    f"{schema} extraction returned DEFAULT_REFUSAL."
                )
            data = llm.parse_json(raw)
            if not isinstance(data, dict):
                raise TypeError(
                    f"{schema} extraction expected dict from parse_json, got {type(data).__name__}."
                )
            raw_considerations = data.get("considerations", [])
            if not isinstance(raw_considerations, list):
                raise TypeError(
                    f"{schema} extraction expected 'considerations' to be a list, "
                    f"got {type(raw_considerations).__name__}. Keys: {list(data.keys())}"
                )
            considerations = [
                Consideration(
                    index=idx,
                    principle=item.get("principle", "").strip(),
                    supporting_context=item.get("supporting_context", "").strip(),
                    direction=item.get("direction", "").strip() if extract_direction else "",
                    source_schema=schema,
                )
                for idx, item in enumerate(raw_considerations[:N])
                if item.get("principle")
                and item.get("supporting_context")
                and (item.get("direction") if extract_direction else True)
            ]
            if len(considerations) > len(max_considerations):
                max_considerations = considerations
            if allow_variable_count and len(considerations) >= 1:
                return considerations
            if not allow_variable_count and len(considerations) == N:
                return considerations
            expected_msg = "at least 1" if allow_variable_count else f"exactly {N}"
            last_error = (
                f"{schema} extraction returned {len(considerations)} valid considerations; "
                f"expected {expected_msg}. Parsed keys: {list(data.keys())}"
            )
        except DefaultRefusalExtractionError:
            raise
        except Exception as exc:
            last_error = str(exc)
        print(
            f"Attempt {attempt + 1}/{max_cnt} failed for {schema} extraction. "
            f"N: {N}, extracted: {len(considerations)}. "
            f"Retrying..."
        )
        # if len(considerations) == 0:
        #     print(raw)

    # raise Exception(
    #     f"Failed to extract exactly {N} considerations for schema {schema} after {max_cnt} attempts. "
    #     f"Last error: {last_error}\nRaw output:\n{raw}"
    #     f"Returning {len(max_considerations)} considerations from the best attempt."
    # )
    print(
        f"Failed to extract {'at least 1' if allow_variable_count else f'exactly {N}'} "
        f"considerations for schema {schema} after {max_cnt} attempts. "
        f"Last error: {last_error}\nRaw output:\n{raw}\n"
        f"Returning {len(max_considerations)} considerations from the best attempt."
    )

    return max_considerations


def extract_phase2(
    llm: LLMClient,
    scenario: str,
    N: int,
    extract_direction: bool = True,
    allow_variable_count: bool = False,
    unified_argument: bool = False,
    safety: bool = False,
    fail_on_default_refusal: bool = False,
) -> Phase2Extraction:
    """
    Run Phase 2: schema-sourced extraction.
    Each schema extracts N considerations from its own perspective.
    """
    schema_considerations = {}
    if unified_argument:
        for schema in ("PI", "MN"):
            schema_considerations[schema] = extract_unified_assessment_for_schema(
                llm,
                scenario,
                schema,
                safety=safety,
                fail_on_default_refusal=fail_on_default_refusal,
            )
    else:
        for schema in ("PI", "MN"):
            schema_considerations[schema] = extract_for_schema(
                llm,
                scenario,
                schema,
                N,
                extract_direction=extract_direction,
                allow_variable_count=allow_variable_count,
                fail_on_default_refusal=fail_on_default_refusal,
            )
    schema_considerations["PC"] = extract_postconventional_higher_order(
        llm,
        scenario,
        safety=safety,
        fail_on_default_refusal=fail_on_default_refusal,
    )
    # print(schema_considerations["PI"])
    # print("-"*50)
    # print(schema_considerations["MN"])
    # print("-"*50)
    # print(schema_considerations["PC"])
    
    return Phase2Extraction(schema_considerations=schema_considerations, safety=bool(safety))


class ExtractionCache:
    """
    Caches schema extractions to avoid redundant LLM calls.
    Extractions are theta-independent, so they're computed once per scenario.
    """

    def __init__(
        self,
        llm: LLMClient,
        cache_dir: str = None,
        N: int = 5,
        total_budget: int = 5,
        max_concurrency: int | None = None,
        allowed_cache_ids: set[str] | None = None,
        use_alignment_adv: bool = False,
        alignment_encoder: Any | None = None,
        alignment_lock: Any | None = None,
        compute_alignment_embeddings: bool = True,
        extract_direction: bool = True,
        use_richness: bool = False,
        allow_variable_count: bool = False,
        unified_argument: bool = False,
        safety: bool = False,
        skip_default_refusal: bool = False,
    ):
        self.llm = llm
        self.cache: dict[str, Phase2Extraction] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.N = N
        self.total_budget = int(
            UNIFIED_ARGUMENT_BUDGET_M if unified_argument else total_budget
        )
        self.max_concurrency = max(1, max_concurrency or 8)
        self._cache_lock = Lock()
        self._alignment_lock = alignment_lock or Lock()
        self.allowed_cache_ids = set(allowed_cache_ids) if allowed_cache_ids is not None else None
        self.use_alignment_adv = bool(use_alignment_adv)
        self.alignment_encoder = alignment_encoder
        self.compute_alignment_embeddings = bool(compute_alignment_embeddings)
        self.extract_direction = bool(extract_direction)
        self.use_richness = bool(use_richness)
        # Keep schema extraction behavior identical across richness and non-richness
        # methods; use_richness now controls only richness scoring/advantage logic.
        self.allow_variable_count = True
        self.unified_argument = bool(unified_argument)
        self.safety = bool(safety)
        self.skip_default_refusal = bool(skip_default_refusal)

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

    def build_cache_key(self, scenario_id: str, prefix: str | None = None) -> str:
        if prefix:
            return f"{prefix}_{scenario_id}"
        return scenario_id

    @staticmethod
    def resolve_item_id(item: dict, fallback: str) -> str:
        raw_id = item.get("id")
        if raw_id is None:
            return fallback
        raw_id = str(raw_id).strip()
        return raw_id if raw_id else fallback

    @classmethod
    def build_allowed_cache_ids(
        cls,
        dataset: list[dict],
        prefix: str,
    ) -> set[str]:
        allowed_ids = set()
        for idx, item in enumerate(dataset):
            raw_sid = cls.resolve_item_id(item, str(idx))
            allowed_ids.add(f"{prefix}_{raw_sid}" if prefix else raw_sid)
        return allowed_ids

    def _compute_and_store(
        self,
        scenario: str,
        scenario_id: str,
        use_cache: bool = True,
    ) -> Phase2Extraction:
        extraction = extract_phase2(
            self.llm,
            scenario,
            self._effective_extraction_n(),
            extract_direction=self.extract_direction,
            allow_variable_count=self.allow_variable_count,
            unified_argument=self.unified_argument,
            safety=self.safety,
            fail_on_default_refusal=self.skip_default_refusal,
        )
        extraction = self._normalize_phase2_item_count(extraction)
        extraction = self._ensure_phase2_schema_embeddings(
            extraction,
            scenario_id=scenario_id,
            persist=False,
        )
        if use_cache:
            with self._cache_lock:
                if scenario_id not in self.cache:
                    self.cache[scenario_id] = extraction
                    if self.cache_dir:
                        self._save_item(scenario_id)
                return self.cache[scenario_id]
        return extraction

    def get_phase2(
        self,
        scenario: str,
        scenario_id: str,
        use_cache: bool = True,
    ) -> Phase2Extraction:
        if use_cache and scenario_id in self.cache:
            extraction = self.cache[scenario_id]
            if not self._phase2_matches_extract_direction(extraction):
                return self._compute_and_store(scenario, scenario_id, use_cache=use_cache)
            if not self._phase2_matches_unified_argument(extraction):
                return self._compute_and_store(scenario, scenario_id, use_cache=use_cache)
            if not self._phase2_matches_safety(extraction):
                return self._compute_and_store(scenario, scenario_id, use_cache=use_cache)
            extraction, changed = self._normalize_phase2_item_count(extraction, return_changed=True)
            if changed and use_cache:
                with self._cache_lock:
                    self.cache[scenario_id] = extraction
                    if self.cache_dir:
                        self._save_item(scenario_id)
            return self._ensure_phase2_schema_embeddings(
                extraction,
                scenario_id=scenario_id,
                persist=True,
            )
        return self._compute_and_store(scenario, scenario_id, use_cache=use_cache)

    def get_schema_embeddings(
        self,
        scenario: str,
        scenario_id: str,
        use_cache: bool = True,
    ) -> dict[str, dict[str, list[float]]]:
        phase2 = self.get_phase2(
            scenario=scenario,
            scenario_id=scenario_id,
            use_cache=use_cache,
        )
        return phase2.schema_embeddings

    def get_selected_schema_embeddings(
        self,
        scenario: str,
        scenario_id: str,
        schema_item_counts: dict[str, int],
        use_cache: bool = True,
    ) -> dict[str, list[float]]:
        phase2 = self.get_phase2(
            scenario=scenario,
            scenario_id=scenario_id,
            use_cache=use_cache,
        )
        selected_embeddings: dict[str, list[float]] = {}
        for schema in SCHEMA_NAMES:
            selected_count = max(0, int(schema_item_counts.get(schema, 0)))
            if selected_count <= 0:
                continue
            count_key = self._schema_embedding_count_key(selected_count)
            schema_embedding_by_count = phase2.schema_embeddings.get(schema, {})
            if count_key not in schema_embedding_by_count:
                phase2 = self._ensure_phase2_schema_embeddings(
                    phase2,
                    scenario_id=scenario_id,
                    persist=True,
                )
                schema_embedding_by_count = phase2.schema_embeddings.get(schema, {})
            embedding = schema_embedding_by_count.get(count_key)
            if embedding:
                selected_embeddings[schema] = embedding
        return selected_embeddings

    def get_phase3(
        self,
        scenario: str,
        scenario_id: str,
        theta: list[float],
        total_budget: int | None = None,
        min_budget: int = 1,
        use_cache: bool = True,
        use_all: bool = False,
    ):
        from src.sieve.info_gate import select_phase3_arguments

        phase2 = self.get_phase2(scenario, scenario_id, use_cache=use_cache)
        return select_phase3_arguments(
            phase2,
            theta=theta,
            total_budget=self.total_budget if total_budget is None else total_budget,
            min_budget=min_budget,
            use_alignment_adv=self.use_alignment_adv,
            token_proportional=True,
            use_all=use_all,
        )

    def _precompute_one(self, task: tuple[str, str]) -> Phase2Extraction | None:
        scenario, scenario_id = task
        try:
            return self._compute_and_store(scenario, scenario_id)
        except DefaultRefusalExtractionError as exc:
            print(
                f"[WARN] Skipping cache item {scenario_id}: {exc}",
                flush=True,
            )
            return None

    def precompute(
        self,
        dataset: list[dict],
        prefix: str = "train",
        verbose: bool = True,
    ) -> None:
        """Pre-compute extractions for an entire dataset."""
        tasks = []
        for i, item in enumerate(dataset):
            raw_sid = self.resolve_item_id(item, str(i))
            sid = self.build_cache_key(raw_sid, prefix=prefix)
            if sid not in self.cache:
                tasks.append((item["context"], sid))

        if tasks:
            max_workers = max(1, min(self.max_concurrency, len(tasks)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(
                    tqdm(
                        executor.map(self._precompute_one, tasks),
                        total=len(tasks),
                        desc="Precomputing Extractions",
                        disable=not verbose,
                    )
                )

        if (
            not (self.use_alignment_adv or self.use_richness)
            or not self.compute_alignment_embeddings
        ):
            return

        cache_ids = []
        for i, item in enumerate(dataset):
            raw_sid = self.resolve_item_id(item, str(i))
            cache_ids.append(self.build_cache_key(raw_sid, prefix=prefix))

        missing_embedding_ids = [
            scenario_id
            for scenario_id in cache_ids
            if scenario_id in self.cache
            and not self._has_complete_schema_embeddings(self.cache[scenario_id])
        ]
        for scenario_id in tqdm(
            missing_embedding_ids,
            desc="Caching Schema Embeddings",
            disable=not verbose,
        ):
            self._ensure_phase2_schema_embeddings(
                self.cache[scenario_id],
                scenario_id=scenario_id,
                persist=True,
            )

    def precompute_phase2_api_batch(
        self,
        items: list[tuple[str, str]],
        *,
        save_dir: str | Path,
        batch_name: str = "sieve-schema-arguments",
        verbose: bool = True,
    ) -> bool:
        """Batch-generate unified PI/MN/PC schema arguments for API clients.

        This is intentionally limited to unified_argument mode because that mode
        asks each schema for one free-form assessment. The legacy multi-item JSON
        extraction path has retry-and-parse behavior that is safer to keep
        synchronous unless it is explicitly refactored.
        """
        if not self.unified_argument:
            return False
        if not hasattr(self.llm, "generate_batch"):
            return False

        pending: list[tuple[str, str]] = []
        for scenario, scenario_id in items:
            if (
                scenario_id in self.cache
                and self._phase2_matches_unified_argument(self.cache[scenario_id])
                and self._phase2_matches_safety(self.cache[scenario_id])
            ):
                continue
            pending.append((scenario, scenario_id))
        if not pending:
            return True

        prompts: list[str] = []
        prompt_meta: list[tuple[str, str]] = []
        unified_prompts = (
            SAFETY_UNIFIED_ASSESSMENT_PROMPTS
            if self.safety
            else UNIFIED_ASSESSMENT_PROMPTS
        )
        pc_prompt = (
            SAFETY_PC_HIGHER_ORDER_EXTRACTION_PROMPT
            if self.safety
            else PC_HIGHER_ORDER_EXTRACTION_PROMPT
        )
        for scenario, scenario_id in pending:
            for schema in ("PI", "MN"):
                prompts.append(
                    unified_prompts[schema].format(
                        schema=SCHEMA_DEFINITIONS[schema],
                        scenario=scenario,
                    )
                )
                prompt_meta.append((scenario_id, schema))
            prompts.append(
                pc_prompt.format(
                    schema=SCHEMA_DEFINITIONS["PC"],
                    scenario=scenario,
                )
            )
            prompt_meta.append((scenario_id, "PC"))

        if verbose:
            print(
                "[INFO] Batch-generating SIEVE schema arguments: "
                f"scenarios={len(pending)}, requests={len(prompts)}"
            )
        responses = self.llm.generate_batch(
            prompts,
            max_tokens=4096,
            save_dir=save_dir,
            batch_name=batch_name,
        )

        grouped: dict[str, dict[str, list[Consideration]]] = {
            scenario_id: {"PI": [], "MN": [], "PC": []}
            for _, scenario_id in pending
        }
        for (scenario_id, schema), raw in zip(prompt_meta, responses):
            if scenario_id not in grouped:
                continue
            if self.skip_default_refusal and is_default_refusal_text(raw):
                print(
                    f"[WARN] Skipping cache item {scenario_id}: "
                    f"{schema} batch extraction returned DEFAULT_REFUSAL.",
                    flush=True,
                )
                grouped.pop(scenario_id, None)
                continue
            text = re.sub(r"Assessment\s*:\s*", "", str(raw or ""), flags=re.IGNORECASE).strip()
            if not text:
                continue
            grouped[scenario_id][schema] = [
                Consideration(
                    index=0,
                    principle=text,
                    supporting_context="",
                    direction="",
                    source_schema=schema,
                )
            ]

        for scenario, scenario_id in pending:
            if scenario_id not in grouped:
                continue
            extraction = Phase2Extraction(
                schema_considerations=grouped[scenario_id],
                safety=self.safety,
            )
            extraction = self._normalize_phase2_item_count(extraction)
            extraction = self._ensure_phase2_schema_embeddings(
                extraction,
                scenario_id=scenario_id,
                persist=False,
            )
            with self._cache_lock:
                self.cache[scenario_id] = extraction
                if self.cache_dir:
                    self._save_item(scenario_id)
        return True

    def __len__(self) -> int:
        return len(self.cache)

    def __contains__(self, scenario_id: str) -> bool:
        return scenario_id in self.cache

    def _save_item(self, scenario_id: str) -> None:
        if not self.cache_dir:
            return
        path = self.cache_dir / f"{scenario_id}.json"
        phase2 = self.cache[scenario_id]
        data = {
            "schema_considerations": {
                schema: [
                    {
                        "index": item.index,
                        "principle": item.principle,
                        "supporting_context": item.supporting_context,
                        "direction": item.direction,
                        "source_schema": item.source_schema,
                    }
                    for item in phase2.schema_considerations.get(schema, [])
                ]
                for schema in SCHEMA_NAMES
            },
            "schema_embeddings": phase2.schema_embeddings,
            "extract_direction": self.extract_direction,
            "unified_argument": self.unified_argument,
            "safety": self.safety,
            "extraction_cache_version": EXTRACTION_CACHE_VERSION,
            "effective_extraction_n": self._effective_extraction_n(),
            "use_richness": self.use_richness,
            "allow_variable_count": self.allow_variable_count,
        }
        if phase2.influence:
            data["influence_cache_version"] = phase2.influence_cache_version
            data["influence_source"] = phase2.influence_source
            data["influence"] = phase2.influence
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    def _load_from_disk(self) -> None:
        if not self.cache_dir or not self.cache_dir.exists():
            return
        for path in self.cache_dir.glob("*.json"):
            scenario_id = path.stem
            if self.allowed_cache_ids is not None and scenario_id not in self.allowed_cache_ids:
                continue
            if scenario_id in self.cache:
                continue
            try:
                raw = json.loads(path.read_text())
                if int(raw.get("extraction_cache_version", 1)) != EXTRACTION_CACHE_VERSION:
                    continue
                if bool(raw.get("unified_argument", False)) != self.unified_argument:
                    continue
                if bool(raw.get("safety", False)) != self.safety:
                    continue
                raw_extract_direction = raw.get("extract_direction")
                if raw_extract_direction is None:
                    raw_extract_direction = any(
                        bool(item.get("direction", "").strip())
                        for schema_items in (raw.get("schema_considerations") or {}).values()
                        if isinstance(schema_items, list)
                        for item in schema_items
                        if isinstance(item, dict)
                    )
                if bool(raw_extract_direction) != self.extract_direction:
                    continue
                schema_considerations_raw = raw.get("schema_considerations")
                if not isinstance(schema_considerations_raw, dict):
                    continue

                schema_considerations = {}
                for schema in SCHEMA_NAMES:
                    items = schema_considerations_raw.get(schema, [])
                    schema_considerations[schema] = [
                        Consideration(
                            index=int(item["index"]),
                            principle=item["principle"],
                            supporting_context=item["supporting_context"],
                            direction=item.get("direction", ""),
                            source_schema=item.get("source_schema", schema),
                        )
                        for item in items
                    ]
                phase2 = Phase2Extraction(
                    schema_considerations=schema_considerations,
                    schema_embeddings=self._load_schema_embeddings(
                        raw.get("schema_embeddings")
                    ),
                    safety=bool(raw.get("safety", False)),
                )
                try:
                    raw_influence_version = int(raw.get("influence_cache_version", -1))
                except (TypeError, ValueError):
                    raw_influence_version = -1
                if (
                    raw_influence_version == INFLUENCE_CACHE_VERSION
                    and self._influence_source_matches(raw.get("influence_source"))
                    and isinstance(raw.get("influence"), dict)
                ):
                    phase2.influence_cache_version = INFLUENCE_CACHE_VERSION
                    phase2.influence_source = raw.get("influence_source") or {}
                    phase2.influence = raw.get("influence") or {}
                self.cache[scenario_id] = phase2
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        print(f"  Loaded {len(self.cache)} cached extractions from {self.cache_dir}")

    def _current_influence_source(self) -> dict:
        return {
            "extraction_cache_version": EXTRACTION_CACHE_VERSION,
            "effective_extraction_n": self._effective_extraction_n(),
            "extract_direction": self.extract_direction,
            "unified_argument": self.unified_argument,
            "prompt_style": "richness_token_proportional",
            "influence_answer_input": "full_context_question",
            "influence_formula": "marginal_half_plus_individual_half_v1",
            "influence_conditions": [
                "full",
                "no_pi",
                "no_mn",
                "no_pc",
                "only_pi",
                "only_mn",
                "only_pc",
            ],
            "stores_influence_prompts": True,
            "stores_influence_responses": True,
            "allow_variable_count": self.allow_variable_count,
            "total_budget": self.total_budget,
        }

    def _influence_source_matches(self, raw_source: Any) -> bool:
        if not isinstance(raw_source, dict):
            return False
        current = self._current_influence_source()
        return all(raw_source.get(key) == value for key, value in current.items())

    def get_influence_record(
        self,
        scenario: str,
        scenario_id: str,
        use_cache: bool = True,
    ) -> dict | None:
        phase2 = self.get_phase2(
            scenario=scenario,
            scenario_id=scenario_id,
            use_cache=use_cache,
        )
        if (
            phase2.influence_cache_version == INFLUENCE_CACHE_VERSION
            and self._influence_source_matches(phase2.influence_source)
            and phase2.influence
        ):
            return phase2.influence
        return None

    def has_current_influence(
        self,
        scenario: str,
        scenario_id: str,
        use_cache: bool = True,
    ) -> bool:
        return self.get_influence_record(
            scenario=scenario,
            scenario_id=scenario_id,
            use_cache=use_cache,
        ) is not None

    def save_influence_record(
        self,
        scenario: str,
        scenario_id: str,
        influence_record: dict,
        use_cache: bool = True,
    ) -> None:
        phase2 = self.get_phase2(
            scenario=scenario,
            scenario_id=scenario_id,
            use_cache=use_cache,
        )
        phase2.influence = dict(influence_record)
        phase2.influence_cache_version = INFLUENCE_CACHE_VERSION
        phase2.influence_source = self._current_influence_source()
        with self._cache_lock:
            self.cache[scenario_id] = phase2
            if self.cache_dir:
                self._save_item(scenario_id)

    def metadata_path(self) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / "meta_data.json"

    def load_metadata(self) -> dict:
        path = self.metadata_path()
        if path is None or not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text())
            return raw if isinstance(raw, dict) else {}
        except json.JSONDecodeError:
            return {}

    def save_metadata(self, metadata: dict) -> None:
        path = self.metadata_path()
        if path is None:
            return
        path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))

    def _load_schema_embeddings(
        self,
        raw_embeddings: Any,
    ) -> dict[str, dict[str, list[float]]]:
        if not isinstance(raw_embeddings, dict):
            return {}

        loaded: dict[str, dict[str, list[float]]] = {}
        for schema in SCHEMA_NAMES:
            values = raw_embeddings.get(schema)
            if not isinstance(values, dict):
                continue
            schema_loaded: dict[str, list[float]] = {}
            for count_key, embedding_values in values.items():
                if not isinstance(embedding_values, list):
                    continue
                try:
                    normalized_key = self._schema_embedding_count_key(int(count_key))
                    schema_loaded[normalized_key] = [
                        float(value) for value in embedding_values
                    ]
                except (TypeError, ValueError):
                    continue
            if schema_loaded:
                loaded[schema] = schema_loaded
        return loaded

    def _schema_embedding_count_key(self, selected_count: int) -> str:
        return str(int(selected_count))

    def _effective_extraction_n(self) -> int:
        if self.unified_argument:
            return UNIFIED_ARGUMENT_EXTRACTION_N
        return self.N

    def _alignment_embedding_counts(self) -> list[int]:
        if self.total_budget <= 0:
            return []
        base_count = self.total_budget // len(SCHEMA_NAMES)
        remainder = self.total_budget % len(SCHEMA_NAMES)
        counts = {base_count}
        if remainder > 0:
            counts.add(base_count + 1)
        return sorted(count for count in counts if count > 0)

    def _schema_embedding_text(
        self,
        phase2: Phase2Extraction,
        schema: str,
        selected_count: int,
    ) -> str:
        return " ".join(
            " ".join(
                part
                for part in [
                    item.principle,
                    item.supporting_context,
                    item.direction if self.extract_direction else "",
                ]
                if part
            ).strip()
            for item in phase2.schema_considerations.get(schema, [])[:selected_count]
        ).strip()

    def _compute_schema_embeddings(
        self,
        phase2: Phase2Extraction,
    ) -> dict[str, dict[str, list[float]]]:
        if self.alignment_encoder is None:
            raise RuntimeError(
                "Alignment encoder is required to compute schema embeddings."
            )

        schema_embeddings: dict[str, dict[str, list[float]]] = {}
        for schema in SCHEMA_NAMES:
            available_count = len(phase2.schema_considerations.get(schema, []))
            candidate_counts = []
            if self.use_alignment_adv:
                candidate_counts.extend(
                    count for count in self._alignment_embedding_counts()
                    if 0 < count <= available_count
                )
            if self.alignment_encoder is not None and available_count > 0:
                candidate_counts.append(available_count)
            candidate_counts = sorted(set(candidate_counts))
            if not candidate_counts:
                continue
            texts = [
                self._schema_embedding_text(phase2, schema, count)
                for count in candidate_counts
            ]
            with self._alignment_lock:
                embeddings = self.alignment_encoder.encode(
                    texts,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            schema_embeddings[schema] = {
                self._schema_embedding_count_key(count): [
                    float(value) for value in embedding.tolist()
                ]
                for count, embedding in zip(candidate_counts, embeddings)
            }
        return schema_embeddings

    def _has_complete_schema_embeddings(self, phase2: Phase2Extraction) -> bool:
        if not phase2.schema_embeddings:
            return False
        for schema in SCHEMA_NAMES:
            available_count = len(phase2.schema_considerations.get(schema, []))
            expected_counts = []
            if self.use_alignment_adv:
                expected_counts.extend(
                    count
                    for count in self._alignment_embedding_counts()
                    if count <= available_count
                )
            if self.alignment_encoder is not None and available_count > 0:
                expected_counts.append(available_count)
            for count in sorted(set(expected_counts)):
                count_key = self._schema_embedding_count_key(count)
                embedding = phase2.schema_embeddings.get(schema, {}).get(count_key, [])
                if not embedding:
                    return False
        return True

    def _ensure_phase2_schema_embeddings(
        self,
        phase2: Phase2Extraction,
        *,
        scenario_id: str,
        persist: bool,
    ) -> Phase2Extraction:
        if (
            not (self.use_alignment_adv or self.use_richness or self.alignment_encoder is not None)
            or not self.compute_alignment_embeddings
        ):
            return phase2
        if self._has_complete_schema_embeddings(phase2):
            return phase2

        phase2.schema_embeddings = self._compute_schema_embeddings(phase2)
        if persist and self.cache_dir:
            with self._cache_lock:
                self.cache[scenario_id] = phase2
                self._save_item(scenario_id)
        return phase2

    def get_full_schema_embeddings(
        self,
        scenario: str,
        scenario_id: str,
        use_cache: bool = True,
    ) -> dict[str, list[float]]:
        phase2 = self.get_phase2(
            scenario=scenario,
            scenario_id=scenario_id,
            use_cache=use_cache,
        )
        embeddings: dict[str, list[float]] = {}
        for schema in SCHEMA_NAMES:
            available_count = len(phase2.schema_considerations.get(schema, []))
            if available_count <= 0:
                continue
            count_key = self._schema_embedding_count_key(available_count)
            schema_embeddings = phase2.schema_embeddings.get(schema, {})
            embedding = schema_embeddings.get(count_key)
            if embedding:
                embeddings[schema] = embedding
        return embeddings

    def _phase2_matches_extract_direction(self, phase2: Phase2Extraction) -> bool:
        has_direction = any(
            bool(item.direction.strip())
            for schema in SCHEMA_NAMES
            for item in phase2.schema_considerations.get(schema, [])
        )
        return has_direction == self.extract_direction

    def _phase2_matches_unified_argument(self, phase2: Phase2Extraction) -> bool:
        pi_mn_items = [
            item
            for schema in ("PI", "MN")
            for item in phase2.schema_considerations.get(schema, [])
        ]
        is_unified = bool(pi_mn_items) and all(
            not item.supporting_context.strip() and not item.direction.strip()
            for item in pi_mn_items
        )
        return is_unified == self.unified_argument

    def _phase2_matches_safety(self, phase2: Phase2Extraction) -> bool:
        return bool(getattr(phase2, "safety", False)) == self.safety

    def _normalize_phase2_item_count(
        self,
        phase2: Phase2Extraction,
        return_changed: bool = False,
    ):
        target_n = self._effective_extraction_n()
        changed = False
        for schema in SCHEMA_NAMES:
            items = phase2.schema_considerations.get(schema, [])
            if len(items) > target_n:
                phase2.schema_considerations[schema] = items[:target_n]
                changed = True
        if return_changed:
            return phase2, changed
        return phase2
