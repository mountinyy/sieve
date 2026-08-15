from __future__ import annotations

"""
PRISM: Information Gate.
Deterministic Phase 3 budget allocation and selection.
"""

from src.sieve.data_types import (
    Phase2Extraction,
    Phase3Selection,
    SCHEMA_NAMES,
)


def allocate_phase3_budgets(
    theta: list[float],
    total_budget: int,
    min_budget: int = 1,
) -> tuple[str, str, str, int, int, int]:
    """
    Allocate budgets to all three schemas using theta.
    """
    ranked = sorted(
        zip(SCHEMA_NAMES, theta),
        key=lambda item: item[1],
        reverse=True,
    )
    primary_schema = ranked[0][0]
    secondary_schema = ranked[1][0]
    tertiary_schema = ranked[2][0]

    if total_budget <= 0:
        return primary_schema, secondary_schema, tertiary_schema, 0, 0, 0

    budgets = {schema: 0 for schema in SCHEMA_NAMES}
    remaining = total_budget

    if total_budget >= len(SCHEMA_NAMES) * min_budget:
        for schema in SCHEMA_NAMES:
            budgets[schema] = min_budget
            remaining -= min_budget
    else:
        for schema, _ in ranked:
            if remaining <= 0:
                break
            budgets[schema] += 1
            remaining -= 1

    raw_allocations = {
        schema: theta_i * total_budget for schema, theta_i in zip(SCHEMA_NAMES, theta)
    }
    while remaining > 0:
        target_schema = max(
            SCHEMA_NAMES,
            key=lambda schema: (
                raw_allocations[schema] - budgets[schema],
                raw_allocations[schema],
            ),
        )
        budgets[target_schema] += 1
        remaining -= 1

    return (
        primary_schema,
        secondary_schema,
        tertiary_schema,
        budgets[primary_schema],
        budgets[secondary_schema],
        budgets[tertiary_schema],
    )


def rebalance_phase3_budgets(
    *,
    schema_order: list[str],
    theta_by_schema: dict[str, float],
    initial_budgets: dict[str, int],
    available_counts: dict[str, int],
    total_budget: int,
) -> dict[str, int]:
    """
    Rebalance budgets so the number of actually selectable arguments sums to total_budget
    whenever the schema pools contain enough total items.
    """
    budgets = {
        schema: min(initial_budgets.get(schema, 0), available_counts.get(schema, 0))
        for schema in schema_order
    }
    remaining = min(total_budget, sum(max(0, available_counts.get(schema, 0)) for schema in schema_order))
    remaining -= sum(budgets.values())

    while remaining > 0:
        expandable = [
            schema for schema in schema_order
            if budgets[schema] < available_counts.get(schema, 0)
        ]
        if not expandable:
            break
        target_schema = max(
            expandable,
            key=lambda schema: (
                theta_by_schema.get(schema, 0.0) - budgets[schema],
                theta_by_schema.get(schema, 0.0),
                -schema_order.index(schema),
            ),
        )
        budgets[target_schema] += 1
        remaining -= 1

    return budgets


def allocate_fixed_phase3_budgets(
    theta: list[float],
    total_budget: int,
) -> tuple[str, str, str, int, int, int]:
    ranked = sorted(
        zip(SCHEMA_NAMES, theta),
        key=lambda item: item[1],
        reverse=True,
    )
    primary_schema = ranked[0][0]
    secondary_schema = ranked[1][0]
    tertiary_schema = ranked[2][0]

    if total_budget <= 0:
        return primary_schema, secondary_schema, tertiary_schema, 0, 0, 0

    base_budget = total_budget // len(SCHEMA_NAMES)
    remainder = total_budget % len(SCHEMA_NAMES)
    ranked_order = [primary_schema, secondary_schema, tertiary_schema]
    budgets = {schema: base_budget for schema in SCHEMA_NAMES}
    for schema in ranked_order[:remainder]:
        budgets[schema] += 1

    return (
        primary_schema,
        secondary_schema,
        tertiary_schema,
        budgets[primary_schema],
        budgets[secondary_schema],
        budgets[tertiary_schema],
    )


def rebalance_fixed_phase3_budgets(
    *,
    schema_order: list[str],
    initial_budgets: dict[str, int],
    available_counts: dict[str, int],
    total_budget: int,
) -> dict[str, int]:
    budgets = {
        schema: min(initial_budgets.get(schema, 0), available_counts.get(schema, 0))
        for schema in schema_order
    }
    remaining = min(
        total_budget,
        sum(max(0, available_counts.get(schema, 0)) for schema in schema_order),
    ) - sum(budgets.values())

    while remaining > 0:
        expandable = [
            schema for schema in schema_order
            if budgets[schema] < available_counts.get(schema, 0)
        ]
        if not expandable:
            break
        for schema in expandable:
            if remaining <= 0:
                break
            budgets[schema] += 1
            remaining -= 1

    return budgets


def select_phase3_arguments(
    phase2: Phase2Extraction,
    theta: list[float],
    total_budget: int = 5,
    min_budget: int = 1,
    use_alignment_adv: bool = False,
    token_proportional: bool = False,
    use_all: bool = False,
) -> Phase3Selection:
    """
    Deterministic Phase 3 selection.
    theta determines primary/secondary/tertiary and budget allocation.
    Each schema's considerations are already ordered by extraction importance.
    """
    if use_alignment_adv:
        (
            primary_schema,
            secondary_schema,
            tertiary_schema,
            primary_budget,
            secondary_budget,
            tertiary_budget,
        ) = allocate_fixed_phase3_budgets(theta, total_budget=total_budget)
    else:
        (
            primary_schema,
            secondary_schema,
            tertiary_schema,
            primary_budget,
            secondary_budget,
            tertiary_budget,
        ) = allocate_phase3_budgets(theta, total_budget=total_budget, min_budget=min_budget)

    primary_pool = phase2.schema_considerations.get(primary_schema, [])
    secondary_pool = phase2.schema_considerations.get(secondary_schema, [])
    tertiary_pool = phase2.schema_considerations.get(tertiary_schema, [])
    if use_all:
        return Phase3Selection(
            theta=theta,
            primary_schema=primary_schema,
            secondary_schema=secondary_schema,
            tertiary_schema=tertiary_schema,
            total_budget=total_budget,
            primary_budget=len(primary_pool),
            secondary_budget=len(secondary_pool),
            tertiary_budget=len(tertiary_pool),
            primary_arguments=primary_pool,
            secondary_arguments=secondary_pool,
            tertiary_principles=tertiary_pool,
        )
    if token_proportional:
        return Phase3Selection(
            theta=theta,
            primary_schema=primary_schema,
            secondary_schema=secondary_schema,
            tertiary_schema=tertiary_schema,
            total_budget=total_budget,
            primary_budget=len(primary_pool),
            secondary_budget=len(secondary_pool),
            tertiary_budget=len(tertiary_pool),
            primary_arguments=primary_pool,
            secondary_arguments=secondary_pool,
            tertiary_principles=tertiary_pool,
        )
    initial_budgets = {
        primary_schema: primary_budget,
        secondary_schema: secondary_budget,
        tertiary_schema: tertiary_budget,
    }
    available_counts = {
        primary_schema: len(primary_pool),
        secondary_schema: len(secondary_pool),
        tertiary_schema: len(tertiary_pool),
    }
    if use_alignment_adv:
        rebalanced_budgets = rebalance_fixed_phase3_budgets(
            schema_order=[primary_schema, secondary_schema, tertiary_schema],
            initial_budgets=initial_budgets,
            available_counts=available_counts,
            total_budget=total_budget,
        )
    else:
        theta_by_schema = {schema: theta_i for schema, theta_i in zip(SCHEMA_NAMES, theta)}
        rebalanced_budgets = rebalance_phase3_budgets(
            schema_order=[primary_schema, secondary_schema, tertiary_schema],
            theta_by_schema=theta_by_schema,
            initial_budgets=initial_budgets,
            available_counts=available_counts,
            total_budget=total_budget,
        )
    primary_budget = rebalanced_budgets[primary_schema]
    secondary_budget = rebalanced_budgets[secondary_schema]
    tertiary_budget = rebalanced_budgets[tertiary_schema]

    return Phase3Selection(
        theta=theta,
        primary_schema=primary_schema,
        secondary_schema=secondary_schema,
        tertiary_schema=tertiary_schema,
        total_budget=total_budget,
        primary_budget=primary_budget,
        secondary_budget=secondary_budget,
        tertiary_budget=tertiary_budget,
        primary_arguments=primary_pool[:primary_budget],
        secondary_arguments=secondary_pool[:secondary_budget],
        tertiary_principles=tertiary_pool[:tertiary_budget],
    )
