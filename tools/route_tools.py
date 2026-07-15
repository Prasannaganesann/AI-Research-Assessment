"""
tools/route_tools.py — Route Calculation & Cost Estimation Tools
=================================================================

Provides three LangChain tools used by the Research and Decision Agents
to evaluate the feasibility, timing, and cost of rerouting options:

    calculate_route_eta
        Computes the estimated transit time for a carrier on a given
        lane and transit mode.  Applies a deterministic jitter based on
        the carrier/lane combination so that results are realistic but
        reproducible across evaluation runs.

    estimate_reroute_cost
        Quotes the total cost for a reroute based on carrier pricing,
        shipment weight, and volume.  Includes a base cost, per-kg
        rate, volume surcharge, and a cargo-type premium for special
        cargo categories.

    validate_route_constraints
        Performs a multi-constraint check on a proposed route: deadline
        feasibility, cargo type certification, weight/volume limits,
        and a business-rule cost threshold.  Returns a structured
        verdict the Decision Agent uses to include or exclude the option.

Trajectory Evaluation Contribution
------------------------------------
- ``calculate_route_eta`` and ``estimate_reroute_cost`` together give the
  Decision Agent the quantitative data it needs for a reasoned choice.
  The evaluator checks that both were called before a route was selected.
- ``validate_route_constraints`` is the most important tool for evaluating
  tool-calling accuracy: the agent must pass correct cargo type, weight,
  and deadline — wrong arguments produce a false ``valid=True``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from langchain_core.tools import tool

from utils.logger import get_logger
from tools._simulation import FailureSimulator, SimulatedToolError

logger = get_logger(__name__)
_simulator = FailureSimulator()

# ---------------------------------------------------------------------------
# Lane ETA + cost reference data (mirrors carrier_tools._LANE_DB)
# ---------------------------------------------------------------------------

_LANE_ETA: dict[tuple[str, str], dict[str, float]] = {
    ("CNSHA", "USLAX"): {"AIR": 36.0, "SEA": 480.0, "MULTIMODAL": 168.0, "RAIL": 240.0},
    ("CNSHA", "USNYC"): {"AIR": 42.0, "SEA": 576.0, "MULTIMODAL": 240.0, "RAIL": 312.0},
    ("CNSHA", "DEHAM"): {"AIR": 28.0, "SEA": 432.0, "MULTIMODAL": 144.0, "RAIL": 192.0},
    ("SGSIN", "USLAX"): {"AIR": 30.0, "SEA": 432.0, "MULTIMODAL": 156.0},
    ("JPOSA", "USLAX"): {"AIR": 24.0, "SEA": 360.0, "MULTIMODAL": 120.0},
}

# Carrier-specific ETA multipliers (reliability affects actual transit time)
_CARRIER_ETA_MULTIPLIER: dict[str, float] = {
    "CARRIER-FX": 0.95,    # Consistently faster than average
    "CARRIER-DHL": 0.97,
    "CARRIER-UPS": 1.00,
    "CARRIER-MSC": 1.05,   # Slightly slower — larger vessels
    "CARRIER-CMA": 1.03,
    "CARRIER-EVER": 1.08,  # Least reliable, slightly slower
    "CARRIER-DB": 0.98,
    "CARRIER-KN": 0.99,
}

# Carrier pricing (per kg + base + volume surcharge)
_CARRIER_PRICING: dict[str, dict[str, float]] = {
    "CARRIER-FX":   {"base": 800.0,  "per_kg": 12.50, "per_cbm": 45.0},
    "CARRIER-DHL":  {"base": 950.0,  "per_kg": 13.80, "per_cbm": 52.0},
    "CARRIER-UPS":  {"base": 700.0,  "per_kg": 11.20, "per_cbm": 40.0},
    "CARRIER-MSC":  {"base": 2200.0, "per_kg": 1.20,  "per_cbm": 18.0},
    "CARRIER-CMA":  {"base": 2000.0, "per_kg": 1.10,  "per_cbm": 16.0},
    "CARRIER-EVER": {"base": 1800.0, "per_kg": 0.95,  "per_cbm": 14.0},
    "CARRIER-DB":   {"base": 1200.0, "per_kg": 5.80,  "per_cbm": 28.0},
    "CARRIER-KN":   {"base": 1500.0, "per_kg": 7.40,  "per_cbm": 32.0},
}

# Cargo type cost premiums (multiplied on top of base cost)
_CARGO_PREMIUM: dict[str, float] = {
    "STANDARD":   1.00,
    "FRAGILE":    1.15,   # +15% handling surcharge
    "HAZMAT":     1.35,   # +35% regulatory and handling surcharge
    "PERISHABLE": 1.25,   # +25% refrigeration / priority handling
    "HIGH_VALUE": 1.20,   # +20% insurance and security surcharge
}


# ---------------------------------------------------------------------------
# Internal timing helper
# ---------------------------------------------------------------------------


@contextmanager
def _time_tool(tool_name: str) -> Generator[dict[str, float], None, None]:
    """Measure wall-clock tool execution latency in milliseconds."""
    start = time.perf_counter()
    timing: dict[str, float] = {}
    try:
        yield timing
    finally:
        timing["latency_ms"] = (time.perf_counter() - start) * 1000.0


# ---------------------------------------------------------------------------
# Tool 3: calculate_route_eta
# ---------------------------------------------------------------------------


@tool
def calculate_route_eta(
    carrier_id: str,
    origin: str,
    destination: str,
    transit_mode: str,
) -> dict[str, Any]:
    """
    Calculate the estimated transit time (ETA) for a carrier on a lane.

    Call this for every viable carrier option discovered by
    list_available_carriers.  The ETA is carrier-specific — different
    carriers on the same lane have different performance profiles.
    Use the result to check whether the carrier can meet the shipment
    deadline before evaluating cost.

    Args:
        carrier_id: Carrier identifier (e.g. 'CARRIER-FX').
        origin: Origin port code (e.g. 'CNSHA').
        destination: Destination port code (e.g. 'USLAX').
        transit_mode: Transit mode (e.g. 'AIR', 'SEA', 'MULTIMODAL').

    Returns:
        Dict with keys:
            success (bool): True if the ETA was computed successfully.
            carrier_id (str): The queried carrier.
            eta_hours (float): Estimated transit time in hours.
            eta_days (float): Same value expressed in days (convenience).
            confidence (str): 'HIGH', 'MEDIUM', or 'LOW' — reflects
                              data freshness and route certainty.
            lane_supported (bool): Whether this carrier+lane combination exists.
            error (str|None): Error description if success is False.
    """
    with _time_tool("calculate_route_eta") as timing:
        logger.info(
            "tool_called",
            tool="calculate_route_eta",
            carrier_id=carrier_id,
            origin=origin,
            destination=destination,
            transit_mode=transit_mode,
        )

        _simulator.maybe_raise("calculate_route_eta")

        key = (origin.upper(), destination.upper())
        mode = transit_mode.upper()

        if key not in _LANE_ETA or mode not in _LANE_ETA[key]:
            return {
                "success": True,
                "carrier_id": carrier_id,
                "eta_hours": None,
                "eta_days": None,
                "confidence": "LOW",
                "lane_supported": False,
                "error": None,
                "latency_ms": timing.get("latency_ms", 0.0),
            }

        base_eta = _LANE_ETA[key][mode]
        multiplier = _CARRIER_ETA_MULTIPLIER.get(carrier_id.upper(), 1.0)

        # Deterministic jitter: reproducible across runs for same inputs
        jitter_seed = hash((carrier_id, origin, destination, mode)) % 10
        jitter_hours = (jitter_seed - 5) * 0.5  # –2.5 to +2.0 hours

        eta_hours = round(base_eta * multiplier + jitter_hours, 1)

        # Confidence based on whether we have carrier-specific data
        has_specific_data = carrier_id.upper() in _CARRIER_ETA_MULTIPLIER
        confidence = "HIGH" if has_specific_data else "MEDIUM"

        logger.info(
            "tool_success",
            tool="calculate_route_eta",
            carrier_id=carrier_id,
            eta_hours=eta_hours,
            confidence=confidence,
            latency_ms=round(timing.get("latency_ms", 0.0), 2),
        )

        return {
            "success": True,
            "carrier_id": carrier_id,
            "eta_hours": eta_hours,
            "eta_days": round(eta_hours / 24.0, 2),
            "confidence": confidence,
            "lane_supported": True,
            "error": None,
            "latency_ms": timing.get("latency_ms", 0.0),
        }


# ---------------------------------------------------------------------------
# Tool 4: estimate_reroute_cost
# ---------------------------------------------------------------------------


@tool
def estimate_reroute_cost(
    carrier_id: str,
    origin: str,
    destination: str,
    transit_mode: str,
    weight_kg: float,
    volume_cbm: float,
    cargo_type: str,
) -> dict[str, Any]:
    """
    Estimate the total cost for rerouting a shipment via a specific carrier.

    Call this for every viable carrier option to support a cost-effective
    decision.  The quote includes the base rate, weight surcharge, volume
    surcharge, and a cargo-type premium for specialised handling.

    Args:
        carrier_id: Carrier identifier (e.g. 'CARRIER-FX').
        origin: Origin port code (e.g. 'CNSHA').
        destination: Destination port code (e.g. 'USLAX').
        transit_mode: Transit mode for pricing context (e.g. 'AIR').
        weight_kg: Total shipment weight in kilograms.
        volume_cbm: Total shipment volume in cubic metres.
        cargo_type: Cargo type for premium calculation (e.g. 'HIGH_VALUE').

    Returns:
        Dict with keys:
            success (bool): True if cost was estimated successfully.
            carrier_id (str): The queried carrier.
            base_cost_usd (float): Fixed base rate for the lane.
            weight_cost_usd (float): Per-kg component.
            volume_cost_usd (float): Per-cbm component.
            cargo_premium_pct (float): Cargo-type premium percentage (e.g. 20.0).
            total_cost_usd (float): Full quoted cost including all surcharges.
            currency (str): Always 'USD'.
            quote_valid_hours (int): How long this quote is valid.
            error (str|None): Error description if success is False.
    """
    with _time_tool("estimate_reroute_cost") as timing:
        logger.info(
            "tool_called",
            tool="estimate_reroute_cost",
            carrier_id=carrier_id,
            weight_kg=weight_kg,
            volume_cbm=volume_cbm,
            cargo_type=cargo_type,
        )

        _simulator.maybe_raise("estimate_reroute_cost")

        pricing = _CARRIER_PRICING.get(carrier_id.upper())
        if not pricing:
            return {
                "success": False,
                "carrier_id": carrier_id,
                "total_cost_usd": None,
                "error": f"No pricing data found for carrier '{carrier_id}'.",
                "latency_ms": timing.get("latency_ms", 0.0),
            }

        cargo_upper = cargo_type.upper()
        premium_multiplier = _CARGO_PREMIUM.get(cargo_upper, 1.0)
        cargo_premium_pct = round((premium_multiplier - 1.0) * 100, 1)

        base = pricing["base"]
        weight_cost = weight_kg * pricing["per_kg"]
        volume_cost = volume_cbm * pricing["per_cbm"]
        subtotal = base + weight_cost + volume_cost
        total = round(subtotal * premium_multiplier, 2)

        logger.info(
            "tool_success",
            tool="estimate_reroute_cost",
            carrier_id=carrier_id,
            total_cost_usd=total,
            latency_ms=round(timing.get("latency_ms", 0.0), 2),
        )

        return {
            "success": True,
            "carrier_id": carrier_id,
            "base_cost_usd": round(base, 2),
            "weight_cost_usd": round(weight_cost, 2),
            "volume_cost_usd": round(volume_cost, 2),
            "cargo_premium_pct": cargo_premium_pct,
            "total_cost_usd": total,
            "currency": "USD",
            "quote_valid_hours": 4,
            "error": None,
            "latency_ms": timing.get("latency_ms", 0.0),
        }


# ---------------------------------------------------------------------------
# Tool 5: validate_route_constraints
# ---------------------------------------------------------------------------


@tool
def validate_route_constraints(
    carrier_id: str,
    route_id: str,
    cargo_type: str,
    weight_kg: float,
    volume_cbm: float,
    deadline_utc: str,
    eta_hours: float,
    max_cost_delta_usd: float,
    estimated_cost_usd: float,
    original_cost_usd: float,
) -> dict[str, Any]:
    """
    Validate that a proposed reroute satisfies all business and physical
    constraints before committing to the Decision Agent's selection.

    Always call this before finalising a route selection.  Constraints
    are evaluated independently and the result includes a breakdown of
    which constraints passed and which failed — this is used by the
    Decision Agent to justify its final recommendation.

    Args:
        carrier_id: Carrier to validate (e.g. 'CARRIER-FX').
        route_id: Route identifier (e.g. 'CNSHA-USLAX-FX-AIR-01').
        cargo_type: Cargo type to check against carrier certification.
        weight_kg: Shipment weight to check against carrier maximum.
        volume_cbm: Shipment volume to check against carrier maximum.
        deadline_utc: ISO 8601 UTC deadline string (e.g. '2024-07-20T18:00:00+00:00').
        eta_hours: Estimated transit time from calculate_route_eta.
        max_cost_delta_usd: Maximum acceptable cost increase vs. original.
        estimated_cost_usd: Total cost from estimate_reroute_cost.
        original_cost_usd: Original shipment cost (basis for delta calculation).

    Returns:
        Dict with keys:
            valid (bool): True only if ALL constraints are satisfied.
            constraints (dict): Per-constraint pass/fail breakdown.
            violations (list[str]): Human-readable list of failed constraints.
            recommendation (str): 'APPROVE', 'REJECT', or 'ESCALATE'.
            error (str|None): Error description if validation itself failed.
    """
    with _time_tool("validate_route_constraints") as timing:
        logger.info(
            "tool_called",
            tool="validate_route_constraints",
            carrier_id=carrier_id,
            route_id=route_id,
            cargo_type=cargo_type,
        )

        _simulator.maybe_raise("validate_route_constraints")

        from datetime import datetime, timezone

        violations: list[str] = []
        constraints: dict[str, bool] = {}

        # ------------------------------------------------------------------
        # Constraint 1: Deadline feasibility
        # ------------------------------------------------------------------
        try:
            deadline = datetime.fromisoformat(deadline_utc)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            available_hours = (deadline - now).total_seconds() / 3600.0
            # Account for 4 hours booking lead time
            deadline_ok = (eta_hours + 4.0) <= available_hours
        except Exception:
            deadline_ok = False

        constraints["deadline_feasible"] = deadline_ok
        if not deadline_ok:
            violations.append(
                f"ETA ({eta_hours}h + 4h booking) exceeds deadline "
                f"(available: {available_hours:.1f}h)."
            )

        # ------------------------------------------------------------------
        # Constraint 2: Cargo type certification
        # ------------------------------------------------------------------
        from tools.carrier_tools import _CARRIER_DB

        carrier = _CARRIER_DB.get(carrier_id.upper(), {})
        supported = carrier.get("supported_cargo", [])
        cargo_ok = cargo_type.upper() in supported
        constraints["cargo_type_certified"] = cargo_ok
        if not cargo_ok:
            violations.append(
                f"Carrier '{carrier_id}' is not certified for cargo type "
                f"'{cargo_type}'. Certified types: {supported}."
            )

        # ------------------------------------------------------------------
        # Constraint 3: Weight limit
        # ------------------------------------------------------------------
        max_weight = carrier.get("max_weight_kg", float("inf"))
        weight_ok = weight_kg <= max_weight
        constraints["weight_within_limit"] = weight_ok
        if not weight_ok:
            violations.append(
                f"Weight {weight_kg}kg exceeds carrier max {max_weight}kg."
            )

        # ------------------------------------------------------------------
        # Constraint 4: Volume limit
        # ------------------------------------------------------------------
        max_volume = carrier.get("max_volume_cbm", float("inf"))
        volume_ok = volume_cbm <= max_volume
        constraints["volume_within_limit"] = volume_ok
        if not volume_ok:
            violations.append(
                f"Volume {volume_cbm}cbm exceeds carrier max {max_volume}cbm."
            )

        # ------------------------------------------------------------------
        # Constraint 5: Cost delta within budget
        # ------------------------------------------------------------------
        cost_delta = estimated_cost_usd - original_cost_usd
        cost_ok = cost_delta <= max_cost_delta_usd
        constraints["cost_within_budget"] = cost_ok
        if not cost_ok:
            violations.append(
                f"Cost delta ${cost_delta:,.0f} exceeds max budget "
                f"${max_cost_delta_usd:,.0f}."
            )

        all_valid = len(violations) == 0

        # ------------------------------------------------------------------
        # Recommendation logic
        # ------------------------------------------------------------------
        if all_valid:
            recommendation = "APPROVE"
        elif len(violations) == 1 and not deadline_ok and cost_ok and cargo_ok:
            # Only deadline is a problem — could escalate for rush decision
            recommendation = "ESCALATE"
        else:
            recommendation = "REJECT"

        logger.info(
            "tool_success",
            tool="validate_route_constraints",
            carrier_id=carrier_id,
            valid=all_valid,
            violations=len(violations),
            recommendation=recommendation,
            latency_ms=round(timing.get("latency_ms", 0.0), 2),
        )

        return {
            "valid": all_valid,
            "constraints": constraints,
            "violations": violations,
            "recommendation": recommendation,
            "cost_delta_usd": round(cost_delta, 2),
            "error": None,
            "latency_ms": timing.get("latency_ms", 0.0),
        }
