"""
tools/carrier_tools.py — Carrier Availability & Capacity Tools
===============================================================

Provides two LangChain tools used by the Research Agent to discover
alternative carriers during a logistics disruption:

    list_available_carriers
        Queries the carrier database for all carriers serving a given
        origin→destination lane that support the specified cargo type
        and transit mode.  Returns a ranked list of viable options.

    lookup_carrier_capacity
        Checks whether a specific carrier can accommodate the shipment
        dimensions (weight + volume) on a given lane.  Returns current
        capacity availability and booking lead time.

Trajectory Evaluation Contribution
------------------------------------
These tools are the Research Agent's primary information sources.
The evaluator checks:
  - Were they called with correct arguments (right carrier IDs, lane)?
  - Did the agent retry when a tool returned an error?
  - Did the agent query enough carriers before handing off to Decision?

The ``FailureSimulator`` deliberately fails specific carriers on the
first attempt, creating measurable error-recovery events.

Simulated Data
--------------
All carrier and lane data is sourced from a realistic static database
(``_CARRIER_DB``, ``_LANE_DB``).  This produces deterministic results
across evaluation runs — essential for comparing GPT-4o vs Qwen3 on
identical inputs.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Generator

from langchain_core.tools import tool

from utils.logger import get_logger
from utils.retry import make_retry_decorator
from tools._simulation import FailureSimulator, SimulatedToolError

logger = get_logger(__name__)
_simulator = FailureSimulator()

# ---------------------------------------------------------------------------
# Static carrier database (realistic, deterministic)
# ---------------------------------------------------------------------------

_CARRIER_DB: dict[str, dict[str, Any]] = {
    "CARRIER-FX": {
        "name": "FedEx International Priority",
        "modes": ["AIR"],
        "supported_cargo": ["STANDARD", "FRAGILE", "HIGH_VALUE"],
        "reliability_score": 0.97,
        "cost_per_kg_usd": 12.50,
        "base_cost_usd": 800.0,
        "max_weight_kg": 10_000.0,
        "max_volume_cbm": 68.0,
        "contact": "ops@fedex-intl.example.com",
    },
    "CARRIER-DHL": {
        "name": "DHL Express Worldwide",
        "modes": ["AIR"],
        "supported_cargo": ["STANDARD", "FRAGILE", "HIGH_VALUE", "HAZMAT"],
        "reliability_score": 0.96,
        "cost_per_kg_usd": 13.80,
        "base_cost_usd": 950.0,
        "max_weight_kg": 8_000.0,
        "max_volume_cbm": 55.0,
        "contact": "enterprise@dhl.example.com",
    },
    "CARRIER-UPS": {
        "name": "UPS Global Freight",
        "modes": ["AIR", "MULTIMODAL"],
        "supported_cargo": ["STANDARD", "FRAGILE", "HIGH_VALUE"],
        "reliability_score": 0.94,
        "cost_per_kg_usd": 11.20,
        "base_cost_usd": 700.0,
        "max_weight_kg": 15_000.0,
        "max_volume_cbm": 100.0,
        "contact": "freight@ups-global.example.com",
    },
    "CARRIER-MSC": {
        "name": "Mediterranean Shipping Co.",
        "modes": ["SEA"],
        "supported_cargo": ["STANDARD", "HAZMAT", "HIGH_VALUE"],
        "reliability_score": 0.88,
        "cost_per_kg_usd": 1.20,
        "base_cost_usd": 2_200.0,
        "max_weight_kg": 50_000.0,
        "max_volume_cbm": 800.0,
        "contact": "bookings@msc-freight.example.com",
    },
    "CARRIER-CMA": {
        "name": "CMA CGM Group",
        "modes": ["SEA", "MULTIMODAL"],
        "supported_cargo": ["STANDARD", "HAZMAT", "PERISHABLE"],
        "reliability_score": 0.87,
        "cost_per_kg_usd": 1.10,
        "base_cost_usd": 2_000.0,
        "max_weight_kg": 45_000.0,
        "max_volume_cbm": 750.0,
        "contact": "cargo@cma-cgm.example.com",
    },
    "CARRIER-EVER": {
        "name": "Evergreen Marine Corporation",
        "modes": ["SEA"],
        "supported_cargo": ["STANDARD", "FRAGILE"],
        "reliability_score": 0.85,
        "cost_per_kg_usd": 0.95,
        "base_cost_usd": 1_800.0,
        "max_weight_kg": 40_000.0,
        "max_volume_cbm": 680.0,
        "contact": "bookings@evergreen-marine.example.com",
    },
    "CARRIER-DB": {
        "name": "DB Schenker Logistics",
        "modes": ["MULTIMODAL", "RAIL"],
        "supported_cargo": ["STANDARD", "FRAGILE", "HIGH_VALUE", "HAZMAT"],
        "reliability_score": 0.93,
        "cost_per_kg_usd": 5.80,
        "base_cost_usd": 1_200.0,
        "max_weight_kg": 20_000.0,
        "max_volume_cbm": 200.0,
        "contact": "global@dbschenker.example.com",
    },
    "CARRIER-KN": {
        "name": "Kuehne+Nagel International",
        "modes": ["MULTIMODAL", "AIR", "SEA"],
        "supported_cargo": ["STANDARD", "FRAGILE", "HIGH_VALUE", "PERISHABLE"],
        "reliability_score": 0.92,
        "cost_per_kg_usd": 7.40,
        "base_cost_usd": 1_500.0,
        "max_weight_kg": 25_000.0,
        "max_volume_cbm": 300.0,
        "contact": "sea-air@kuehne-nagel.example.com",
    },
}

# ---------------------------------------------------------------------------
# Static lane database: which carriers serve which lane by mode
# ---------------------------------------------------------------------------

_LANE_DB: dict[tuple[str, str], dict[str, dict[str, Any]]] = {
    ("CNSHA", "USLAX"): {
        "AIR": {
            "carriers": ["CARRIER-FX", "CARRIER-DHL", "CARRIER-UPS"],
            "base_eta_hours": 36.0,
            "eta_jitter_hours": 4.0,
        },
        "SEA": {
            "carriers": ["CARRIER-MSC", "CARRIER-CMA", "CARRIER-EVER"],
            "base_eta_hours": 480.0,
            "eta_jitter_hours": 24.0,
        },
        "MULTIMODAL": {
            "carriers": ["CARRIER-KN", "CARRIER-DB"],
            "base_eta_hours": 168.0,
            "eta_jitter_hours": 12.0,
        },
    },
    ("CNSHA", "USNYC"): {
        "AIR": {
            "carriers": ["CARRIER-DHL", "CARRIER-FX"],
            "base_eta_hours": 42.0,
            "eta_jitter_hours": 5.0,
        },
        "SEA": {
            "carriers": ["CARRIER-MSC", "CARRIER-CMA"],
            "base_eta_hours": 576.0,
            "eta_jitter_hours": 36.0,
        },
        "MULTIMODAL": {
            "carriers": ["CARRIER-KN"],
            "base_eta_hours": 240.0,
            "eta_jitter_hours": 18.0,
        },
    },
    ("CNSHA", "DEHAM"): {
        "AIR": {
            "carriers": ["CARRIER-DHL", "CARRIER-UPS"],
            "base_eta_hours": 28.0,
            "eta_jitter_hours": 3.0,
        },
        "SEA": {
            "carriers": ["CARRIER-CMA", "CARRIER-EVER"],
            "base_eta_hours": 432.0,
            "eta_jitter_hours": 24.0,
        },
        "MULTIMODAL": {
            "carriers": ["CARRIER-DB", "CARRIER-KN"],
            "base_eta_hours": 144.0,
            "eta_jitter_hours": 8.0,
        },
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@contextmanager
def _time_tool(tool_name: str) -> Generator[dict[str, float], None, None]:
    """
    Context manager that measures tool execution latency.

    Yields a mutable dict; on exit, sets ``result["latency_ms"]``
    to the wall-clock duration in milliseconds.

    Usage::

        with _time_tool("carrier_lookup") as timing:
            result = _do_work()
        latency = timing["latency_ms"]
    """
    start = time.perf_counter()
    timing: dict[str, float] = {}
    try:
        yield timing
    finally:
        timing["latency_ms"] = (time.perf_counter() - start) * 1000.0


def _get_lane_info(
    origin: str, destination: str
) -> dict[str, dict[str, Any]] | None:
    """Return lane info for an origin→destination pair, or None if not found."""
    key = (origin.upper(), destination.upper())
    return _LANE_DB.get(key)


def _carrier_score(
    carrier_id: str,
    weight_kg: float,
    volume_cbm: float,
) -> float:
    """
    Compute a deterministic availability score for ranking.

    Score is penalised when shipment weight or volume approaches
    the carrier's maximum — carriers near capacity are ranked lower.
    """
    db = _CARRIER_DB.get(carrier_id, {})
    max_w = db.get("max_weight_kg", 1.0)
    max_v = db.get("max_volume_cbm", 1.0)
    weight_util = min(weight_kg / max_w, 1.0)
    volume_util = min(volume_cbm / max_v, 1.0)
    reliability = db.get("reliability_score", 0.5)
    return reliability - (weight_util * 0.1) - (volume_util * 0.1)


# ---------------------------------------------------------------------------
# Tool 1: list_available_carriers
# ---------------------------------------------------------------------------


@tool
def list_available_carriers(
    origin: str,
    destination: str,
    cargo_type: str,
    transit_mode: str,
    weight_kg: float,
    volume_cbm: float,
    exclude_carrier_ids: str = "",
) -> dict[str, Any]:
    """
    List all carriers available on a given lane that can handle the shipment.

    Use this tool when you need to discover alternative carriers after a
    shipment disruption.  Always call this before calling
    lookup_carrier_capacity — it returns the shortlist of viable options
    that you should investigate further.

    Args:
        origin: IATA/UN-LOCODE origin port or warehouse code (e.g. 'CNSHA').
        destination: IATA/UN-LOCODE destination code (e.g. 'USLAX').
        cargo_type: Cargo classification. One of: STANDARD, FRAGILE,
                    HAZMAT, PERISHABLE, HIGH_VALUE.
        transit_mode: Preferred mode. One of: AIR, SEA, RAIL, ROAD,
                      MULTIMODAL.  Pass 'ANY' to get all modes.
        weight_kg: Total shipment weight in kilograms.
        volume_cbm: Total shipment volume in cubic metres.
        exclude_carrier_ids: Comma-separated carrier IDs to exclude
                             (e.g. 'CARRIER-FX,CARRIER-MSC').
                             Leave empty to include all.

    Returns:
        Dict with keys:
            success (bool): True if the query succeeded.
            carriers (list): List of carrier dicts, each with carrier_id,
                carrier_name, mode, reliability_score, available.
            lane_found (bool): Whether this lane is in our network.
            error (str|None): Error description if success is False.
    """
    with _time_tool("list_available_carriers") as timing:
        logger.info(
            "tool_called",
            tool="list_available_carriers",
            origin=origin,
            destination=destination,
            cargo_type=cargo_type,
            transit_mode=transit_mode,
        )

        # Failure simulation (enables error-recovery testing)
        _simulator.maybe_raise("list_available_carriers")

        excluded = {
            c.strip().upper()
            for c in exclude_carrier_ids.split(",")
            if c.strip()
        }

        lane_info = _get_lane_info(origin, destination)
        if not lane_info:
            logger.warning(
                "lane_not_found",
                tool="list_available_carriers",
                origin=origin,
                destination=destination,
            )
            return {
                "success": True,
                "carriers": [],
                "lane_found": False,
                "total_found": 0,
                "error": None,
                "latency_ms": timing.get("latency_ms", 0.0),
            }

        # Collect carriers for the requested mode (or all modes)
        modes_to_check = (
            list(lane_info.keys())
            if transit_mode.upper() == "ANY"
            else [transit_mode.upper()]
        )

        results: list[dict[str, Any]] = []
        for mode in modes_to_check:
            if mode not in lane_info:
                continue
            for cid in lane_info[mode]["carriers"]:
                if cid in excluded:
                    continue
                carrier = _CARRIER_DB.get(cid)
                if not carrier:
                    continue
                # Check cargo type support
                cargo_supported = cargo_type.upper() in carrier["supported_cargo"]
                # Check weight/volume
                capacity_ok = (
                    weight_kg <= carrier["max_weight_kg"]
                    and volume_cbm <= carrier["max_volume_cbm"]
                )
                score = _carrier_score(cid, weight_kg, volume_cbm)

                results.append(
                    {
                        "carrier_id": cid,
                        "carrier_name": carrier["name"],
                        "transit_mode": mode,
                        "reliability_score": carrier["reliability_score"],
                        "cargo_type_supported": cargo_supported,
                        "within_capacity": capacity_ok,
                        "available": cargo_supported and capacity_ok,
                        "max_weight_kg": carrier["max_weight_kg"],
                        "max_volume_cbm": carrier["max_volume_cbm"],
                        "score": round(score, 4),
                        "contact": carrier["contact"],
                    }
                )

        # Sort by score descending so the best options appear first
        results.sort(key=lambda x: x["score"], reverse=True)

        logger.info(
            "tool_success",
            tool="list_available_carriers",
            total_found=len(results),
            viable=sum(1 for r in results if r["available"]),
            latency_ms=round(timing.get("latency_ms", 0.0), 2),
        )

        return {
            "success": True,
            "carriers": results,
            "lane_found": True,
            "total_found": len(results),
            "viable_count": sum(1 for r in results if r["available"]),
            "error": None,
            "latency_ms": timing.get("latency_ms", 0.0),
        }


# ---------------------------------------------------------------------------
# Tool 2: lookup_carrier_capacity
# ---------------------------------------------------------------------------


@tool
def lookup_carrier_capacity(
    carrier_id: str,
    origin: str,
    destination: str,
    transit_mode: str,
    weight_kg: float,
    volume_cbm: float,
) -> dict[str, Any]:
    """
    Check whether a specific carrier has confirmed capacity for a shipment
    on a given lane and transit mode.

    Use this after list_available_carriers to confirm that your chosen
    carrier can physically accommodate the shipment.  This tool also
    returns the earliest available booking slot and booking lead time.

    Args:
        carrier_id: Carrier identifier (e.g. 'CARRIER-FX', 'CARRIER-DHL').
        origin: Origin port code (e.g. 'CNSHA').
        destination: Destination port code (e.g. 'USLAX').
        transit_mode: Transit mode for this booking (e.g. 'AIR', 'SEA').
        weight_kg: Total shipment weight in kilograms.
        volume_cbm: Total shipment volume in cubic metres.

    Returns:
        Dict with keys:
            success (bool): True if the query succeeded.
            carrier_id (str): The queried carrier ID.
            carrier_name (str): Human-readable carrier name.
            capacity_available (bool): True if the carrier can take this shipment.
            utilisation_pct (float): Current capacity utilisation (0-100).
            earliest_slot_hours (float): Hours until the earliest available slot.
            booking_lead_time_hours (float): Minimum advance notice required.
            rejection_reason (str|None): Why capacity is unavailable, if applicable.
            error (str|None): Error description if success is False.
    """
    with _time_tool("lookup_carrier_capacity") as timing:
        logger.info(
            "tool_called",
            tool="lookup_carrier_capacity",
            carrier_id=carrier_id,
            transit_mode=transit_mode,
        )

        # Failure simulation
        _simulator.maybe_raise("lookup_carrier_capacity", context=carrier_id)

        carrier = _CARRIER_DB.get(carrier_id.upper())
        if not carrier:
            logger.warning(
                "carrier_not_found",
                tool="lookup_carrier_capacity",
                carrier_id=carrier_id,
            )
            return {
                "success": True,
                "carrier_id": carrier_id,
                "carrier_name": "Unknown",
                "capacity_available": False,
                "rejection_reason": f"Carrier '{carrier_id}' not found in network.",
                "error": None,
                "latency_ms": timing.get("latency_ms", 0.0),
            }

        lane_info = _get_lane_info(origin, destination)
        mode_upper = transit_mode.upper()

        # Check if carrier serves this lane+mode
        if (
            not lane_info
            or mode_upper not in lane_info
            or carrier_id.upper() not in lane_info[mode_upper]["carriers"]
        ):
            return {
                "success": True,
                "carrier_id": carrier_id,
                "carrier_name": carrier["name"],
                "capacity_available": False,
                "utilisation_pct": 0.0,
                "earliest_slot_hours": None,
                "booking_lead_time_hours": None,
                "rejection_reason": (
                    f"Carrier '{carrier_id}' does not serve "
                    f"{origin}→{destination} via {transit_mode}."
                ),
                "error": None,
                "latency_ms": timing.get("latency_ms", 0.0),
            }

        # Simulate realistic capacity utilisation
        # We use a deterministic seed based on carrier + lane for reproducibility
        _seed = hash((carrier_id, origin, destination, mode_upper)) % 100
        utilisation_pct = 40.0 + (_seed % 40)  # 40–80% — never fully booked

        within_weight = weight_kg <= carrier["max_weight_kg"]
        within_volume = volume_cbm <= carrier["max_volume_cbm"]
        capacity_available = within_weight and within_volume and utilisation_pct < 90.0

        rejection_reason: str | None = None
        if not within_weight:
            rejection_reason = (
                f"Shipment weight {weight_kg}kg exceeds carrier maximum "
                f"{carrier['max_weight_kg']}kg."
            )
        elif not within_volume:
            rejection_reason = (
                f"Shipment volume {volume_cbm}cbm exceeds carrier maximum "
                f"{carrier['max_volume_cbm']}cbm."
            )

        # Deterministic slot availability
        earliest_slot = 4.0 + (_seed % 12)  # 4–16 hours
        booking_lead = 2.0 + (_seed % 6)    # 2–8 hours

        logger.info(
            "tool_success",
            tool="lookup_carrier_capacity",
            carrier_id=carrier_id,
            capacity_available=capacity_available,
            utilisation_pct=round(utilisation_pct, 1),
            latency_ms=round(timing.get("latency_ms", 0.0), 2),
        )

        return {
            "success": True,
            "carrier_id": carrier_id,
            "carrier_name": carrier["name"],
            "capacity_available": capacity_available,
            "utilisation_pct": round(utilisation_pct, 1),
            "earliest_slot_hours": round(earliest_slot, 1),
            "booking_lead_time_hours": round(booking_lead, 1),
            "rejection_reason": rejection_reason,
            "supported_cargo_types": carrier["supported_cargo"],
            "contact": carrier["contact"],
            "error": None,
            "latency_ms": timing.get("latency_ms", 0.0),
        }
