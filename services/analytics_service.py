"""Dashboard data and rule-based insights.

Insights are produced by fixed rules over the aggregates below; no AI model
is involved. Each insight states a count from the database and never claims
a cause. Comparisons between groups are only made when every compared group
has at least ``MIN_GROUP_SIZE`` leads.
"""

import sqlite3
from dataclasses import dataclass, field

from database import analytics

MIN_GROUP_SIZE = 5     # leads per group before comparing conversion between groups
MIN_CLOSED = 5         # closed leads before any conversion comparison is made
GAP_THRESHOLD = 0.25   # report a missing field once at least 25% of leads lack it


@dataclass
class Dashboard:
    kpis: dict
    by_date: list[dict]
    by_service: list[dict]
    by_industry: list[dict]
    by_category: list[dict]
    funnel: list[dict]
    sources: list[dict]
    follow_ups: dict
    conversion_trend: list[dict]
    insights: list[str] = field(default_factory=list)


def _pct(part: int, whole: int) -> str:
    return f"{part / whole:.0%}"


def build_insights(k: dict, by_service: list[dict], sources: list[dict], follow_ups: dict,
                   gaps: dict, attention: dict) -> list[str]:
    total = k["total"]
    if total == 0:
        return []
    insights = []

    scored = total - k["unscored"]
    if scored:
        insights.append(f"{k['hot']} of {scored} scored leads ({_pct(k['hot'], scored)}) are HOT, "
                        f"{k['warm']} WARM and {k['cold']} COLD.")

    named = [s for s in by_service if s["value"] != analytics.NOT_STATED]
    if named:
        top = named[0]
        tied = [s["value"] for s in named if s["leads"] == top["leads"]]
        if len(tied) > 1:
            insights.append(f"Most requested services (tied, {top['leads']} lead{'s' if top['leads'] != 1 else ''} "
                            f"each): {', '.join(tied)}.")
        else:
            insights.append(f"Most requested service: {top['value']} ({top['leads']} of {total} leads, "
                            f"{_pct(top['leads'], total)}).")

    if attention["hot_still_new"]:
        insights.append(f"{attention['hot_still_new']} HOT lead(s) are still in New status (not yet moved "
                        "to Contacted).")
    if attention["hot_unassigned"]:
        insights.append(f"{attention['hot_unassigned']} open HOT lead(s) have no salesperson assigned.")

    if follow_ups["overdue"]:
        insights.append(f"{follow_ups['overdue']} follow-up(s) are overdue (due before today and still pending).")
    completed = follow_ups["done_on_time"] + follow_ups["done_late"]
    if completed:
        insights.append(f"{follow_ups['done_on_time']} of {completed} completed follow-ups "
                        f"({_pct(follow_ups['done_on_time'], completed)}) were done on or before the due date.")

    labels = {"no_budget": "budget", "no_timeline": "timeline", "no_company": "company name",
              "no_contact": "phone number or email"}
    worst = max(labels, key=lambda key: gaps[key])
    if gaps[worst] / total >= GAP_THRESHOLD:
        insights.append(f"{gaps[worst]} of {total} leads ({_pct(gaps[worst], total)}) have no "
                        f"{labels[worst]} recorded yet.")

    closed = k["converted"] + k["lost"]
    if closed < MIN_CLOSED:
        insights.append(f"{closed} lead(s) are closed (converted or lost), too few to compare conversion "
                        f"rates between sources (needs at least {MIN_CLOSED}).")
    else:
        insights.append(f"{k['converted']} of {closed} closed leads were converted ({_pct(k['converted'], closed)}).")
        eligible = [s for s in sources if s["source"] != analytics.NOT_STATED and s["leads"] >= MIN_GROUP_SIZE]
        if len(eligible) >= 2:
            best = max(eligible, key=lambda s: (s["converted"] / s["leads"], s["leads"]))
            insights.append(f"Among sources with at least {MIN_GROUP_SIZE} leads, {best['source']} has the highest "
                            f"conversion so far ({best['converted']} of {best['leads']}). This is an observed "
                            "difference, not evidence of cause.")
    return insights


def build_dashboard(conn: sqlite3.Connection, today: str) -> Dashboard:
    k = analytics.kpis(conn, today)
    by_service = analytics.leads_by(conn, "service_intent")
    sources = analytics.source_performance(conn)
    follow_ups = analytics.follow_up_performance(conn, today)
    return Dashboard(
        kpis=k,
        by_date=analytics.leads_by_date(conn),
        by_service=by_service,
        by_industry=analytics.leads_by(conn, "industry"),
        by_category=analytics.leads_by(conn, "category"),
        funnel=analytics.funnel(conn),
        sources=sources,
        follow_ups=follow_ups,
        conversion_trend=analytics.conversion_trend(conn),
        insights=build_insights(k, by_service, sources, follow_ups,
                                analytics.data_gaps(conn), analytics.attention_counts(conn)),
    )
