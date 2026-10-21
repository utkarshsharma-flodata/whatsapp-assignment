"""Admin CRM page: dashboard (KPIs, charts, insights) and lead management.

All data comes from SQLite through the service and repository layers; this
module contains no SQL.
"""

import sqlite3
from datetime import date, datetime
from html import escape

import pandas as pd
import streamlit as st

from core.config import Settings
from database import repository as repo
from database.analytics import FUNNEL_STAGES, NOT_STATED
from database.connection import connect
from services import admin_service
from services.admin_service import STAGE_LABELS, STAGES
from services.analytics_service import build_dashboard
from ui import charts
from ui.sidebar import render_entity_table

_CATEGORY_ORDER = ["HOT", "WARM", "COLD", NOT_STATED]


def _local(iso: str | None, fmt: str = "%d %b %Y %H:%M") -> str:
    return datetime.fromisoformat(iso).astimezone().strftime(fmt) if iso else "—"


def _pct(value: float | None) -> str:
    return f"{value:.0%}" if value is not None else "—"


def _flash(message: str) -> None:
    st.session_state.admin_flash = message


# ---------------------------------------------------------------- dashboard

def render_dashboard(settings: Settings) -> None:
    with connect(settings.database_path) as conn:
        d = build_dashboard(conn, date.today().isoformat())
    k = d.kpis

    if k["total"] == 0:
        st.info("No leads in the database yet. Leads appear here after customers chat in the simulator.", icon="📭")

    row1 = st.columns(5)
    row1[0].metric("Total leads", k["total"], border=True)
    row1[1].metric("New", k["new"], help="Leads whose status is New (not yet contacted).", border=True)
    row1[2].metric("Hot", k["hot"], border=True)
    row1[3].metric("Warm", k["warm"], border=True)
    row1[4].metric("Cold", k["cold"], border=True)
    row2 = st.columns(5)
    row2[0].metric("Converted", k["converted"], border=True)
    row2[1].metric("Lost", k["lost"], border=True)
    row2[2].metric("Pending follow-ups", k["pending_follow_ups"],
                   help=f"{k['overdue_follow_ups']} of them overdue.", border=True)
    row2[3].metric("Conversion rate", _pct(k["conversion_rate"]),
                   help="Converted ÷ total leads. Win rate (converted ÷ closed): " + _pct(k["win_rate"]),
                   border=True)
    row2[4].metric("Average score", k["avg_score"] if k["avg_score"] is not None else "—", border=True)
    if k["unscored"]:
        st.caption(f"{k['unscored']} lead(s) have no score yet.")

    with st.container(border=True):
        st.markdown("**💡 Insights**")
        st.caption("Rule-based analytics: fixed rules over the stored data. No AI is used, and no causes are claimed.")
        if d.insights:
            st.markdown("\n".join(f"- {escape(i)}" for i in d.insights))
        else:
            st.caption("No insights yet: there is no data to analyse.")

    left, right = st.columns(2)
    with left:
        charts.leads_over_time(d.by_date, "by_date")
    with right:
        charts.horizontal_bars(d.by_category, "value", "leads", "HOT / WARM / COLD distribution", "by_category",
                               sort=[c for c in _CATEGORY_ORDER if c in {r["value"] for r in d.by_category}])
    left, right = st.columns(2)
    with left:
        charts.horizontal_bars(d.by_service, "value", "leads", "Leads by service", "by_service")
    with right:
        charts.horizontal_bars(d.by_industry, "value", "leads", "Leads by industry", "by_industry")

    left, right = st.columns(2)
    with left:
        funnel = [{"stage": STAGE_LABELS[r["stage"]], "leads": r["leads"]} for r in d.funnel]
        charts.horizontal_bars(funnel, "stage", "leads", "Conversion funnel (leads that reached each stage)",
                               "funnel", sort=[STAGE_LABELS[s] for s in FUNNEL_STAGES])
    with right:
        charts.conversion_trend(d.conversion_trend, "trend")

    left, right = st.columns(2)
    with left:
        charts.horizontal_bars([{"source": r["source"], "leads": r["leads"]} for r in d.sources],
                               "source", "leads", "Lead sources", "sources")
        if d.sources:
            st.dataframe(
                pd.DataFrame(d.sources).assign(
                    conversion=lambda df: (df["converted"] / df["leads"]).map("{:.0%}".format)
                ).rename(columns={"source": "Source", "leads": "Leads", "converted": "Converted", "lost": "Lost",
                                  "avg_score": "Avg score", "conversion": "Conversion"}),
                hide_index=True, width="stretch",
            )
    with right:
        f = d.follow_ups
        rows = [
            {"state": "Upcoming", "follow_ups": f["upcoming"]},
            {"state": "Overdue", "follow_ups": f["overdue"]},
            {"state": "Done on time", "follow_ups": f["done_on_time"]},
            {"state": "Done late", "follow_ups": f["done_late"]},
            {"state": "Missed", "follow_ups": f["missed"]},
            {"state": "Cancelled", "follow_ups": f["cancelled"]},
        ]
        charts.horizontal_bars(rows, "state", "follow_ups", "Follow-up performance", "follow_ups",
                               sort=[r["state"] for r in rows], value_title="Follow-ups")


# ------------------------------------------------------------- lead list

def _lead_label(lead: dict) -> str:
    name = lead["customer_name"] or "Unknown customer"
    company = f" · {lead['company']}" if lead["company"] else ""
    score = f" · {lead['score']} {lead['category']}" if lead["score"] is not None else ""
    return f"#{lead['id']} · {name}{company}{score} · {STAGE_LABELS[lead['status']]}"


def render_leads(settings: Settings) -> None:
    with connect(settings.database_path) as conn:
        options = {c: repo.distinct_lead_values(conn, c) for c in ("category", "status", "service_intent", "industry")}

    text = st.text_input("Search", placeholder="Name, company, phone, email, service or requirement",
                         key="admin_search")
    cols = st.columns(4)
    filters = {
        "category": cols[0].multiselect("Category", options["category"], key="f_category"),
        "status": cols[1].multiselect("Status", options["status"], format_func=STAGE_LABELS.get, key="f_status"),
        "service_intent": cols[2].multiselect("Service", options["service_intent"], key="f_service"),
        "industry": cols[3].multiselect("Industry", options["industry"], key="f_industry"),
    }
    date_range = st.date_input("First contact between", value=(), key="f_dates", format="DD/MM/YYYY")
    created_from = date_range[0].isoformat() if len(date_range) >= 1 else None
    created_to = date_range[1].isoformat() if len(date_range) == 2 else None

    with connect(settings.database_path) as conn:
        leads = repo.search_leads(conn, text, filters, created_from, created_to)

    st.caption(f"{len(leads)} lead(s) match.")
    if not leads:
        st.info("No leads match these filters." if any([text, *filters.values(), date_range])
                else "No leads saved yet.", icon="📭")
        return
    st.dataframe(
        [{
            "ID": l["id"], "Name": l["customer_name"], "Company": l["company"], "Phone": l["phone"],
            "Email": l["email"], "Service": l["service_intent"], "Industry": l["industry"],
            "Score": l["score"], "Category": l["category"], "Status": STAGE_LABELS[l["status"]],
            "Salesperson": l["assigned_salesperson"], "Next follow-up": l["next_follow_up_date"],
            "First contact": _local(l["first_contact_date"], "%d %b %Y"),
            "Last contact": _local(l["last_contact_date"]),
        } for l in leads],
        hide_index=True, width="stretch",
    )

    labels = {_lead_label(l): l["id"] for l in leads}
    choice = st.selectbox("Open lead", list(labels), key="admin_lead")
    render_lead_detail(settings, labels[choice])


# ------------------------------------------------------------ lead detail

def render_lead_detail(settings: Settings, lead_id: int) -> None:
    with connect(settings.database_path) as conn:
        detail = admin_service.lead_detail(conn, lead_id)
    if detail is None:
        st.warning("This lead no longer exists.")
        return
    lead = detail["lead"]

    st.divider()
    st.subheader(f"#{lead['id']} · {lead['customer_name'] or 'Unknown customer'}"
                 + (f" · {lead['company']}" if lead["company"] else ""))
    top = st.columns(5)
    top[0].metric("Score", f"{lead['score']}/100" if lead["score"] is not None else "—", border=True)
    top[1].metric("Category", lead["category"] or "—", border=True)
    top[2].metric("Status", STAGE_LABELS[lead["status"]], border=True)
    top[3].metric("Salesperson", lead["assigned_salesperson"] or "—", border=True)
    top[4].metric("Next follow-up", lead["next_follow_up_date"] or "—", border=True)

    manage, profile, score, conversations, activity = st.tabs(
        ["Manage", "Profile", "Score", f"Conversations ({len(detail['conversations'])})", "Activity"])
    with manage:
        try:
            _render_manage(settings, detail)
        except (ValueError, sqlite3.Error) as exc:
            st.error(f"The change could not be saved: {exc}", icon="⚠️")
    with profile:
        _render_profile(detail)
    with score:
        _render_score(detail)
    with conversations:
        _render_conversations(detail)
    with activity:
        _render_activity(detail)


def _render_manage(settings: Settings, detail: dict) -> None:
    lead = detail["lead"]
    lead_id = lead["id"]
    left, right = st.columns(2)

    with left:
        with st.form(f"status_{lead_id}"):
            st.markdown("**Lifecycle status**")
            st.caption(" → ".join(STAGE_LABELS[s] for s in STAGES[:-2]) + " → Converted / Lost")
            new_status = st.selectbox("Status", STAGES, index=STAGES.index(lead["status"]),
                                      format_func=STAGE_LABELS.get)
            note = st.text_input("Note (optional)", placeholder="e.g. reason for losing the deal")
            if st.form_submit_button("Update status", type="primary"):
                with connect(settings.database_path) as conn:
                    changed = admin_service.change_status(conn, lead_id, new_status, note)
                _flash(f"Status changed to {STAGE_LABELS[new_status]}." if changed else "Status unchanged.")
                st.rerun()

        with st.form(f"assign_{lead_id}"):
            st.markdown("**Salesperson**")
            name = st.text_input("Assigned salesperson", value=lead["assigned_salesperson"] or "",
                                 placeholder="Leave empty to unassign")
            if st.form_submit_button("Save salesperson"):
                with connect(settings.database_path) as conn:
                    changed = admin_service.assign_salesperson(conn, lead_id, name)
                _flash("Salesperson saved." if changed else "Salesperson unchanged.")
                st.rerun()

        with st.form(f"note_{lead_id}", clear_on_submit=True):
            st.markdown("**Add note**")
            text = st.text_area("Note", placeholder="Internal note (not shown to the customer)")
            if st.form_submit_button("Add note"):
                if text.strip():
                    with connect(settings.database_path) as conn:
                        admin_service.add_note(conn, lead_id, text)
                    _flash("Note added.")
                    st.rerun()
                else:
                    st.warning("Write a note first.")

    with right:
        with st.form(f"follow_up_{lead_id}", clear_on_submit=True):
            st.markdown("**Schedule follow-up**")
            due = st.date_input("Due date", value=date.today(), min_value=date.today(), format="DD/MM/YYYY")
            action = st.text_input("Next action", placeholder="e.g. Send proposal, call to confirm budget")
            if st.form_submit_button("Schedule"):
                with connect(settings.database_path) as conn:
                    admin_service.schedule_follow_up(conn, lead_id, due, action)
                _flash(f"Follow-up scheduled for {due:%d %b %Y}.")
                st.rerun()

        pending = [f for f in detail["follow_ups"] if f["status"] == "pending"]
        st.markdown("**Pending follow-ups**")
        if not pending:
            st.caption("None.")
        for f in pending:
            overdue = f["due_date"] < date.today().isoformat()
            with st.form(f"complete_{f['id']}"):
                st.markdown(f"#{f['follow_up_count']} · due {f['due_date']}" + (" · ⚠️ overdue" if overdue else "")
                            + (f" · {escape(f['next_action'])}" if f["next_action"] else ""))
                outcome = st.selectbox("Outcome", admin_service.FOLLOW_UP_OUTCOMES,
                                       format_func=lambda o: {"done": "Done", "missed": "Missed",
                                                              "cancelled": "Cancelled"}[o])
                response = st.text_input("Customer response (optional)")
                if st.form_submit_button("Record outcome"):
                    with connect(settings.database_path) as conn:
                        admin_service.complete_follow_up(conn, f["id"], outcome, response)
                    _flash("Follow-up updated.")
                    st.rerun()

        st.markdown("**Close the lead**")
        reason = st.text_input("Reason / note", key=f"close_reason_{lead_id}",
                               placeholder="Optional, e.g. signed contract, chose competitor")
        won, lost = st.columns(2)
        if won.button("Mark Converted", icon="✅", key=f"won_{lead_id}", disabled=lead["status"] == "converted",
                      width="stretch"):
            with connect(settings.database_path) as conn:
                admin_service.mark_converted(conn, lead_id, reason)
            _flash("Lead marked Converted.")
            st.rerun()
        if lost.button("Mark Lost", icon="✖️", key=f"lost_{lead_id}", disabled=lead["status"] == "lost",
                       width="stretch"):
            with connect(settings.database_path) as conn:
                admin_service.mark_lost(conn, lead_id, reason)
            _flash("Lead marked Lost.")
            st.rerun()


def _render_profile(detail: dict) -> None:
    lead, profile = detail["lead"], detail["profile"]
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Extracted profile** (with extraction confidence)")
        if profile:
            render_entity_table(profile.entities)
            st.caption(f"Service intent: {profile.intent.category} ({profile.intent.confidence:.0%}) · "
                       f"Purchase intent: {profile.purchase_intent.level}")
            for conflict in profile.conflicts:
                st.warning(f"Conflict, {conflict.field}: '{conflict.previous}' vs '{conflict.new}'", icon="⚠️")
        else:
            st.caption("No extracted profile stored.")
    with right:
        st.markdown("**Record**")
        rows = [
            ("Phone", lead["phone"]), ("Email", lead["email"]), ("Lead source", lead["lead_source"]),
            ("Conversion", lead["conversion_status"]),
            ("First contact", _local(lead["first_contact_date"])),
            ("Last contact", _local(lead["last_contact_date"])),
            ("Status changed", _local(lead["status_changed_at"])), ("Closed", _local(lead["closed_at"])),
            ("Created", _local(lead["created_at"])), ("Updated", _local(lead["updated_at"])),
        ]
        st.dataframe(pd.DataFrame(rows, columns=["Field", "Value"]).fillna("—"), hide_index=True, width="stretch")


def _render_score(detail: dict) -> None:
    score = detail["score"]
    if score is None:
        st.caption("No score recorded yet.")
        return
    st.markdown(f"**Current score: {score.total}/100 · {score.category}**")
    st.dataframe(
        [{"Criterion": c.label, "Points": f"{c.earned}/{c.maximum}", "Evidence": c.evidence, "Reason": c.reason}
         for c in score.criteria],
        hide_index=True, width="stretch",
    )
    if score.missing_information:
        st.caption("Missing to improve qualification: " + ", ".join(score.missing_information))
    st.markdown(f"**Score history** ({len(detail['score_history'])} record(s))")
    charts.score_history(detail["score_history"], f"history_{detail['lead']['id']}")
    with st.expander("Score history table"):
        st.dataframe(
            [{"Scored at": _local(h["scored_at"]), "Score": h["total_score"], "Category": h["category"],
              "Explanation": h["explanation"]} for h in detail["score_history"]],
            hide_index=True, width="stretch",
        )


def _render_conversations(detail: dict) -> None:
    if not detail["conversations"]:
        st.caption("No conversations.")
    for i, conversation in enumerate(detail["conversations"]):
        title = (f"Conversation #{conversation['id']} · {conversation['channel']} · "
                 f"{conversation['message_count']} message(s) · started {_local(conversation['started_at'])}")
        with st.expander(title, expanded=i == 0):
            for m in conversation["messages"]:
                role = "user" if m["sender_role"] == "customer" else "assistant"
                with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
                    st.markdown(m["message_text"])
                    st.caption(_local(m["sent_at"]) + ("" if m["processing_status"] in ("processed", "skipped")
                                                       else f" · {m['processing_status']}"))


def _render_activity(detail: dict) -> None:
    st.markdown("**Status history**")
    st.dataframe(
        [{"Changed at": _local(h["changed_at"]),
          "From": STAGE_LABELS.get(h["from_status"], "—") if h["from_status"] else "—",
          "To": STAGE_LABELS.get(h["to_status"], h["to_status"]), "Note": h["note"]}
         for h in detail["status_history"]],
        hide_index=True, width="stretch",
    )
    st.markdown("**Follow-ups**")
    if detail["follow_ups"]:
        st.dataframe(
            [{"#": f["follow_up_count"], "Due": f["due_date"], "Status": f["status"], "Next action": f["next_action"],
              "Customer response": f["customer_response"], "Completed": _local(f["completed_at"])}
             for f in detail["follow_ups"]],
            hide_index=True, width="stretch",
        )
    else:
        st.caption("None scheduled.")
    st.markdown("**Activity log** (notes, assignments, updates)")
    st.dataframe(
        [{"When": _local(a["created_at"]), "Type": a["activity_type"].replace("_", " "),
          "Description": a["description"]} for a in detail["activities"]],
        hide_index=True, width="stretch",
    )


# ------------------------------------------------------------------- page

def render_admin_page(settings: Settings) -> None:
    # The chat page is capped at 960px for readability; the dashboard needs the full width.
    st.markdown("<style>.block-container { max-width: 1400px; }</style>", unsafe_allow_html=True)
    st.title("📊 Admin CRM")
    if message := st.session_state.pop("admin_flash", None):
        st.toast(message, icon="✅")
    dashboard, leads = st.tabs(["Dashboard", "Leads"])
    for tab, render in ((dashboard, render_dashboard), (leads, render_leads)):
        with tab:
            try:
                render(settings)
            except sqlite3.Error as exc:
                st.error(f"Database error: {exc}", icon="⚠️")
