"""Developer sidebar showing the NLP analysis and lead score of the conversation.

Everything shown comes from the validated ``LeadAnalysis`` and ``LeadScore``
stored in ``st.session_state.analysis``. When there is no result, an empty state
or the real error is shown instead. Nothing here is generated.
"""

from html import escape

import streamlit as st

from ai.nlp_service import FIELD_LABELS, LIST_FIELDS, TEXT_FIELDS
from ai.schemas import Entities, LeadAnalysis
from core.config import Settings

TOPIC_LABELS = {
    "confirm_conflict": "confirming a conflicting detail", "issue_details": "issue details",
    "requirement": "requirement", "features": "features", "budget": "budget", "timeline": "timeline",
    "company": "company name", "name": "name", "contact": "email", "decision": "decision process",
}
from scoring.models import LeadScore

_PURCHASE_CLASS = {"high": "high", "medium": "medium", "low": "low", "unknown": "unknown"}
_REQUIRED_FIELD_COUNT = 7  # see nlp_service.missing_information


def _heading(text: str) -> None:
    st.markdown(f'<div class="panel-heading">{text}</div>', unsafe_allow_html=True)


def _empty(text: str) -> None:
    st.markdown(f'<div class="empty-state">{escape(text)}</div>', unsafe_allow_html=True)


def _pct(confidence: float) -> str:
    return f"{confidence:.0%}"


def _entity_rows(entities: Entities) -> list[tuple[str, str | None, float]]:
    rows = []
    order = list(FIELD_LABELS)
    for field in order:
        if field in TEXT_FIELDS:
            f = getattr(entities, field)
            rows.append((FIELD_LABELS[field], f.value, f.confidence))
        elif field in LIST_FIELDS:
            f = getattr(entities, field)
            rows.append((FIELD_LABELS[field], "; ".join(f.values) or None, f.confidence))
        elif field == "budget":
            b = entities.budget
            rows.append((FIELD_LABELS[field], b.display(), b.confidence))
        elif field == "decision_authority":
            d = entities.decision_authority
            rows.append((FIELD_LABELS[field], None if d.value == "unknown" else d.value, d.confidence))
    return rows


def render_entity_table(entities: Entities) -> None:
    rows = []
    for label, value, confidence in _entity_rows(entities):
        if value is None:
            rows.append(
                f'<tr class="missing"><td>{label}</td><td>—</td><td></td></tr>'
            )
        else:
            rows.append(
                f"<tr><td>{label}</td><td>{escape(value)}</td>"
                f'<td class="conf">{_pct(confidence)}</td></tr>'
            )
    st.markdown(
        '<table class="entity-table"><thead><tr><th>Field</th><th>Value</th>'
        f'<th class="conf">Conf.</th></tr></thead><tbody>{"".join(rows)}</tbody></table>',
        unsafe_allow_html=True,
    )


def _render_intent(result: LeadAnalysis) -> None:
    intent = result.intent
    st.markdown(f"**{intent.category}**")
    st.progress(intent.confidence, text=f"Confidence {_pct(intent.confidence)}")
    if intent.secondary_categories:
        st.caption("Also related: " + ", ".join(intent.secondary_categories))
    if intent.rationale:
        st.caption(intent.rationale)

    purchase = result.purchase_intent
    st.markdown(
        f'Purchase intent <span class="level-badge {_PURCHASE_CLASS[purchase.level]}">'
        f"{purchase.level}</span> <small>({_pct(purchase.confidence)})</small>",
        unsafe_allow_html=True,
    )
    if purchase.rationale:
        st.caption(purchase.rationale)


def _render_profile(result: LeadAnalysis) -> None:
    known = _REQUIRED_FIELD_COUNT - len(result.missing_information)
    st.progress(known / _REQUIRED_FIELD_COUNT, text=f"Qualification data: {known}/{_REQUIRED_FIELD_COUNT} fields")
    if result.missing_information:
        chips = "".join(f'<span class="chip">{escape(m)}</span>' for m in result.missing_information)
        st.markdown(f"<div><small>Missing:</small><br>{chips}</div>", unsafe_allow_html=True)
    for conflict in result.conflicts:
        st.warning(
            f"**Conflict, {conflict.field}:** kept '{conflict.previous}', "
            f"customer also said '{conflict.new}'. Confirm with the customer.",
            icon="⚠️",
        )
    for correction in result.corrections:
        st.info(f"Corrected: {correction}", icon="✏️")
    for warning in result.warnings:
        st.caption(f"⚠️ {warning}")


def _render_score(score: LeadScore) -> None:
    st.markdown(
        f'''<div class="score-card {score.category.lower()}">
            <div class="score-value">{score.total}<span>/100</span></div>
            <div class="score-category">{score.category}</div>
        </div>''',
        unsafe_allow_html=True,
    )
    rows = []
    for c in score.criteria:
        pct = round(100 * c.earned / c.maximum)
        rows.append(
            f'''<div class="criterion">
                <div class="criterion-head"><span>{c.label}</span>
                <span><span class="evidence {c.evidence}">{c.evidence}</span> <b>{c.earned}</b>/{c.maximum}</span></div>
                <div class="bar"><div style="width:{pct}%"></div></div>
                <div class="criterion-reason">{escape(c.reason)}</div>
            </div>'''
        )
    st.markdown("".join(rows), unsafe_allow_html=True)
    if score.missing_information:
        chips = "".join(f'<span class="chip">{escape(m)}</span>' for m in score.missing_information)
        st.markdown(f"<div><small>To improve qualification, ask for:</small><br>{chips}</div>", unsafe_allow_html=True)


_ACTION_ICONS = {"created": "🆕", "updated": "🔄", "merged": "🔗", "unchanged": "✔️", "reopened": "📂"}


def _render_crm_record(persist: dict | None, lead_id: int | None, activities: list[dict]) -> None:
    if persist is None and lead_id is None:
        _empty("Not saved yet")
        return
    if persist:
        icon = _ACTION_ICONS.get(persist["action"], "💾")
        if persist["action"] == "created":
            st.success(persist["summary"], icon=icon)
        elif persist["action"] in ("updated", "merged"):
            st.info(persist["summary"], icon=icon)
        else:
            st.caption(f"{icon} {persist['summary']}")
        for note in persist.get("notes", []):
            st.warning(note, icon="⚠️")
    if activities:
        rows = "".join(
            f"<li><b>{escape(a['activity_type'].replace('_', ' '))}</b>: {escape(a['description'])}</li>"
            for a in activities
        )
        st.markdown(f'<ul class="activity-list">{rows}</ul>', unsafe_allow_html=True)


def _render_reply_plan(reply: dict | None) -> None:
    if not reply:
        _empty("No reply generated yet")
        return
    asked = ", ".join(TOPIC_LABELS.get(t, t) for t in reply["asked"]) or "nothing (all key details known)"
    st.caption(f"Follow-up asked about: {asked}")
    if reply["regenerated"]:
        st.caption("⚠️ First draft was rejected by the safety checks and regenerated.")


def render_sidebar(settings: Settings, activities: list[dict] | None = None) -> bool:
    """Render the developer panel. Returns True when Re-run analysis is clicked."""
    state = st.session_state.analysis
    result: LeadAnalysis | None = state["result"]
    score: LeadScore | None = state["score"]
    error: str | None = state["error"]
    has_user_messages = any(m["role"] == "user" for m in st.session_state.messages)

    if result:
        empty_text = ""
    elif error:
        empty_text = "No analysis available: the last request failed"
    elif has_user_messages:
        empty_text = "Analysis pending"
    else:
        empty_text = "Waiting for a message"

    with st.sidebar:
        st.markdown("### 🛠️ Developer Panel")
        st.caption("Live NLP analysis of the current conversation.")

        if error:
            st.error(error, icon="⚠️")
            if result and state.get("failed_stage") != "reply":
                st.caption("Showing the last successful analysis (stale).")

        _heading("CRM record")
        _render_crm_record(st.session_state.persist, st.session_state.lead_id, activities or [])

        _heading("Lead qualification & score")
        if score:
            _render_score(score)
        else:
            _empty(empty_text)

        _heading("Assistant reply")
        _render_reply_plan(state.get("reply"))

        _heading("Detected intent")
        if result:
            _render_intent(result)
        else:
            _empty(empty_text)

        _heading("Extracted entities")
        if result:
            render_entity_table(result.entities)
        else:
            _empty(empty_text)

        _heading("Current lead profile")
        if result:
            _render_profile(result)
        else:
            _empty(empty_text)

        st.divider()
        rerun = st.button(
            "Re-run analysis",
            icon="🔄",
            width="stretch",
            disabled=not has_user_messages,
        )
        if result:
            st.caption(f"Analysed {result.analysed_messages} customer message(s) with `{result.model}`")
        st.caption(
            f"Model: `{settings.groq_model or 'not set'}` · "
            f"API key {'set' if settings.groq_api_key else 'not set'}"
        )
    return rerun
