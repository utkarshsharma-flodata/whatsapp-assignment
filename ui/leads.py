"""Lead browser: CRM table and controls to reopen a lead or conversation.

Receives plain data from the app; it never touches the database itself.
"""

from collections.abc import Callable
from datetime import datetime

import streamlit as st

NEW_CONVERSATION = "new"
UNLINKED = "unlinked"


def _date(iso: str | None) -> str:
    return datetime.fromisoformat(iso).astimezone().strftime("%d %b %Y %H:%M") if iso else "—"


def _lead_label(lead: dict) -> str:
    name = lead["customer_name"] or "Unknown customer"
    extra = " · ".join(x for x in (lead["company"], lead["phone"] or lead["email"]) if x)
    score = f"{lead['score']} {lead['category']}" if lead["score"] is not None else "not scored"
    return f"#{lead['id']} · {name}" + (f" · {extra}" if extra else "") + f" · {score}"


def _conversation_label(c: dict) -> str:
    return f"Conversation #{c['id']} · {c['channel']} · {c['message_count']} message(s) · last {_date(c['last_message_at'])}"


def render_lead_browser(
    leads: list[dict],
    unlinked_count: int,
    load_conversations: Callable[[int | None], list[dict]],
) -> dict | None:
    """Render the browser. Returns an action dict when Open is clicked:
    {"type": "open_conversation", "conversation_id": int} or
    {"type": "new_conversation", "lead_id": int}.
    """
    with st.expander(f"📂 Leads & conversations ({len(leads)} lead(s))"):
        if not leads and not unlinked_count:
            st.caption("No saved leads yet. Leads are saved after the first analysed message.")
            return None

        if leads:
            st.dataframe(
                [
                    {
                        "ID": l["id"],
                        "Name": l["customer_name"],
                        "Company": l["company"],
                        "Phone": l["phone"],
                        "Email": l["email"],
                        "Service": l["service_intent"],
                        "Score": l["score"],
                        "Category": l["category"],
                        "Status": l["status"],
                        "Conversations": l["conversation_count"],
                        "Last contact": _date(l["last_contact_date"]),
                    }
                    for l in leads
                ],
                hide_index=True,
                width="stretch",
            )

        # Options are display labels (unique because they include the ID), mapped back to values.
        lead_options = {_lead_label(l): l["id"] for l in leads}
        if unlinked_count:
            lead_options[f"Unidentified conversations ({unlinked_count})"] = UNLINKED
        lead_choice = lead_options[st.selectbox("Lead", list(lead_options), key="browser_lead")]

        lead_id = None if lead_choice == UNLINKED else lead_choice
        conv_options = {"➕ Start a new conversation with this lead": NEW_CONVERSATION} if lead_id is not None else {}
        conv_options |= {_conversation_label(c): c["id"] for c in load_conversations(lead_id)}
        conv_label = st.selectbox("Conversation", list(conv_options), key="browser_conv")
        conv_choice = conv_options.get(conv_label)

        if st.button("Open", type="primary", disabled=conv_choice is None):
            if conv_choice == NEW_CONVERSATION:
                return {"type": "new_conversation", "lead_id": lead_id}
            return {"type": "open_conversation", "conversation_id": conv_choice}
    return None
