"""Streamlit session-state helpers for the chat and the developer sidebar."""

from datetime import datetime

import streamlit as st


def empty_analysis() -> dict:
    """Analysis state shown in the sidebar.

    ``result`` holds the lead profile (``LeadAnalysis``) or None.
    ``error`` holds the message of the most recent failure (or None);
    when set, ``result`` is from an earlier message and is marked stale.
    ``score`` holds the ``LeadScore`` calculated from ``result``.
    """
    return {"result": None, "score": None, "error": None, "failed_stage": None, "reply": None,
            # False, or the work to run on the next rerun: "turn", "analysis" or "reply".
            "pending": False}


def _defaults() -> dict:
    return {
        "messages": [],
        "analysis": empty_analysis(),
        # Database identity of the open conversation (None until the first message).
        "conversation_id": None,
        # Lead the conversation belongs to; preset when reopening an existing lead.
        "lead_id": None,
        # Outcome of the last save, e.g. {"summary": "New lead #3 created", "action": "created"}.
        "persist": None,
    }


def init_session() -> None:
    for key, value in _defaults().items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_chat() -> None:
    for key, value in _defaults().items():
        st.session_state[key] = value
    st.session_state.pop("sender_phone", None)


def _local_time(iso_timestamp: str) -> str:
    return datetime.fromisoformat(iso_timestamp).astimezone().strftime("%d %b %H:%M")


def add_message(role: str, content: str, message_id: int | None = None, sent_at: str | None = None) -> None:
    st.session_state.messages.append(
        {
            "id": message_id,
            "role": role,
            "content": content,
            "time": _local_time(sent_at) if sent_at else datetime.now().strftime("%H:%M"),
        }
    )


def set_messages(rows: list[dict]) -> None:
    """Replace the chat history with message rows from the database."""
    st.session_state.messages = []
    for m in rows:
        role = "user" if m["sender_role"] == "customer" else "assistant"
        add_message(role, m["message_text"], message_id=m["id"], sent_at=m["sent_at"])


def open_conversation(loaded: dict) -> None:
    """Replace the session with a conversation loaded from the database."""
    reset_chat()
    st.session_state.conversation_id = loaded["conversation"]["id"]
    st.session_state.lead_id = loaded["lead"]["id"] if loaded["lead"] else None
    set_messages(loaded["messages"])
    st.session_state.analysis["result"] = loaded["profile"]
    st.session_state.analysis["score"] = loaded["score"]
    st.session_state.persist = {"summary": f"Reopened conversation #{loaded['conversation']['id']}",
                                "action": "reopened", "notes": []}


def start_for_lead(loaded: dict) -> None:
    """Start a new (unsaved) conversation for an existing lead."""
    reset_chat()
    st.session_state.lead_id = loaded["lead"]["id"]
    st.session_state.analysis["result"] = loaded["profile"]
    st.session_state.analysis["score"] = loaded["score"]
    st.session_state.persist = {"summary": f"New conversation for lead #{loaded['lead']['id']}",
                                "action": "reopened", "notes": []}
