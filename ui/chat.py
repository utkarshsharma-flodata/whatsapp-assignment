"""Main chat area: header, conversation history and message composer."""

from html import escape

import streamlit as st

from core.config import Settings

_AVATARS = {"user": "🧑", "assistant": "🤖"}


def render_header(settings: Settings) -> bool:
    """Render the page header. Returns True when New Chat is clicked."""
    if settings.ai_configured:
        status = f'<span class="status-pill ok">NLP: {escape(settings.groq_model)}</span>'
    else:
        status = '<span class="status-pill">NLP not configured: set GROQ_API_KEY and GROQ_MODEL</span>'
    title_col, action_col = st.columns([5, 1], vertical_alignment="center")
    with title_col:
        st.markdown(
            f"""
            <p class="app-title">💬 {escape(settings.app_title)}</p>
            <p class="app-subtitle">WhatsApp lead conversation simulator</p>
            {status}
            """,
            unsafe_allow_html=True,
        )
    with action_col:
        return st.button("New Chat", icon="➕", width="stretch")


def render_chat_history(messages: list[dict]) -> None:
    with st.container(height=480, border=True):
        if not messages:
            st.markdown(
                """
                <div class="chat-empty">
                    <h3>Start a conversation</h3>
                    <p>Type a message below as if you were a customer
                    contacting the business on WhatsApp.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            return

        for message in messages:
            with st.chat_message(message["role"], avatar=_AVATARS.get(message["role"])):
                st.markdown(message["content"])
                st.markdown(
                    f'<div class="msg-time">{message["time"]}</div>',
                    unsafe_allow_html=True,
                )


def render_conversation_status(conversation_id: int | None, lead_id: int | None) -> None:
    """One-line description of what the open chat is saved as."""
    if conversation_id:
        lead = f"lead #{lead_id}" if lead_id else "lead not identified yet"
        st.caption(f"💾 Conversation #{conversation_id} · {lead} · saved to the database")
    elif lead_id:
        st.caption(f"💾 New conversation for lead #{lead_id} · saved when you send the first message")
    else:
        st.caption("💾 New conversation · saved when you send the first message")


def render_sender_phone_input() -> None:
    """Optional simulated WhatsApp sender number, editable until the first message."""
    st.text_input(
        "Simulated WhatsApp number (optional)",
        key="sender_phone",
        placeholder="+91 98765 43210",
        help="On WhatsApp the sender's number is always known. Enter one to simulate that: "
             "a returning number reopens the same lead. Leave blank for an anonymous chat.",
    )


def render_message_input() -> str | None:
    """Render the composer. Returns the submitted text, or None."""
    with st.form("composer", clear_on_submit=True, border=False):
        input_col, send_col = st.columns([6, 1], vertical_alignment="bottom")
        with input_col:
            text = st.text_input(
                "Message",
                placeholder="Type a message…",
                label_visibility="collapsed",
            )
        with send_col:
            sent = st.form_submit_button("Send", type="primary", width="stretch")

    if sent and text.strip():
        return text.strip()
    return None
