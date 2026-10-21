"""Entry point for the AI WhatsApp Lead Automation chat simulator.

Run with:  streamlit run app.py
"""

import sqlite3

import streamlit as st

from core.config import Settings, get_settings
from core.logging_setup import configure_logging
from core.session import add_message, init_session, open_conversation, reset_chat, set_messages, start_for_lead
from database import repository as repo
from database.connection import connect
from services import conversation_service, lead_service
from ui.chat import (
    render_chat_history,
    render_conversation_status,
    render_header,
    render_message_input,
    render_sender_phone_input,
)
from ui.admin import render_admin_page
from ui.leads import render_lead_browser
from ui.sidebar import render_sidebar
from ui.styles import inject_styles

log = configure_logging()


def save_customer_message(settings: Settings, text: str) -> bool:
    """Store the message (creating the conversation on first send). Returns False on error."""
    try:
        with connect(settings.database_path) as conn:
            if st.session_state.conversation_id is None:
                conversation_id = lead_service.start_conversation(
                    conn,
                    sender_phone=(st.session_state.get("sender_phone") or "").strip() or None,
                    lead_id=st.session_state.lead_id,
                    default_country_code=settings.default_country_code,
                )
                st.session_state.conversation_id = conversation_id
                loaded = lead_service.load_conversation(conn, conversation_id)
                if loaded["lead"] and st.session_state.lead_id is None:
                    # The simulated WhatsApp number belongs to an existing lead.
                    st.session_state.lead_id = loaded["lead"]["id"]
                    st.session_state.analysis["result"] = loaded["profile"]
                    st.session_state.analysis["score"] = loaded["score"]
            message_id = lead_service.record_customer_message(conn, st.session_state.conversation_id, text)
    except ValueError as exc:
        st.session_state.input_error = str(exc)
        return False
    except sqlite3.Error as exc:
        st.session_state.input_error = f"Database error, message not saved: {exc}"
        return False
    add_message("user", text, message_id=message_id)
    return True


def run_pending(settings: Settings, mode: str) -> None:
    """Run a turn ("turn"), analysis only ("analysis") or the reply only ("reply")."""
    state = st.session_state.analysis
    conversation_id = st.session_state.conversation_id
    labels = {"turn": "Processing message…", "analysis": "Re-running analysis…", "reply": "Writing reply…"}
    with st.status(labels[mode], expanded=False) as status, connect(settings.database_path) as conn:
        if mode == "reply":
            result = conversation_service.retry_reply(conn, conversation_id, settings)
        else:
            status.update(label="Analysing message and updating the lead…")
            result = conversation_service.process_turn(conn, conversation_id, settings, generate=mode == "turn")
        set_messages(repo.get_messages(conn, conversation_id))
        if result.ok:
            status.update(label="Done", state="complete")
        else:
            status.update(label="Failed", state="error")

    state["pending"] = False
    state["error"] = result.error
    state["failed_stage"] = result.failed_stage
    if result.reply:
        state["reply"] = {"asked": result.reply.asked_topics, "regenerated": result.reply.regenerated}
    if result.outcome:
        state["result"] = result.outcome.profile
        state["score"] = result.outcome.score
        st.session_state.lead_id = result.outcome.lead_id
        st.session_state.persist = {"summary": result.outcome.summary(), "action": result.outcome.action,
                                    "notes": result.outcome.notes}
        st.toast(result.outcome.summary(), icon="💾")


def handle_browser_action(settings: Settings, action: dict) -> None:
    with connect(settings.database_path) as conn:
        if action["type"] == "open_conversation":
            open_conversation(lead_service.load_conversation(conn, action["conversation_id"]))
        else:
            start_for_lead(lead_service.load_lead(conn, action["lead_id"]))


def render_error(state: dict) -> None:
    """Show the last error with a Retry button that resumes from the failed stage."""
    st.error(state["error"], icon="⚠️")
    if state.get("failed_stage") in ("analysis", "save", "reply"):
        label = "Retry reply" if state["failed_stage"] == "reply" else "Retry"
        if st.button(label, icon="🔁"):
            state["pending"] = "reply" if state["failed_stage"] == "reply" else "turn"
            state["error"] = None
            st.rerun()


def chat_page() -> None:
    settings = get_settings()
    init_session()

    with connect(settings.database_path) as conn:
        leads = repo.list_leads(conn)
        unlinked_count = len(repo.list_conversations(conn, None))
        activities = repo.list_activities(conn, st.session_state.lead_id, limit=5) if st.session_state.lead_id else []

    if render_sidebar(settings, activities):
        st.session_state.analysis["pending"] = "analysis"

    if render_header(settings):
        reset_chat()
        st.rerun()

    def load_conversations(lead_id: int | None) -> list[dict]:
        with connect(settings.database_path) as conn:
            return repo.list_conversations(conn, lead_id)

    action = render_lead_browser(leads, unlinked_count, load_conversations)
    if action:
        handle_browser_action(settings, action)
        st.rerun()

    render_conversation_status(st.session_state.conversation_id, st.session_state.lead_id)
    render_chat_history(st.session_state.messages)

    # Process after the new message is already visible in the history.
    if st.session_state.analysis["pending"]:
        run_pending(settings, st.session_state.analysis["pending"])
        st.rerun()

    if st.session_state.analysis["error"]:
        render_error(st.session_state.analysis)
    if st.session_state.get("input_error"):
        st.error(st.session_state.pop("input_error"), icon="⚠️")

    if st.session_state.conversation_id is None and st.session_state.lead_id is None:
        render_sender_phone_input()

    submitted_text = render_message_input()
    if submitted_text and save_customer_message(settings, submitted_text):
        st.session_state.analysis["pending"] = "turn"
        st.rerun()
    elif submitted_text:
        st.rerun()



def admin_page() -> None:
    render_admin_page(get_settings())


def check_startup(settings: Settings) -> bool:
    """Show configuration warnings; stop with a clear message if the database cannot be opened."""
    for warning in settings.config_warnings:
        st.warning(f"Configuration: {warning}", icon="⚙️")
    try:
        with connect(settings.database_path):
            pass
    except (sqlite3.Error, OSError, RuntimeError, ValueError) as exc:
        log.error("Cannot open database at %s: %s", settings.database_path, exc)
        st.error(f"Cannot open the database at `{settings.database_path}`: {exc}. "
                 "Check DATABASE_PATH in .env and that the folder is writable.", icon="🛑")
        return False
    return True


def main() -> None:
    settings = get_settings()
    st.set_page_config(
        page_title=settings.app_title,
        page_icon="💬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    if not check_startup(settings):
        st.stop()
    page = st.navigation(
        [
            st.Page(chat_page, title="Chat simulator", icon="💬", default=True),
            st.Page(admin_page, title="Admin CRM", icon="📊", url_path="admin"),
        ],
        position="top",
    )
    page.run()


if __name__ == "__main__":
    main()
