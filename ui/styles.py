"""Global CSS for the chat interface.

Colors use translucent values or Streamlit theme variables so the UI
works in both the light and dark themes.
"""

import streamlit as st

_CSS = """
<style>
.block-container {
    max-width: 960px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header */
.app-title {
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0;
    line-height: 1.2;
}
.app-subtitle {
    margin: 0.25rem 0 0;
    opacity: 0.7;
    font-size: 0.95rem;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    margin-top: 0.6rem;
    padding: 0.2rem 0.65rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    background: rgba(245, 158, 11, 0.14);
    color: #b45309;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.status-pill::before {
    content: "";
    width: 0.5rem;
    height: 0.5rem;
    border-radius: 50%;
    background: #f59e0b;
}

/* Chat */
.chat-empty {
    text-align: center;
    padding: 3.5rem 1rem;
    opacity: 0.75;
}
.chat-empty h3 {
    margin-bottom: 0.4rem;
}
.msg-time {
    font-size: 0.72rem;
    opacity: 0.5;
    margin-top: -0.4rem;
}

/* Sidebar */
.panel-heading {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    opacity: 0.65;
    margin: 1.1rem 0 0.4rem;
}
.status-pill.ok {
    background: rgba(22, 163, 74, 0.12);
    color: #15803d;
    border-color: rgba(22, 163, 74, 0.35);
}
.status-pill.ok::before {
    background: #16a34a;
}
.entity-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
}
.entity-table th {
    text-align: left;
    font-weight: 600;
    opacity: 0.6;
    border-bottom: 1px solid rgba(148, 163, 184, 0.4);
    padding: 0.2rem 0.3rem;
}
.entity-table td {
    padding: 0.25rem 0.3rem;
    vertical-align: top;
    border-bottom: 1px solid rgba(148, 163, 184, 0.15);
    word-break: break-word;
}
.entity-table td:first-child {
    white-space: nowrap;
    opacity: 0.75;
}
.entity-table .conf {
    text-align: right;
    white-space: nowrap;
}
.entity-table tr.missing td {
    opacity: 0.4;
}
.level-badge {
    padding: 0.05rem 0.5rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
}
.level-badge.high { background: rgba(22, 163, 74, 0.18); color: #15803d; }
.level-badge.medium { background: rgba(245, 158, 11, 0.18); color: #b45309; }
.level-badge.low { background: rgba(239, 68, 68, 0.15); color: #b91c1c; }
.level-badge.unknown { background: rgba(148, 163, 184, 0.2); }
.chip {
    display: inline-block;
    margin: 0.15rem 0.2rem 0 0;
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    font-size: 0.75rem;
    border: 1px solid rgba(239, 68, 68, 0.4);
    background: rgba(239, 68, 68, 0.08);
}
.score-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 1rem;
    border-radius: 12px;
    margin-bottom: 0.6rem;
    border: 1px solid rgba(148, 163, 184, 0.35);
}
.score-card.hot { background: rgba(239, 68, 68, 0.12); border-color: rgba(239, 68, 68, 0.45); }
.score-card.warm { background: rgba(245, 158, 11, 0.12); border-color: rgba(245, 158, 11, 0.45); }
.score-card.cold { background: rgba(59, 130, 246, 0.10); border-color: rgba(59, 130, 246, 0.4); }
.score-value {
    font-size: 2.2rem;
    font-weight: 800;
    line-height: 1;
}
.score-value span {
    font-size: 0.9rem;
    font-weight: 500;
    opacity: 0.6;
}
.score-category {
    font-size: 1.1rem;
    font-weight: 800;
    letter-spacing: 0.08em;
}
.score-card.hot .score-category { color: #dc2626; }
.score-card.warm .score-category { color: #d97706; }
.score-card.cold .score-category { color: #2563eb; }
.criterion {
    margin-bottom: 0.55rem;
    font-size: 0.8rem;
}
.criterion-head {
    display: flex;
    justify-content: space-between;
    gap: 0.4rem;
}
.criterion .bar {
    height: 5px;
    border-radius: 3px;
    background: rgba(148, 163, 184, 0.25);
    margin: 0.2rem 0;
}
.criterion .bar div {
    height: 100%;
    border-radius: 3px;
    background: #16a34a;
}
.criterion-reason {
    opacity: 0.7;
    font-size: 0.75rem;
}
.evidence {
    font-size: 0.65rem;
    padding: 0 0.35rem;
    border-radius: 4px;
    text-transform: uppercase;
    opacity: 0.8;
    border: 1px solid rgba(148, 163, 184, 0.5);
}
.evidence.explicit { border-color: rgba(22, 163, 74, 0.6); }
.evidence.missing { opacity: 0.45; }
.activity-list {
    margin: 0.3rem 0 0;
    padding-left: 1rem;
    font-size: 0.75rem;
    opacity: 0.8;
}
.empty-state {
    padding: 0.7rem 0.85rem;
    border-radius: 8px;
    border: 1px dashed rgba(148, 163, 184, 0.5);
    font-size: 0.85rem;
    opacity: 0.75;
}

@media (max-width: 640px) {
    .block-container {
        padding-top: 1rem;
    }
    .app-title {
        font-size: 1.3rem;
    }
}
</style>
"""


def inject_styles() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
