# AI WhatsApp Lead Automation System

A local AI assistant that qualifies sales leads from customer chats. Customers
chat in a **web-based WhatsApp simulator**. The app:
- detects the intent and extracts requirements with a Groq-hosted Qwen model
- scores each lead with deterministic rules (HOT / WARM / COLD)
- stores everything in SQLite
- replies with useful follow-up questions

An **Admin CRM** page provides lead management, KPIs and analytics.

> WhatsApp integration and automatic follow-up sending are not implemented yet.
> Replies appear only in the simulator.

## Features

| Area | What it does |
|---|---|
| Chat simulator | Conversation history, Send and New Chat, an optional simulated WhatsApp number, reopening any saved conversation |
| NLP (Groq/Qwen) | 12 service-intent categories, a separate purchase intent, 17 entity fields with confidence, merging across messages, conflict and correction handling |
| Validation | Every model response is validated with Pydantic, with one corrective retry; contact details absent from the customer's text are discarded |
| Scoring | Deterministic rules with seven criteria (20/20/15/20/10/10/5 = 100); HOT ≥ 80, WARM 50–79, COLD < 50 |
| Replies | Context-aware; Python picks at most two follow-up questions; replies are checked for invented prices and internal data before they are shown |
| Persistence | SQLite with versioned migrations; returning customers update their existing lead; score and status history |
| Admin CRM | Search and filters, lead detail, lifecycle status, salesperson, notes, follow-ups, 10 KPIs, 8 charts, rule-based insights |
| Reliability | Clear errors for missing or invalid settings and Groq failures; rollback on failed saves; logs with secrets masked |

## Installation

Requires **Python 3.12+** (the app uses `sqlite3`'s `autocommit` option).

```bash
cd assignment
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

## Environment setup

```bash
copy .env.example .env        # macOS/Linux: cp .env.example .env
```

Edit `.env` and set at least:

```
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=exact_groq_model_id        # e.g. the Qwen model enabled on your Groq account
GROQ_REASONING_FORMAT=hidden          # for Qwen3-style reasoning models; leave blank otherwise
```

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Groq API key | **required** |
| `GROQ_MODEL` | Exact Groq model ID (never substituted) | **required** |
| `GROQ_REASONING_FORMAT` | `hidden`, `raw` or `parsed` for reasoning models | blank |
| `GROQ_TEMPERATURE` | Extraction temperature (0–2) | `0` |
| `GROQ_REPLY_TEMPERATURE` | Reply temperature (0–2) | `0.4` |
| `DATABASE_PATH` | SQLite file, relative to the project folder | `data/leads.db` |
| `DEFAULT_COUNTRY_CODE` | Added to 10-digit phone numbers (1–3 digits) | `91` |
| `BUSINESS_PROFILE_PATH` | Business details and prices the assistant may quote | `business_profile.json` |
| `APP_TITLE` | Title shown in the header and browser tab | `Lead Assistant` |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`… | `INFO` |

- **Re-read on every message:** changes to `.env` apply without a restart.
- **Invalid values never crash the app:** each falls back to its default and
  appears as a ⚙️ configuration warning at the top of the page.
- **Missing key or model:** an error is shown on the first message, and no
  AI call is made.

**Optional business profile.** Without it, the assistant never states
prices or office details; it says the team will confirm.

```bash
copy business_profile.example.json business_profile.json   # macOS/Linux: cp
```

Edit the name, office location, hours, contacts and `pricing`. Prices are
quoted word for word, and only for the matching service.

## Running the app

```bash
streamlit run app.py
```

Open http://localhost:8501. The top navigation has two pages:

- **💬 Chat simulator:**
  - Type as a customer. Optionally enter a *simulated WhatsApp number*
    before the first message; a known number continues that lead.
  - Use **New Chat** to start fresh.
  - Use **📂 Leads & conversations** to reopen a saved conversation or start
    a new one with an existing lead.
  - The developer sidebar shows the score breakdown, intent, entities,
    missing information, CRM status and the chosen follow-up topics.
- **📊 Admin CRM:**
  - **Dashboard:** KPIs, charts and rule-based insights.
  - **Leads:** search, filter, open a lead, update its status, assign a
    salesperson, add notes, schedule follow-ups, mark it Converted or Lost.

## Database

- **Location:** `data/leads.db` (or `DATABASE_PATH`).
- **Setup:** the folder, tables and seed data are created automatically, and
  migrations run automatically on start-up (current schema version: **3**,
  stored in `PRAGMA user_version`).
- **Extra files:** SQLite runs in WAL mode, so `leads.db-wal` and
  `leads.db-shm` may appear. Keep them with the `.db` file.

| Table | Contents |
|---|---|
| `leads` | One row per customer: profile, score, category, lifecycle status, salesperson, contact and follow-up dates |
| `conversations` | Linked to a lead; channel, session identifier, sender phone, latest analysis |
| `messages` | Customer and assistant messages; processing status; follow-up topics asked |
| `lead_scores` | Score history with the criterion breakdown as JSON |
| `lead_status_history` | Every lifecycle change, with an optional note |
| `follow_ups` | Due date, status, number, customer response, next action, completion time |
| `activities` | Audit trail (created, updated, corrected, conflict, merged, notes, assignments…) |
| `services` | Service catalogue (9 services) |

**Inspect** (read-only):

```bash
python -m scripts.inspect_db        # table counts + recent leads
python -m scripts.inspect_db 3      # lead #3: fields, score history, conversations, activities
```

You can also open the file in a SQLite viewer such as DB Browser for SQLite
or the VS Code "SQLite Viewer" extension.

**Back up and reset.** Stop the app, then copy `data/` to back up, or
delete `data/` to start with an empty CRM.

## How it works

**One customer turn** (`services/conversation_service.py`):
1. Store the message.
2. Extract intent and entities from the full stored conversation, with the
   lead's stored profile as context.
3. Merge, score and persist in a single transaction.
4. Generate the reply from the updated profile.
5. Store the reply.

If extraction fails, no reply is attempted (**Retry**). If only the reply
fails, the lead update is kept (**Retry reply**). Context always comes from
the database, so reopened and returning conversations behave exactly like
live ones.

**Extraction** (`ai/nlp_service.py`, `ai/schemas.py`):
- Groq JSON mode, with code fences stripped.
- Pydantic validation, plus one retry with the validation errors.
- "N/A", "unknown" and similar values become `null`.
- Emails and phones not literally present in the customer's messages are
  discarded.
- Merging keeps earlier values, applies explicit corrections ("actually…"),
  and flags conflicts instead of overwriting.

**Scoring** (`scoring/engine.py`). Rules are documented at the top of the
file. A value earns full points only if it is *explicit*: confidence ≥ 0.8
and its evidence appears in the customer's words. Inferred values earn
reduced points, and missing values earn 0. The validated result guarantees
an integer total from 0 to 100 that equals the sum of the criteria, no
criterion above its maximum, and a category matching the boundaries.

| Criterion | Max | Criterion | Max |
|---|---|---|---|
| Requirement clarity | 20 | Business information | 10 |
| Budget | 20 | Engagement | 10 |
| Timeline | 15 | Decision-making authority | 5 |
| Purchase intent | 20 | **Total** | **100** |

**Replies** (`ai/responder.py`):
- **Question planning:** Python picks the questions. The priority order
  depends on the intent, known topics are never re-asked, there are at most
  two questions, and a question ignored twice is dropped.
- **What the model sees:** only facts the customer gave plus the business
  profile. It never sees scores, categories or purchase intent.
- **Checks before a reply is saved:** amounts that are neither configured
  nor stated by the customer, scoring language ("hot lead", "80/100"), and
  developer vocabulary (extraction, JSON, field names) are rejected. A
  failing reply is regenerated once; if it fails again, it is reported as
  an error.

**Returning customers** (`services/lead_service.py`):
- **Matching order:** normalised phone first, then email (only if the
  phones don't conflict), then an explicitly reopened lead.
- **Anonymous chats** are never merged by guessing.
- **A temporary lead that turns out to be a returning customer** is merged
  into the existing lead, keeping all its history.

**Admin CRM** (`services/admin_service.py`, `database/analytics.py`):
- **Lifecycle:** New → Contacted → Qualified → Proposal → Negotiation →
  Converted/Lost.
- **What admin actions touch:** only admin fields and history rows, never
  messages, the extracted profile or score history.
- **Insights:** fixed rules over real counts. No AI is used, and no causes
  are claimed. Conversion comparisons need at least 5 closed leads.

## Project structure

```
assignment/
├── app.py                      # Entry point: navigation, chat page, start-up checks
├── core/
│   ├── config.py               # Settings from .env (validated, with fallbacks)
│   ├── logging_setup.py        # Rotating log file + secret redaction
│   ├── business.py             # Optional business profile (details + prices)
│   └── session.py              # Streamlit session state
├── ai/
│   ├── groq_client.py          # Groq SDK calls + error mapping + timing logs
│   ├── schemas.py              # Pydantic models for extraction and analysis
│   ├── prompts.py              # Extraction prompts
│   ├── nlp_service.py          # Extraction, validation, merge, missing info
│   ├── responder.py            # Question planning, reply prompt, reply checks
│   └── errors.py               # Error types shown in the UI
├── scoring/
│   ├── engine.py               # Deterministic scoring rules
│   └── models.py               # Validated LeadScore / CriterionScore
├── database/
│   ├── schema.py               # Tables, indexes, versioned migrations
│   ├── connection.py           # Connections, transactions
│   ├── repository.py           # All SQL (parameterised, allowlisted identifiers)
│   └── analytics.py            # Read-only dashboard aggregates
├── services/
│   ├── conversation_service.py # Turn pipeline: analyse → persist → reply
│   ├── lead_service.py         # Lead matching, merging, scoring, history
│   ├── admin_service.py        # Lifecycle, assignment, notes, follow-ups
│   └── analytics_service.py    # Dashboard data + rule-based insights
├── ui/
│   ├── chat.py, sidebar.py, leads.py, styles.py   # Chat page
│   ├── admin.py                # Admin CRM page
│   └── charts.py               # Altair charts
├── tests/                      # Offline automated tests (fake Groq client)
├── scripts/
│   ├── inspect_db.py           # Read-only database overview
│   ├── test_extraction_live.py # Live extraction examples (uses Groq quota)
│   └── test_conversation_live.py  # Live end-to-end conversation (temporary database)
├── business_profile.example.json
├── requirements.txt
└── .env.example
```

## Tests

```bash
# All offline tests (no network, no API cost, temporary databases)
python -m unittest discover tests

# One area
python -m unittest tests.test_reliability
```

| File | Covers |
|---|---|
| `test_nlp_service.py` | Extraction validation, retries, merging, conflicts, Groq error mapping |
| `test_scoring.py` | Scoring rules, explicit vs inferred evidence, boundaries, maximums, determinism |
| `test_persistence.py` | Schema and migrations, lead create and update, duplicates, merging, history, injection |
| `test_conversation.py` | Multi-turn replies, pricing honesty, returning customers, question planning, reply checks |
| `test_admin.py` | v3 migration, lifecycle, follow-ups, search and filters, KPIs and charts vs the database, restart |
| `test_reliability.py` | End-to-end workflow (10 steps), invalid settings, malformed output, logging privacy, rollback, safety |

**Live checks** use your Groq quota and a reasoning model, so they take a
few minutes:

```bash
python -m scripts.test_extraction_live      # 7 enquiry types with scores
python -m scripts.test_conversation_live    # full conversation + dashboard/reopen checks (temporary database)
```

## Logging and security

- **Logs:** written to `logs/app.log` (rotating, 1 MB × 3). Warnings also go
  to the console.
- **What's logged:** lead ids, actions, timings, token counts and error
  types. Customer message text, emails and phone numbers are never logged.
- **Secrets:** a filter masks the configured API key and any `gsk_…` token
  in every log record, and `Settings` never prints the key.
- **Version control:** `.env`, `.env.*` (except `.env.example`), `data/`,
  `logs/` and database files are git-ignored. This folder is not a git
  repository yet; run `git init` before your first commit, and the ignore
  rules apply from then on.
- **SQL safety:** every value is a bound parameter. Column and table names
  come only from fixed allowlists, and unknown names raise an error.
  Multi-step writes run in transactions and roll back on failure.

## Limitations

- **Simulator only.** No WhatsApp integration and no automatic follow-up
  sending; follow-ups are recorded for salespeople to act on.
- **Speed.** Each message makes two or more Groq requests. With a reasoning
  model a reply can take tens of seconds, and the Groq free tier has rate
  limits.
- **Model judgement.** Extraction quality depends on the model. Contact
  details are verified against the text, but other fields (industry,
  intent) are the model's interpretation, shown with confidence and scored
  lower when not explicit.
- **Reply checks are pattern-based.** They block amounts and known internal
  terms, not every possible wording.
- **Single local user.** There is no authentication on the Admin CRM. Run
  it on a trusted machine only.
- **Light theme only.** `.streamlit/config.toml` pins the light theme.
- **Insights are rule-based.** They describe counts, not causes, and
  comparisons need enough closed leads.

## Troubleshooting

| Symptom | Fix |
|---|---|
| "GROQ_API_KEY is not set" / "GROQ_MODEL is not set" | Copy `.env.example` to `.env` and fill both values; the next message picks them up |
| "Groq rejected the API key (401)" | Check the key in `.env` (no quotes or spaces) |
| "Model '…' was not found (404)" | Use the exact model ID from your Groq console; the app never swaps models |
| "Groq rejected the request (400) … reasoning_format" | Your model is not a reasoning model: clear `GROQ_REASONING_FORMAT` |
| "Groq rate limit reached (429)" | Wait a minute and click **Retry** |
| "The generated reply was rejected" | The model tried to state a price or internal detail twice; click **Retry reply** (or configure prices in `business_profile.json`) |
| ⚙️ Configuration warning banner | A `.env` value is invalid; the message names it and the default being used |
| "Cannot open the database at …" | Fix `DATABASE_PATH` or the folder's permissions; restore a backup of `data/` if the file is corrupt |
| "Database schema version N is newer than this app supports" | The database came from a newer version of the app; use that version or restore a backup |
| `database is locked` | Close other programs that have `leads.db` open (e.g. a DB viewer in write mode) |
| Old data or behaviour after editing code | Stop Streamlit (Ctrl+C) and start it again |
| `ModuleNotFoundError` | Activate the virtual environment and run `pip install -r requirements.txt` |

To diagnose, check `logs/app.log`; it records each Groq request, failure
stage and lead update.
