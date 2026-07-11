# 🤖 Misty — A Memory-Aware Conversational Agent for Human-Robot Interaction

> *"Not just a chatbot. A bot that remembers you."*

Misty is a **RAG-based (Retrieval-Augmented Generation) conversational agent** designed for Human-Robot Interaction (HRI) research. Unlike standard chatbots, Misty maintains **persistent memory across sessions** — she remembers what she has shared with each user, what the user has told her, and continues conversations naturally, just like a real person would.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **Persistent Memory** | Remembers past conversations across multiple sessions per user |
| 🔁 **Self-Disclosure Engine** | Selectively shares personal memories based on emotional relevance |
| 🚫 **No Repeat Disclosures** | Never shares the same memory twice with the same user |
| 👤 **User Context Tracking** | Stores what the user has shared — grief, struggles, life events |
| 💬 **Human-like Continuity** | Picks up conversations where they left off, like two friends reuniting |
| 🔍 **Semantic Memory Retrieval** | Uses ChromaDB + Google Generative AI Embeddings for context-aware retrieval |
| 📋 **Comprehensive Logging** | Every session logged in human-readable `.log` and machine-readable `.json` |
| 🧪 **Synthetic Dataset** | Includes a labeled multi-turn self-disclosure dataset for research |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      rag_bot.py                         │
│              (Main entry point + Chat loop)             │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
┌─────────────┐  ┌──────────────┐  ┌──────────┐
│PromptManager│  │MemoryRetriever│  │ MistyLogger│
│ (prompts.py)│  │(memory_       │  │(logger.py) │
│             │  │ retriever.py) │  └──────────┘
└──────┬──────┘  └──────┬───────┘
       │                │
       ▼                ▼
┌─────────────┐  ┌──────────────┐
│  PostgreSQL │  │   ChromaDB   │
│  Database   │  │ Vector Store │
│             │  │              │
│ • users     │  │ • Misty's    │
│ • disclosures│ │   memories   │
│ • user_     │  │   (embedded) │
│   conversation│ └──────────────┘
│   _contexts │
└─────────────┘
```

---

## 🧠 How Memory Works

Misty's memory system operates on **two tracks simultaneously**:

### Track 1 — Misty's Own Memories (Self-Disclosure)
Misty has a set of personal memories stored in **ChromaDB** as vector embeddings. When a user shares something emotionally relevant, Misty retrieves the most semantically similar memory and decides (via a **Decider LLM**) whether to disclose it.

- Once disclosed to a user → **never repeated**
- Stored in `disclosures` table with `memory_id`, `memory_topic`, `session_id`
- Full memory text recovered from ChromaDB for rich context

### Track 2 — What the User Shared (User Context)
At the end of every session, an **LLM-generated summary** of what the user discussed is saved:

- Their situation, emotions, life events
- Key topics mentioned
- Last message exchanged

This is stored in `user_conversation_contexts` table and **injected into every future prompt**, so Misty never asks the user to repeat themselves.

---

## 💡 Self-Disclosure Decision Pipeline

```
User Message
     │
     ▼
ChromaDB Semantic Search
     │
     ▼
Score >= 0.20 threshold?
     │
    YES
     │
     ▼
Already disclosed to this user? ──YES──► Base Prompt (no re-disclosure)
     │
    NO
     │
     ▼
Decider LLM evaluates:
  • Is user sharing something emotional?
  • Would this memory build connection?
  • Is it relevant without being intrusive?
     │
   TRUE
     │
     ▼
Memory Disclosed → Saved to DB → Memory Prompt Built
```

---

## 🗄️ Database Schema

### `users`
| Column | Type | Description |
|---|---|---|
| `user_id` | Integer PK | Unique user identifier |
| `name` | String | User's name |
| `rapport_level` | Integer | Relationship depth score |
| `user_preference` | JSON | Stored preferences |

### `disclosures`
| Column | Type | Description |
|---|---|---|
| `disclosure_id` | Integer PK | Auto-increment |
| `user_id` | FK → users | Which user |
| `session_id` | Integer | Which session |
| `memory_id` | String | ChromaDB memory identifier |
| `memory_topic` | String | First 250 chars of memory |
| `disclosure_timestamp` | DateTime | When disclosed |

### `user_conversation_contexts`
| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `user_id` | FK → users | Which user |
| `session_id` | Integer | Which session |
| `user_summary` | Text | LLM-generated summary of what user shared |
| `topics` | String | Comma-separated key topics |
| `last_user_message` | Text | Final user message of session |
| `last_bot_response` | Text | Final Misty response of session |
| `created_at` | DateTime | Session timestamp |

---

## 📁 Project Structure

```
RADIA_PROJECT/
│
├── rag_bot.py              # Main entry point — chat loop + session management
├── prompts.py              # PromptManager — all prompt building + memory logic
├── memory_retriever.py     # ChromaDB semantic search + memory retrieval
├── models.py               # SQLAlchemy ORM models (3 tables)
├── database.py             # DB connection + init_db()
├── logger.py               # Session logging (plain text + JSON)
│
├── synthetic_dataset.json  # Labeled self-disclosure dataset (5 users × 3 sessions)
│
├── .env                    # 🔒 Secret keys (NOT committed to GitHub)
├── .env.example            # Template showing required environment variables
├── .gitignore              # Excludes .env, logs/, chroma_db/, __pycache__/
│
├── logs/                   # Auto-generated session logs
│   ├── user_1_session_*.log
│   └── user_1_session_*_summary.json
│
└── chroma_db/              # ChromaDB vector store (Misty's memories)
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- PostgreSQL running locally
- Google AI Studio API Key

### 1. Clone the repository

```bash
git clone https://github.com/Ravindra-Singh-Rajpurohit7/HRI_PROJECT_CONTRIBUTION.git
cd HRI_PROJECT_CONTRIBUTION
```

### 2. Install dependencies

```bash
pip install langchain-google-genai langchain-chroma sqlalchemy psycopg2-binary python-dotenv
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:

```
GOOGLE_API_KEY=your_google_api_key_here
DATABASE_URL=postgresql://username:password@localhost:5432/your_db_name
```

### 4. Set up the database

Create a PostgreSQL database, then run:

```bash
python -c "from database import init_db; init_db()"
```

This automatically creates all required tables.

### 5. Run the bot

```bash
python rag_bot.py
```

---

## 🚀 Example Interaction

**Session 1:**
```
Enter User Name: Aditya

Misty is ready. Type 'exit' to quit.

You: I embarrassed myself in front of everyone today
Misty: Oh I get that — I once tripped walking up to receive an
       award in front of a whole room. That split second of wanting
       to disappear is real. People are more forgiving than it feels
       in the moment though.

[MEMORY DISCLOSED] -> challenge_003
```

**Session 2 (next day, fresh start):**
```
Enter User Name: Aditya

You: Hey
Misty: Hey Aditya! How have you been? Last time we talked you were
       going through a pretty tough moment — how are things now?

You: Do you remember the embarrassing story you told me?
Misty: Yeah, I remember — I told you about tripping at the award
       ceremony. Why, something similar happen again?
```

---

## 📊 Synthetic Self-Disclosure Dataset

Included in `synthetic_dataset.json` — a labeled multi-turn conversational dataset for HRI and self-disclosure research.

**Structure:**
```json
{
  "user_1": {
    "session_1": {
      "topic": "Exam stress and self-doubt",
      "turns": [
        {
          "turn": 1,
          "role": "agent",
          "content": "Hey! How are you doing today?",
          "self_disclosure": false,
          "disclosure_type": "none",
          "disclosure_topic": null
        },
        {
          "turn": 3,
          "role": "agent",
          "content": "I remember sitting with my notes the night before...",
          "self_disclosure": true,
          "disclosure_type": "personal_experience",
          "disclosure_topic": "exam anxiety"
        }
      ]
    }
  }
}
```

**Dataset Stats:**

| Metric | Value |
|---|---|
| Users | 5 |
| Sessions per user | 3 |
| Total sessions | 15 |
| Avg turns per session | 9-10 |
| Disclosure turns | ~28% |
| Non-disclosure turns | ~72% |
| Disclosure types | personal_experience, emotion, opinion, vulnerability |

---

## 📋 Session Logging

Every session generates two log files in `logs/`:

**`user_1_session_*.log`** — Human readable:
```
════════════════════════════════════════
  TURN 1  |  10:32:05 UTC
════════════════════════════════════════
  [ USER INPUT ]
  I embarrassed myself today

  [ CHROMA RETRIEVAL ]
  Memory ID        : challenge_003
  Para Score       : 0.7141
  Already Disclosed: False

  [ DECIDER OUTPUT ]
  Decision : TRUE

  [ MISTY RESPONSE ]
  Oh I get that feeling...
```

**`user_1_session_*_summary.json`** — Machine readable, contains full turn-by-turn trace for analysis.

---

## 🔬 Research Context

This system was developed as part of **Human-Robot Interaction (HRI)** research exploring:

- **Selective self-disclosure** in conversational agents
- **Rapport building** through memory and continuity
- **Long-term user modeling** without restarting context each session
- **Self-disclosure detection** using annotated conversational data

---

## 🔒 Security Notes

- API keys are managed via `.env` file — never committed to version control
- `.env` is listed in `.gitignore`
- Use `.env.example` as a template when setting up on a new machine

---

## 👤 Author

**Ravindra Singh Rajpurohit**
HRI Project Contribution