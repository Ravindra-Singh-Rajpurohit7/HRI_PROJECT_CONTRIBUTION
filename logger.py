"""
logger.py
---------
Comprehensive logging and debugging framework for the Misty RAG bot.

Drop this file in the same folder as rag_bot.py and prompts.py.

Usage in rag_bot.py
--------------------
    from logger import MistyLogger

    logger = MistyLogger(user_id=USER_ID, session_id=SESSION_ID)

    # inside chat loop:
    logger.start_turn(query)
    logger.log_retrieval(...)
    logger.log_disclosures(...)
    logger.log_user_context(...)
    logger.log_decider(...)
    logger.log_disclosure_save(...)
    logger.log_prompt(...)
    logger.log_response(answer)
    logger.log_chat_history(...)

    # on exit:
    logger.save(chat_history=prompt_manager.chat_history)

Output files (created automatically in ./logs/)
------------------------------------------------
  user_<id>_session_<id>.log          <- human-readable plain text
  user_<id>_session_<id>_summary.json <- machine-readable JSON
"""

import json
import os
from datetime import datetime


# ==================================================
# LOG DIRECTORY
# ==================================================

LOG_DIR = "./logs"


def _ensure_log_dir():

    os.makedirs(
        LOG_DIR,
        exist_ok=True
    )


# ==================================================
# MISTY LOGGER
# ==================================================

class MistyLogger:

    def __init__(
        self,
        user_id,
        session_id
    ):

        self.user_id = user_id
        self.session_id = session_id
        self.turn_number = 0
        self.entries = []
        self.current_turn = {}
        self.session_start = datetime.utcnow().isoformat()

        _ensure_log_dir()

        self.log_path = os.path.join(
            LOG_DIR,
            f"user_{user_id}_session_{session_id}.log"
        )

        self._write_header()

    # --------------------------------------------------
    # INTERNAL HELPERS
    # --------------------------------------------------

    def _write_header(self):

        header = (
            "\n"
            + "=" * 60 + "\n"
            + "  MISTY RAG BOT SESSION LOG\n"
            + f"  User ID    : {self.user_id}\n"
            + f"  Session ID : {self.session_id}\n"
            + f"  Started    : {self.session_start} UTC\n"
            + "=" * 60 + "\n"
        )

        self._append_raw(header)

    def _append_raw(self, text):

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text)

    def _section(self, title, content=""):

        block = (
            "\n"
            + f"  [ {title} ]\n"
            + "-" * 50 + "\n"
            + str(content).strip() + "\n"
        )

        self._append_raw(block)

    # --------------------------------------------------
    # TURN START
    # --------------------------------------------------

    def start_turn(self, user_query):

        self.turn_number += 1

        self.current_turn = {
            "turn": self.turn_number,
            "timestamp": datetime.utcnow().isoformat(),
            "user_query": user_query
        }

        header = (
            "\n"
            + "=" * 60 + "\n"
            + f"  TURN {self.turn_number}"
            + f"  |  {datetime.utcnow().strftime('%H:%M:%S')} UTC\n"
            + "=" * 60 + "\n"
        )

        self._append_raw(header)
        self._section("USER INPUT", user_query)

    # --------------------------------------------------
    # CHROMA RETRIEVAL LOG
    # --------------------------------------------------

    def log_retrieval(
        self,
        memory_id,
        best_para,
        best_sent,
        best_sent_score,
        best_para_score,
        already_disclosed
    ):

        para_lines = "\n".join(
            f"  {i + 1}. {s}"
            for i, s in enumerate(best_para)
        )

        content = (
            f"Memory ID        : {memory_id}\n"
            f"Para Score       : {round(best_para_score, 4)}\n"
            f"Sent Score       : {round(best_sent_score, 4)}\n"
            f"Already Disclosed: {already_disclosed}\n\n"
            f"Best Sentence:\n"
            f"  {best_sent}\n\n"
            f"Full Paragraph Retrieved:\n"
            f"{para_lines}"
        )

        self._section("CHROMA RETRIEVAL", content)

        self.current_turn["retrieval"] = {
            "memory_id": memory_id,
            "best_para_score": best_para_score,
            "best_sent_score": best_sent_score,
            "already_disclosed": already_disclosed,
            "best_sent": best_sent,
            "best_para": best_para
        }

    # --------------------------------------------------
    # DB DISCLOSURES LOG
    # --------------------------------------------------

    def log_disclosures(self, previous_disclosures):

        if previous_disclosures:

            content = "\n".join(
                f"  - {mid}"
                for mid in previous_disclosures
            )

        else:

            content = (
                "  (none — new user or no prior disclosures)"
            )

        self._section(
            "DB: PREVIOUS DISCLOSURES FOR THIS USER",
            content
        )

        self.current_turn["previous_disclosures"] = (
            previous_disclosures
        )

    # --------------------------------------------------
    # KNOWN USER CONTEXT LOG
    # --------------------------------------------------

    def log_user_context(self, user_context):

        self._section(
            "KNOWN USER CONTEXT (injected into prompt)",
            user_context
        )

        self.current_turn["user_context"] = user_context

    # --------------------------------------------------
    # DECIDER LOG
    # --------------------------------------------------

    def log_decider(
        self,
        candidate_paragraph,
        decision,
        reason=None
    ):

        snippet = (
            candidate_paragraph[:250] + "..."
            if len(candidate_paragraph) > 250
            else candidate_paragraph
        )

        content = (
            f"Candidate Memory Snippet:\n"
            f"  {snippet}\n\n"
            f"Decision : {decision.upper()}\n"
        )

        if reason:
            content += f"Reason   : {reason}\n"

        self._section("DECIDER OUTPUT", content)

        self.current_turn["decider"] = {
            "decision": decision,
            "reason": reason or "",
            "candidate_snippet": snippet
        }

    # --------------------------------------------------
    # DISCLOSURE SAVE LOG
    # --------------------------------------------------

    def log_disclosure_save(self, memory_id, saved):

        content = (
            f"Memory ID : {memory_id}\n"
            f"Action    : "
            + (
                "SAVED — new disclosure written to DB."
                if saved
                else "SKIPPED — already exists in DB."
            )
        )

        self._section("DISCLOSURE SAVE", content)

        self.current_turn["disclosure_save"] = {
            "memory_id": memory_id,
            "saved": saved
        }

    # --------------------------------------------------
    # FINAL PROMPT LOG
    # --------------------------------------------------

    def log_prompt(self, final_prompt, prompt_type="base"):

        content = (
            f"Prompt Type : {prompt_type}\n\n"
            + final_prompt
        )

        self._section(
            "FINAL PROMPT SENT TO LLM",
            content
        )

        self.current_turn["prompt"] = {
            "type": prompt_type,
            "content": final_prompt
        }

    # --------------------------------------------------
    # BOT RESPONSE LOG
    # --------------------------------------------------

    def log_response(self, answer):

        self._section("MISTY RESPONSE", answer)

        self.current_turn["response"] = answer

        # Finalise this turn and push to entries list
        self.entries.append(dict(self.current_turn))

    # --------------------------------------------------
    # FULL CHAT HISTORY SNAPSHOT
    # --------------------------------------------------

    def log_chat_history(self, chat_history):

        if not chat_history:
            return

        content = "\n".join(
            f"  {line}"
            for line in chat_history
        )

        self._section(
            "FULL CHAT HISTORY (this session so far)",
            content
        )

    # --------------------------------------------------
    # SESSION END — SAVE JSON SUMMARY
    # --------------------------------------------------

    def save(self, chat_history=None):

        if chat_history:
            self.log_chat_history(chat_history)

        summary_path = os.path.join(
            LOG_DIR,
            f"user_{self.user_id}_session_{self.session_id}_summary.json"
        )

        summary = {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "session_start": self.session_start,
            "session_end": datetime.utcnow().isoformat(),
            "total_turns": self.turn_number,
            "turns": self.entries
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                summary,
                f,
                indent=2,
                default=str
            )

        footer = (
            "\n"
            + "=" * 60 + "\n"
            + f"  SESSION ENDED\n"
            + f"  Ended      : {datetime.utcnow().isoformat()} UTC\n"
            + f"  Total Turns: {self.turn_number}\n"
            + f"  Log File   : {self.log_path}\n"
            + f"  JSON Summary: {summary_path}\n"
            + "=" * 60 + "\n"
        )

        self._append_raw(footer)

        print(
            f"\n[LOG] Session log saved  -> {self.log_path}"
        )
        print(
            f"[LOG] JSON summary saved -> {summary_path}"
        )