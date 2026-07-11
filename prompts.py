from datetime import datetime

from database import SessionLocal
from models import Disclosure, UserConversationContext


class PromptManager:

    def __init__(self, decider_llm, retriever):

        self.decider_llm = decider_llm
        self.retriever = retriever

        self.used_memories = {}
        self.chat_history = []

    # ==================================================
    # MEMORY RETRIEVAL
    # ==================================================

    def _retrieve_memory(self, user_input):

        return self.retriever.retrieve_memory(
            user_input
        )

    # ==================================================
    # RESPONSE GENERATION
    # ==================================================

    def generate_response(
        self,
        client,
        prompt,
        user_input,
        add_turn_to_history=True,
    ):

        final_prompt = f"""
{prompt}

User:
{user_input}
"""

        response = client.invoke(
            final_prompt
        )

        answer = response.content

        if add_turn_to_history:

            self.chat_history.append(
                f"User: {user_input}"
            )

            self.chat_history.append(
                f"Assistant: {answer}"
            )

        return answer

    # ==================================================
    # DISCLOSURE STORAGE
    # ==================================================

    def save_disclosure(
        self,
        user_id,
        session_id,
        memory_id,
        memory_text
    ):

        db = SessionLocal()

        try:

            existing = (
                db.query(Disclosure)
                .filter(
                    Disclosure.user_id == user_id,
                    Disclosure.memory_id == memory_id
                )
                .first()
            )

            if existing:
                return

            disclosure = Disclosure(
                user_id=user_id,
                session_id=session_id,
                memory_id=memory_id,
                memory_topic=memory_text[:250],
                disclosure_timestamp=datetime.utcnow()
            )

            db.add(disclosure)
            db.commit()

            print(
                f"[DISCLOSURE SAVED] -> {memory_id}"
            )

        except Exception as e:

            print(
                "DB ERROR:",
                e
            )

            db.rollback()

        finally:

            db.close()

    # ==================================================
    # LOAD PREVIOUS DISCLOSURES (IDs ONLY)
    # ==================================================

    def get_user_disclosures(
        self,
        user_id
    ):

        db = SessionLocal()

        try:

            disclosures = (
                db.query(Disclosure)
                .filter(
                    Disclosure.user_id == user_id
                )
                .all()
            )

            memory_ids = []

            for d in disclosures:
                memory_ids.append(
                    d.memory_id
                )

            return memory_ids

        finally:

            db.close()

    # ==================================================
    # FETCH FULL MEMORY TEXT FROM CHROMA BY ID
    # ==================================================

    def _fetch_full_memory_text(
        self,
        memory_id,
        fallback_text=""
    ):

        # Strategy 1: metadata filter
        try:

            result = self.retriever.vectorstore.get(
                where={"id": memory_id}
            )

            documents = result.get("documents") or []

            if documents and documents[0]:
                return documents[0]

        except Exception as e:

            print(
                f"CHROMA FETCH (metadata filter) failed "
                f"for {memory_id}: {e}"
            )

        # Strategy 2: direct id lookup
        try:

            result = self.retriever.vectorstore.get(
                ids=[memory_id]
            )

            documents = result.get("documents") or []

            if documents and documents[0]:
                return documents[0]

        except Exception as e:

            print(
                f"CHROMA FETCH (id lookup) failed "
                f"for {memory_id}: {e}"
            )

        # Strategy 3: semantic search using stored topic as query
        if fallback_text:

            try:

                docs = (
                    self.retriever.vectorstore
                    .similarity_search(
                        fallback_text,
                        k=5
                    )
                )

                for doc in docs:

                    if doc.metadata.get("id") == memory_id:
                        return doc.page_content

            except Exception as e:

                print(
                    f"CHROMA FETCH (semantic search) failed "
                    f"for {memory_id}: {e}"
                )

        return fallback_text

    # ==================================================
    # MISTY'S DISCLOSED MEMORIES CONTEXT
    # ==================================================

    def get_user_context(
        self,
        user_id,
        limit=5
    ):

        db = SessionLocal()

        try:

            disclosures = (
                db.query(Disclosure)
                .filter(
                    Disclosure.user_id == user_id
                )
                .order_by(
                    Disclosure.disclosure_timestamp.desc()
                )
                .limit(limit)
                .all()
            )

            if not disclosures:
                return "No prior stories shared with this user yet."

            # Group by session
            sessions = {}

            for d in disclosures:

                sid = d.session_id

                if sid not in sessions:
                    sessions[sid] = {
                        "session_id": sid,
                        "timestamp": d.disclosure_timestamp,
                        "disclosures": []
                    }

                full_text = self._fetch_full_memory_text(
                    d.memory_id,
                    fallback_text=d.memory_topic
                )

                sessions[sid]["disclosures"].append({
                    "disclosure_id": d.memory_id,
                    "full_text": full_text
                })

            sorted_sessions = sorted(
                sessions.values(),
                key=lambda s: s["timestamp"],
                reverse=True
            )

            header = (
                f"Stories Misty has already shared with this user "
                f"({len(sorted_sessions)} session(s)):\n"
            )

            blocks = []

            for i, sess in enumerate(sorted_sessions):

                label = (
                    "Most recent session"
                    if i == 0
                    else f"Earlier session {i}"
                )

                ts = sess["timestamp"]
                ts_str = (
                    ts.strftime("%Y-%m-%d")
                    if ts else "unknown date"
                )

                disc_lines = []

                for disc in sess["disclosures"]:

                    disc_lines.append(
                        f"  [memory_id: {disc['disclosure_id']}]\n"
                        f"  {disc['full_text']}"
                    )

                blocks.append(
                    f"--- {label} "
                    f"(session_id={sess['session_id']}, {ts_str}) ---\n"
                    + "\n\n".join(disc_lines)
                )

            return header + "\n" + "\n\n".join(blocks)

        finally:

            db.close()

    # ==================================================
    # SAVE USER CONVERSATION CONTEXT (end of session)
    # ==================================================

    def save_user_context(
        self,
        user_id,
        session_id,
        user_summary,
        topics,
        last_user_message,
        last_bot_response
    ):

        db = SessionLocal()

        try:

            existing = (
                db.query(UserConversationContext)
                .filter(
                    UserConversationContext.user_id == user_id,
                    UserConversationContext.session_id == session_id
                )
                .first()
            )

            if existing:

                existing.user_summary = user_summary
                existing.topics = topics
                existing.last_user_message = last_user_message
                existing.last_bot_response = last_bot_response
                existing.created_at = datetime.utcnow()

                db.commit()

                print(
                    "[USER CONTEXT UPDATED] -> "
                    f"session {session_id}"
                )

            else:

                ctx = UserConversationContext(
                    user_id=user_id,
                    session_id=session_id,
                    user_summary=user_summary,
                    topics=topics,
                    last_user_message=last_user_message,
                    last_bot_response=last_bot_response,
                    created_at=datetime.utcnow()
                )

                db.add(ctx)
                db.commit()

                print(
                    "[USER CONTEXT SAVED] -> "
                    f"session {session_id}"
                )

        except Exception as e:

            print(
                "DB ERROR (save_user_context):",
                e
            )

            db.rollback()

        finally:

            db.close()

    # ==================================================
    # LOAD USER CONVERSATION CONTEXT (start of session)
    # ==================================================

    def load_user_conversation_history(
        self,
        user_id,
        limit=3
    ):

        db = SessionLocal()

        try:

            contexts = (
                db.query(UserConversationContext)
                .filter(
                    UserConversationContext.user_id == user_id
                )
                .order_by(
                    UserConversationContext.created_at.desc()
                )
                .limit(limit)
                .all()
            )

            if not contexts:
                return None

            blocks = []

            for i, ctx in enumerate(contexts):

                label = (
                    "Last conversation"
                    if i == 0
                    else f"Earlier conversation {i}"
                )

                ts_str = (
                    ctx.created_at.strftime("%Y-%m-%d")
                    if ctx.created_at else "unknown date"
                )

                block = (
                    f"--- {label} ({ts_str}) ---\n"
                )

                if ctx.user_summary:
                    block += (
                        f"What the user shared:\n"
                        f"  {ctx.user_summary}\n"
                    )

                if ctx.topics:
                    block += (
                        f"Topics discussed: {ctx.topics}\n"
                    )

                if ctx.last_user_message:
                    block += (
                        f"\nHow the conversation ended:\n"
                        f"  User said: \"{ctx.last_user_message}\"\n"
                        f"  Misty replied: \"{ctx.last_bot_response}\"\n"
                    )

                blocks.append(block)

            return "\n\n".join(blocks)

        finally:

            db.close()

    # ==================================================
    # GENERATE USER SUMMARY (LLM call at session end)
    # ==================================================

    def generate_user_summary(self, chat_history):

        if not chat_history:
            return "", ""

        history_text = "\n".join(chat_history)

        prompt = f"""
You are reading a conversation between a user and Misty (a bot).

Conversation:
{history_text}

Your task:
1. Write a 2-3 sentence summary of what the USER shared about
   themselves — their situation, feelings, problems, or life events.
   Focus only on what the USER said, not what Misty said.

2. List the key topics the user mentioned as comma-separated words.
   Example: grief, grandfather, loneliness, job stress

Reply in this exact format:
SUMMARY: <your summary here>
TOPICS: <comma separated topics>
"""

        try:

            response = self.decider_llm.invoke(prompt)
            text = response.content.strip()

            summary = ""
            topics = ""

            for line in text.splitlines():

                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()

                elif line.startswith("TOPICS:"):
                    topics = line.replace("TOPICS:", "").strip()

            return summary, topics

        except Exception as e:

            print(
                "SUMMARY GENERATION ERROR:",
                e
            )

            return "", ""

    # ==================================================
    # MAIN ENTRY
    # ==================================================

    def prompt_generation(
        self,
        user_text,
        user_id,
        session_id
    ):

        (
            memory_id,
            best_para,
            best_sent,
            best_sent_score,
            best_para_score,
            memory_scores,
            best_memory
        ) = self._retrieve_memory(
            user_text
        )

        previous_disclosures = (
            self.get_user_disclosures(
                user_id
            )
        )

        already_disclosed = (
            memory_id in previous_disclosures
        )

        if (
            best_para_score >= 0.20
            and not already_disclosed
            and self._should_use_memory(
                user_text,
                user_id,
                memory_id,
                " ".join(best_para)
            )
        ):

            self.used_memories[
                memory_id
            ] = " ".join(best_para)

            self.save_disclosure(
                user_id,
                session_id,
                memory_id,
                " ".join(best_para)
            )

            prompt = self._build_memory_prompt(
                user_text,
                best_para,
                best_sent,
                best_para_score,
                best_sent_score,
                user_id
            )

            return {
                "prompt": prompt,
                "memory_used": True,
                "memory_id": memory_id,
                "memory_text": " ".join(best_para)
            }

        return {
            "prompt": self._build_base_prompt(
                user_id
            ),
            "memory_used": False,
            "memory_id": None,
            "memory_text": None
        }

    # ==================================================
    # MEMORY DECISION
    # ==================================================

    def _should_use_memory(
        self,
        user_input,
        user_id,
        memory_id,
        best_paragraph
    ):

        previous_disclosures = (
            self.get_user_disclosures(
                user_id
            )
        )

        if memory_id in previous_disclosures:

            print(
                "Memory already disclosed to this user."
            )

            return False

        if memory_id in self.used_memories:

            print(
                "Memory already used in current session."
            )

            return False

        used_memories_text = "\n".join(
            [
                f"- {x}"
                for x in self.used_memories.values()
            ]
        ) or "None"

        history = "\n".join(
            self.chat_history[-10:]
        )

        prompt = self.decider_prompt(
            history,
            used_memories_text,
            best_paragraph
        )

        response = self.generate_response(
            self.decider_llm,
            prompt,
            user_input,
            False
        )

        decision = (
            response.strip()
            .lower()
        )

        return "true" in decision

    # ==================================================
    # DECIDER PROMPT
    # ==================================================

    def decider_prompt(
        self,
        history_text,
        used_memories_text,
        best_paragraph
    ):

        return f"""
You are a disclosure decision maker.

Conversation:
{history_text}

Already Shared Memories:
{used_memories_text}

Candidate Memory:
{best_paragraph}

Return TRUE only if:

- user asks opinion
- user asks personal experience
- user shares something emotionally similar
- memory naturally builds connection
- memory improves the conversation

Return FALSE if:

- greeting
- thanks
- factual question
- procedural question
- weak relevance
- memory would distract from user

Reply only:

true

or

false
"""

    # ==================================================
    # MEMORY RESPONSE PROMPT
    # ==================================================

    def _build_memory_prompt(
        self,
        user_input,
        best_paragraph,
        best_sent,
        best_para_score,
        best_sent_score,
        user_id
    ):

        misty_context = self.get_user_context(user_id)

        user_history = self.load_user_conversation_history(user_id)

        # Check if actual history exists — these are the guards
        # against hallucinated "like I mentioned before" references.

        has_misty_context = (
            misty_context
            != "No prior stories shared with this user yet."
        )

        has_user_history = user_history is not None

        try:
            best_index = best_paragraph.index(best_sent)
        except ValueError:
            best_index = 0

        if best_sent_score > 0.30:

            context_indices = [
                i for i in (best_index - 1, best_index)
                if 0 <= i < len(best_paragraph)
            ]

            current_memory = " ".join(
                best_paragraph[i] for i in context_indices
            )

            instruction = (
                "Mention just this one detail briefly, like it crossed "
                "your mind naturally."
            )

            length_hint = "30-60 words."

        elif best_sent_score > 0.15:

            lead_indices = list(
                range(min(2, len(best_paragraph)))
            )

            if best_index not in lead_indices:
                lead_indices.append(best_index)

            current_memory = " ".join(
                best_paragraph[i]
                for i in sorted(set(lead_indices))
            )

            instruction = (
                "Center on the main point. Add at most one extra detail "
                "only if it flows naturally."
            )

            length_hint = "40-80 words."

        else:

            current_memory = best_sent

            instruction = (
                "Reference it lightly as a passing thought."
            )

            length_hint = "30-50 words."

        # Build context blocks only when real data exists.
        # If empty, inject a strict NO-REFERENCE instruction instead
        # so the model cannot hallucinate shared history.

        if has_misty_context:

            misty_context_block = (
                f"Stories you have already shared with this user "
                f"(ONLY reference these — nothing else):\n"
                f"{misty_context}\n"
            )

            callback_rule = (
                "- You may briefly reference a story ONLY if it is "
                "literally listed in 'Stories already shared' above.\n"
                "- If the current memory is NOT in that list, you are "
                "sharing it FOR THE FIRST TIME. Do not say 'like I "
                "mentioned', 'as I told you', 'you know how I said', "
                "or any similar phrase."
            )

        else:

            misty_context_block = (
                "Stories already shared with this user: NONE.\n"
            )

            callback_rule = (
                "- !! IMPORTANT !! You have NEVER shared any personal "
                "story with this user before. This is your FIRST "
                "conversation with them.\n"
                "- NEVER say 'like I mentioned', 'as I told you before',"
                " 'you know how I said', 'like I said', or ANY phrase "
                "that implies a previous conversation happened.\n"
                "- Share the Current Memory as something NEW you are "
                "telling them for the first time right now."
            )

        if has_user_history:

            user_history_block = (
                f"What this user has shared about themselves "
                f"in past conversations:\n{user_history}\n"
            )

            know_user_rule = (
                "- You already know this person's situation from past "
                "conversations. Use that context naturally — do not ask "
                "them to repeat what they have already told you."
            )

        else:

            user_history_block = ""

            know_user_rule = (
                "- This is the first time you are speaking with this "
                "person. Do not reference any past conversations."
            )

        first_time_label = (
            ""
            if has_misty_context
            else
            "(You are sharing this for the FIRST TIME with this user. "
            "Do NOT use any phrase like 'like I mentioned before'.)\n"
        )

        return f"""
You are Misty, talking casually like a real person.

{misty_context_block}
{user_history_block}
Current Memory — share this naturally:
{first_time_label}{current_memory}

User just said:
{user_input}

HOW TO RESPOND:

{callback_rule}
- Talk warm and casual, like a real friend.
- Pick ONE moment from Current Memory and mention it briefly.
- Never list events in order or retell the whole story.
{know_user_rule}
- React to the user's feeling first, then bring in your own moment
  in 1-2 sentences max.
- Do not say "I have a memory" or "according to my memory".

{instruction}

Length:
{length_hint}
"""

    # ==================================================
    # BASE PROMPT
    # ==================================================

    def _build_base_prompt(
        self,
        user_id
    ):

        misty_context = self.get_user_context(user_id)

        user_history = self.load_user_conversation_history(user_id)

        has_history = user_history is not None

        has_misty_context = (
            misty_context
            != "No prior stories shared with this user yet."
        )

        if has_history or has_misty_context:

            user_history_block = (
                f"\nWhat this user has shared about themselves "
                f"in past conversations:\n{user_history}\n"
                if has_history
                else ""
            )

            misty_context_block = (
                f"\nStories Misty has already shared with this user "
                f"(EXACT list — nothing outside this was ever mentioned):\n"
                f"{misty_context}\n"
                if has_misty_context
                else (
                    "\nStories Misty has already shared: NONE.\n"
                    "You have NOT shared any personal stories yet.\n"
                )
            )

            context_block = f"""
YOU ARE TALKING TO SOMEONE YOU KNOW.
{user_history_block}{misty_context_block}

STRICT RULES FOR REFERENCING PAST CONVERSATIONS:

1. Before saying "like I mentioned", "as I told you", "you know how
   I said", or ANY similar phrase — you MUST verify that the exact
   topic appears word-for-word in the "Stories Misty has already
   shared" list above.

2. If the topic is NOT in that list — even if it feels related —
   do NOT say you mentioned it before. Say it as if for the first
   time, with no callback phrase.

3. If the user references something from a past conversation:
   - Check the context above first.
   - If it's there: acknowledge it naturally.
     Example: "Yeah, I remember — you told me about your grandfather."
   - If it's NOT there: say you don't recall that specific thing,
     but stay warm and invite them to share again.

4. If the user asks "do you remember...":
   - YES only if it is literally in the context above.
   - NO if it is not — never guess or assume.

5. NEVER invent shared history. NEVER say "like I mentioned before"
   about something not in the stories list above."""

        else:

            context_block = (
                "You have never spoken to this user before.\n"
                "This is a completely fresh start.\n"
                "Do NOT reference any past conversations or stories.\n"
                "Do NOT say 'like I mentioned', 'as I told you', or\n"
                "any phrase implying prior conversation."
            )

        return f"""
You are Misty — warm, empathetic, real.

{context_block}

Rules:
- Be conversational and human. No robotic tone.
- Focus on the user and what they are feeling right now.
- Do not invent personal experiences.
- Keep responses between 30 and 60 words.
"""