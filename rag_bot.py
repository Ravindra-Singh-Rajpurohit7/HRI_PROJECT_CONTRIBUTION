import os
import time
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI

from memory_retriever import MemoryRetriever
from prompts import PromptManager
from database import SessionLocal, init_db
from models import User
from logger import MistyLogger
# Yeh add karo sabse upar — imports ke baad
from dotenv import load_dotenv
load_dotenv()

# Yeh pehle se hai, koi change nahi:
API_KEY = os.environ.get("GOOGLE_API_KEY")
DB_PATH = "./chroma_db"

if not API_KEY:
    print(
        "\nERROR: GOOGLE_API_KEY environment variable is not set."
    )
    print(
        "Set it before running:\n"
        "  export GOOGLE_API_KEY=your_key_here   (Linux/Mac)\n"
        "  setx GOOGLE_API_KEY your_key_here      (Windows)\n"
    )
    exit()

# -------------------------
# INIT DB (creates missing tables)
# -------------------------
init_db()

# -------------------------
# USER SELECTION
# -------------------------
db = SessionLocal()

try:

    user_name = input(
        "Enter User Name: "
    )

    user = (
        db.query(User)
        .filter(User.name == user_name)
        .first()
    )

    if not user:
        print(
            f"\nUser '{user_name}' not found."
        )
        exit()

    USER_ID = user.user_id

finally:
    db.close()

# unique session id
SESSION_ID = int(
    datetime.now().timestamp()
)

# -------------------------
# MODELS
# -------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.8
)

decider_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0
)

# -------------------------
# RETRIEVER + PROMPTS
# -------------------------
retriever = MemoryRetriever(
    DB_PATH,
    API_KEY
)

prompt_manager = PromptManager(
    decider_llm,
    retriever
)

# -------------------------
# LOGGER
# -------------------------
logger = MistyLogger(
    user_id=USER_ID,
    session_id=SESSION_ID
)


# -------------------------
# SAFE INVOKE
# -------------------------
def safe_invoke(model, prompt):

    for attempt in range(5):

        try:
            return model.invoke(prompt)

        except Exception as e:
            print(
                f"\nGemini busy. Retry {attempt + 1}/5"
            )
            print(e)
            time.sleep(5)

    return None


# -------------------------
# START CHAT
# -------------------------
print(f"\nLogged in as: {user_name}")
print(f"User ID     : {USER_ID}")
print(f"Session ID  : {SESSION_ID}")
print("\nMisty is ready.")
print("Type 'exit' to quit.\n")

last_user_message = ""
last_bot_response = ""


# -------------------------
# SESSION SAVE FUNCTION
# Called on both normal exit and Ctrl+C
# -------------------------
def save_session():

    if not prompt_manager.chat_history:
        print("\n[No conversation to save.]")
        return

    print("\n[Saving session context...]")

    user_summary, topics = (
        prompt_manager.generate_user_summary(
            prompt_manager.chat_history
        )
    )

    prompt_manager.save_user_context(
        user_id=USER_ID,
        session_id=SESSION_ID,
        user_summary=user_summary,
        topics=topics,
        last_user_message=last_user_message,
        last_bot_response=last_bot_response
    )

    logger.save(
        chat_history=prompt_manager.chat_history
    )

    print("[Session saved. See you next time!]")


# -------------------------
# CHAT LOOP
# try/finally ensures save runs on exit, Ctrl+C, or crash
# -------------------------
try:

    while True:

        query = input("You: ")

        if query.lower() == "exit":
            print("\nGoodbye!")
            break

        try:

            # LOG: turn start
            logger.start_turn(query)

            # ------------------------------------------------
            # STEP 1: retrieve memory
            # ------------------------------------------------
            (
                memory_id,
                best_para,
                best_sent,
                best_sent_score,
                best_para_score,
                memory_scores,
                best_memory
            ) = prompt_manager._retrieve_memory(query)

            # ------------------------------------------------
            # STEP 2: check disclosures
            # ------------------------------------------------
            previous_disclosures = (
                prompt_manager.get_user_disclosures(USER_ID)
            )

            already_disclosed = (
                memory_id in previous_disclosures
            )

            # ------------------------------------------------
            # LOG: retrieval + disclosures
            # ------------------------------------------------
            logger.log_retrieval(
                memory_id,
                best_para,
                best_sent,
                best_sent_score,
                best_para_score,
                already_disclosed
            )

            logger.log_disclosures(previous_disclosures)

            user_context = prompt_manager.get_user_context(USER_ID)
            logger.log_user_context(user_context)

            # ------------------------------------------------
            # STEP 3: logger only — no manual decider call here
            # prompt_generation() handles decider internally.
            # We log the outcome after it runs (Step 4).
            # ------------------------------------------------

            # ------------------------------------------------
            # STEP 4: prompt generation
            # ------------------------------------------------
            result = prompt_manager.prompt_generation(
                query,
                USER_ID,
                SESSION_ID
            )

            prompt_type = (
                "memory" if result["memory_used"] else "base"
            )

            # Log decider outcome based on result
            if result["memory_used"]:
                logger.log_decider(
                    " ".join(best_para),
                    "true"
                )
            elif already_disclosed:
                logger.log_decider(
                    " ".join(best_para),
                    "skipped",
                    reason="Memory already disclosed."
                )
            elif best_para_score < 0.20:
                logger.log_decider(
                    " ".join(best_para),
                    "skipped",
                    reason=(
                        f"Para score {round(best_para_score, 4)} "
                        f"below threshold 0.20."
                    )
                )
            else:
                logger.log_decider(
                    " ".join(best_para),
                    "false"
                )

            final_prompt = f"""
{result['prompt']}
User:
{query}
"""

            if result["memory_used"]:
                logger.log_disclosure_save(
                    result["memory_id"],
                    saved=not already_disclosed
                )

            logger.log_prompt(final_prompt, prompt_type)

            # ------------------------------------------------
            # STEP 5: LLM call
            # ------------------------------------------------
            response = safe_invoke(llm, final_prompt)

            if response is None:
                print(
                    "\nMisty: Sorry, I'm having trouble right now."
                )
                logger.log_response(
                    "[ERROR: LLM returned None after 5 retries]"
                )
                continue

            answer = response.content

            print(f"\nMisty: {answer}")

            # ------------------------------------------------
            # TRACK last exchange for context saving
            # ------------------------------------------------
            last_user_message = query
            last_bot_response = answer

            # ------------------------------------------------
            # UPDATE CHAT HISTORY
            # ------------------------------------------------
            prompt_manager.chat_history.append(
                f"User: {query}"
            )
            prompt_manager.chat_history.append(
                f"Assistant: {answer}"
            )

            # ------------------------------------------------
            # LOG: response + history
            # ------------------------------------------------
            logger.log_response(answer)
            logger.log_chat_history(prompt_manager.chat_history)

            if result["memory_used"]:
                print(
                    f"\n[MEMORY DISCLOSED] -> {result['memory_id']}"
                )

        except KeyboardInterrupt:

            # Let outer finally handle the save
            raise

        except Exception as e:

            print("\nUnexpected Error:")
            print(e)

            if hasattr(logger, "current_turn"):
                logger.log_response(f"[EXCEPTION: {str(e)}]")

finally:

    # -------------------------
    # SESSION END
    # Runs on: normal exit, Ctrl+C, or any unhandled crash
    # -------------------------
    save_session()