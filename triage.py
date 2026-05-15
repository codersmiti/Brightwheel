from openai import OpenAI
import os
import json
import re
import time
from dotenv import load_dotenv
from pii_scrubber import scrub_pii

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Keep secrets in .env or deployment environment variables. Never commit keys.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
USE_AI = os.getenv("USE_AI", "true").lower() == "true"
DEBUG_LLM = os.getenv("DEBUG_LLM", "false").lower() == "true"
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    timeout=LLM_TIMEOUT_SECONDS,
) if GROQ_API_KEY else None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLA_BY_PRIORITY = {
    "Critical": "Respond within 1 hour",
    "High": "Respond same business day",
    "Medium": "Respond within 24 hours",
    "Low": "Respond within 48 hours",
}

DEFAULT_OWNER_BY_CATEGORY = {
    "Setup & Onboarding": "Onboarding Specialist",
    "Technical Issue": "Technical Support",
    "Billing & Accounts": "Billing Team",
    "Privacy & Security": "Security Team",
    "Feature Question": "Onboarding Specialist",
    "Account Management": "Account Manager",
    "Urgent Escalation": "Escalation Manager",
    "Wrong Team": None,
}

SUPPORTING_TEAM_BY_CATEGORY = {
    "Setup & Onboarding": "Onboarding Specialist",
    "Technical Issue": "Technical Support",
    "Billing & Accounts": "Billing Team",
    "Privacy & Security": "Security Team",
    "Feature Question": "Onboarding Specialist",
    "Account Management": "Account Manager",
    "Urgent Escalation": "Onboarding Specialist",
    "Wrong Team": None,
}

VALID_CATEGORIES = set(DEFAULT_OWNER_BY_CATEGORY.keys())
VALID_PRIORITIES = set(SLA_BY_PRIORITY.keys())

# ---------------------------------------------------------------------------
# Prompt / Taxonomy
# ---------------------------------------------------------------------------

TAXONOMY = """
You triage inbound messages for Brightwheel's onboarding team.

Return ONE JSON object with exactly these fields:
category, secondary_category, priority, sla, owner, supporting_team,
escalation_type, confidence, needs_human_review, human_review_reason,
safe_mode, draft_reply, issues_detected, weekly_impact.

CATEGORIES:
- Setup & Onboarding   : adding teachers, rosters, classrooms, setup how-to questions
- Technical Issue      : login errors, invitations, tablets, QR codes, app crashes, trouble logging in
- Billing & Accounts   : invoices, charges, payments, plan upgrades, billing questions
- Privacy & Security   : wrong child visible, unauthorized access, child data exposure
- Wrong Team           : sales/demo/pricing prospects, job applications, unrelated
- Feature Question     : export, customize, or enable non-blocking features
- Account Management   : transfer admin access or account ownership
- Urgent Escalation    : no onboarding contact + no setup before imminent school launch

PRIORITY:
- Critical : data breach, total system failure, school opens <72h with blocking issue
- High     : billing dispute, 2+ distinct issues, deadline this week, time-sensitive account change
- Medium   : single non-blocking issue, one device/user affected, no deadline
- Low      : simple setup how-to, general guidance, wrong-team emails, feature/export/customization questions with no urgency

SCOPE RULE for Technical Issues:
- One device or one user, others fine -> Medium
- All devices / all users / blocking launch -> High or Critical

CRITICAL ROUTING RULES — NEVER VIOLATE:
- "Billing & Accounts" category MUST route to "Billing Team" — never "Account Manager"
- "Account Management" category MUST route to "Account Manager" — never "Billing Team"
- "Billing & Accounts" vs "Account Management" distinction:
  * Billing & Accounts = money issues (charges, invoices, payments, double charge, amount wrong)
  * Account Management = access issues (transfer admin, ownership change, director leaving)
- A double charge or duplicate payment is ALWAYS Billing & Accounts, never Account Management
- "Technical Issue" MUST route to "Technical Support" unless priority is Critical
- "Setup & Onboarding" MUST route to "Onboarding Specialist"
- "Feature Question" MUST route to "Onboarding Specialist"
- "Wrong Team" sales -> "Sales Team", jobs -> "HR Team"
- Critical priority -> owner MUST be "Escalation Manager" always

ROUTING:
- Critical -> owner must be Escalation Manager
- If owner is Escalation Manager, supporting_team must be the functional team:
  MAKE SURE Technical Issue -> Technical Support
  Privacy & Security -> Security Team
  Billing & Accounts -> Billing Team
  Setup & Onboarding / Urgent Escalation -> Onboarding Specialist
- Wrong Team sales/demo/pricing -> Sales Team, supporting_team null
- Wrong Team job application -> HR Team, supporting_team null

TRIAGE RULES:
1. category = issue type, never urgency level.
2. Urgent technical failure before launch -> Technical Issue + Critical.
3. No onboarding contact + no setup before launch -> Urgent Escalation + Critical.
4. Any privacy/child data exposure -> Privacy & Security + Critical always.
5. Billing amount mismatch or duplicate charge -> Billing & Accounts + High minimum.
6. Two or more distinct problems -> High minimum + populate secondary_category.
7. Vague, detail-free message -> Medium + confidence 0.2-0.4 + needs_human_review true.
8. Sales lead or job application -> Wrong Team + Low.
9. Critical messages -> needs_human_review true + safe_mode "Draft only - do not auto-send".
10. Simple how-to/setup/feature questions should be Low unless deadline, blocker, or multi-issue context is stated.

DRAFT REPLY RULES:
- Max 4 sentences.
- Direct, operational, and human.
- Acknowledge the issue, state routing/escalation, and give correct SLA.
- Do NOT invent product navigation steps, UI instructions, troubleshooting steps, or exact Brightwheel workflows.
- Do NOT say the issue is solved.
- For Critical issues, mention follow-up within 1 hour.
- Sign off exactly as: Brightwheel Onboarding Team
- If the message is a follow-up (contains "following up", "any update", "haven't heard back"), acknowledge the wait briefly and give a specific timeframe matching the SLA. Do NOT say "2 business days".

Return ONLY valid JSON. No markdown fences. No explanation outside the JSON.
"""

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _debug(*args):
    """Print only when DEBUG_LLM=true."""
    if DEBUG_LLM:
        print(*args, flush=True)


def _contains(text: str, pattern: str) -> bool:
    return bool(re.search(pattern, text, flags=re.IGNORECASE))


def detect_issues(message: str) -> list:
    """Detects issue signals for explainability and fallback logic."""
    text = message.lower()
    patterns = [
        (r"wrong child|another family|another child|not hers|not his|privacy|unauthorized|child data", "Possible privacy/security exposure"),
        (r"cannot log in|can't log in|unable to log in|login|log in", "Login/access issue"),
        (r"invite|invitation|families.*received", "Invitation delivery issue"),
        (r"tablet|stuck on loading|loading screen|cannot check in|cannot message|crash|qr code|spinning", "App/device technical failure"),
        (r"invoice|billing|charge|payment|amount doesn't match|amount does not match|charged twice|double charge", "Billing/account question"),
        (r"add teachers|teacher|roster|classroom|setup|ratio|families yet|staff", "Setup/onboarding question"),
        (r"export|customize|enable|feature|personalize", "Feature/how-to question"),
        (r"demo|pricing", "Sales inquiry"),
        (r"resume|job|interview|applying|teacher assistant", "Job/careers inquiry"),
        (r"admin access|transfer ownership|account owner|director", "Account ownership/admin request"),
        (r"opens tomorrow|open in 3 days|school starts monday|starts monday|school opens|open tomorrow|starts in two days|open on friday|orientation tomorrow", "Near-term launch deadline"),
        (r"no one has contacted|haven't heard|have not heard|still have not been contacted|onboarding rep|no rep", "No onboarding contact received"),
        (r"not set up|no setup|nothing is set up|do not have the app set up|don't have the app set up|not added staff|not added.*classrooms|not added.*families", "No setup completed"),
    ]
    found = [label for pattern, label in patterns if re.search(pattern, text)]
    return list(dict.fromkeys(found))


def infer_secondary(text: str, primary: str) -> str | None:
    """Infers a secondary category for fallback only."""
    candidates = []
    if _contains(text, r"invoice|billing|charge|payment|amount doesn't match|amount does not match|charged twice|double charge"):
        candidates.append("Billing & Accounts")
    if _contains(text, r"cannot log in|login|tablet|loading|crash|invite|invitation|cannot check in|cannot message|qr code|spinning"):
        candidates.append("Technical Issue")
    if _contains(text, r"classroom|ratio|add teachers|roster|setup|go live|go-live|staff|families"):
        candidates.append("Setup & Onboarding")
    if _contains(text, r"admin access|transfer ownership|account owner"):
        candidates.append("Account Management")
    if _contains(text, r"export|customize|enable|feature|personalize"):
        candidates.append("Feature Question")
    return next((c for c in candidates if c != primary), None)


def get_escalation_type(category: str, priority: str, text: str) -> str:
    """Human-readable reason for escalation."""
    if category == "Privacy & Security":
        return "Privacy/Security"
    if category == "Urgent Escalation":
        return "Failed Onboarding Ownership"
    if priority == "Critical" and _contains(text, r"school opens|opens tomorrow|open in 3 days|starts monday|school starts|starts in two days|open on friday|orientation tomorrow"):
        return "Launch Blocker"
    if priority == "Critical" and _contains(text, r"all .*tablet|every single|cannot check in|cannot message|system down|none of our parents"):
        return "Technical Outage"
    if priority in ("High", "Critical") and _contains(text, r"invoice|billing|charge|payment|amount|double charge|charged twice"):
        return "Billing Dispute"
    if _contains(text, r"admin access|transfer ownership|account owner|director.*friday|leaves.*friday"):
        return "Account Ownership"
    return "None"


def get_supporting_team(category: str, priority: str, owner: str) -> str | None:
    """Keeps critical owner vs functional team consistent."""
    if owner in ("Sales Team", "HR Team"):
        return None
    if priority == "Critical":
        return SUPPORTING_TEAM_BY_CATEGORY.get(category) or "Onboarding Specialist"
    return SUPPORTING_TEAM_BY_CATEGORY.get(category)

# ---------------------------------------------------------------------------
# Draft reply generation
# ---------------------------------------------------------------------------

def make_reply(category: str, priority: str, owner: str,
               sender_name: str = "there", supporting_team: str | None = None) -> str:
    """Creates safe operational draft replies without hallucinated product instructions."""
    g = f"Hi {sender_name},"

    if category == "Privacy & Security":
        return (
            f"{g} thank you for flagging this. We take potential child data exposure very seriously and have escalated this immediately to our Security Team for urgent review. "
            "You will hear back within 1 hour with next steps.\n\n"
            "Brightwheel Onboarding Team"
        )

    if category == "Wrong Team" and owner == "Sales Team":
        return (
            f"{g} thanks for your interest in Brightwheel. This inbox handles existing customer onboarding, so we are routing your note to our Sales Team for pricing and demo next steps.\n\n"
            "Brightwheel Onboarding Team"
        )

    if category == "Wrong Team" and owner == "HR Team":
        return (
            f"{g} thanks for reaching out. This inbox is for customer onboarding, so we are redirecting your note to the appropriate HR or careers channel.\n\n"
            "Brightwheel Onboarding Team"
        )

    if priority == "Critical":
        support = f" with support from {supporting_team}" if supporting_team and supporting_team != owner else ""
        return (
            f"{g} thanks for flagging this. This is being escalated immediately to {owner}{support} because it appears to be blocking operations or your school launch. "
            "Expect a follow-up within 1 hour.\n\n"
            "Brightwheel Onboarding Team"
        )

    if category == "Billing & Accounts":
        return (
            f"{g} thanks for reaching out. We are routing this to our Billing Team to review the invoice or account details. "
            "They will follow up by end of business today.\n\n"
            "Brightwheel Onboarding Team"
        )

    if category == "Account Management":
        return (
            f"{g} thanks for the heads-up. We are routing this to an Account Manager to handle the ownership or admin access change. "
            "They will follow up by end of business today.\n\n"
            "Brightwheel Onboarding Team"
        )

    if category == "Technical Issue":
        return (
            f"{g} thanks for reporting this. We are routing it to Technical Support to investigate and help resolve the issue. "
            f"Expected response: {SLA_BY_PRIORITY[priority]}.\n\n"
            "Brightwheel Onboarding Team"
        )

    if category in ("Setup & Onboarding", "Feature Question") and priority == "Low":
        return (
            f"{g} thanks for reaching out. We are routing this to an Onboarding Specialist who can help with your setup or product question. "
            "They will respond within 48 hours.\n\n"
            "Brightwheel Onboarding Team"
        )

    if priority == "Medium":
        return (
            f"{g} thanks for reaching out. Could you share a few more details about what you are experiencing and where you are getting stuck? "
            "That will help us route this to the right person quickly.\n\n"
            "Brightwheel Onboarding Team"
        )

    return (
        f"{g} thanks for reaching out. We are routing this to the right teammate to assist you. "
        f"Expected response: {SLA_BY_PRIORITY[priority]}.\n\n"
        "Brightwheel Onboarding Team"
    )

# ---------------------------------------------------------------------------
# Rule-based triage: used ONLY when Groq fails or USE_AI=false
# ---------------------------------------------------------------------------

def rule_based_triage(message_body: str, sender_name: str = "there", error: str | None = None) -> dict:
    """Deterministic fallback when Groq is unavailable or disabled."""
    text = message_body.lower()
    issues = detect_issues(message_body)

    category = "Setup & Onboarding"
    priority = "Medium"
    owner = "Onboarding Specialist"
    confidence = 0.65

    if _contains(text, r"resume|job|interview|applying|teacher assistant") and not _contains(text, r"onboarding|billing|invoice|login|setup|contract"):
        category, priority, owner, confidence = "Wrong Team", "Low", "HR Team", 0.92
    elif _contains(text, r"demo|pricing") and not _contains(text, r"signed|contract|invoice|billing|customer"):
        category, priority, owner, confidence = "Wrong Team", "Low", "Sales Team", 0.92
    elif _contains(text, r"wrong child|another family|another child|not hers|not his|privacy|unauthorized|child data"):
        category, priority, owner, confidence = "Privacy & Security", "Critical", "Escalation Manager", 0.93
    elif (_contains(text, r"school starts monday|opens tomorrow|open in 3 days|school opens|open tomorrow|starts in two days")
          and _contains(text, r"no one has contacted|haven't heard|have not heard|still have not been contacted|not set up|no setup|nothing is set up|not added staff|not added.*classrooms|not added.*families")):
        category, priority, owner, confidence = "Urgent Escalation", "Critical", "Escalation Manager", 0.90
    elif (_contains(text, r"opens tomorrow|open in 3 days|school opens|school starts|starts monday|open tomorrow|open on friday|orientation tomorrow")
          and _contains(text, r"cannot log in|can't log in|unable to log in|stuck on loading|cannot check in|cannot message|nothing is working|invite|invitation|qr code|spinning|none of our parents")):
        category, priority, owner, confidence = "Technical Issue", "Critical", "Escalation Manager", 0.90
    elif _contains(text, r"all .*tablet|every single .*tablet|cannot check in|cannot message|system down|none of our parents"):
        category, priority, owner, confidence = "Technical Issue", "Critical", "Escalation Manager", 0.88
    elif _contains(text, r"director.*friday|leaves.*friday|departure is friday|transfer ownership|transfer admin access|account owner"):
        category, priority, owner, confidence = "Account Management", "High", "Account Manager", 0.86
    elif _contains(text, r"invoice|billing|charge|payment|amount doesn't match|amount does not match|charged twice|double charge"):
        priority = "High" if _contains(text, r"amount doesn't match|amount does not match|charged twice|double charge|dispute|contract") else "Medium"
        confidence = 0.84 if priority == "High" else 0.82
        category, owner = "Billing & Accounts", "Billing Team"
        if _contains(text, r"classroom|ratio|setup|go live|go-live|roster|upload"):
            category, owner = "Setup & Onboarding", "Onboarding Specialist"
    elif _contains(text, r"cannot log in|can't log in|unable to log in|stuck on loading|crash|tablet|invite|invitation|qr code|spinning"):
        single_scope = _contains(text, r"one of our|one parent|one teacher|director's ipad|her ipad|his ipad|one device|one staff|one user|my ipad")
        category = "Technical Issue"
        priority = "Medium" if single_scope else "High"
        owner = "Technical Support"
        confidence = 0.78 if single_scope else 0.80
    elif _contains(text, r"export|customize|enable|feature|personalize"):
        category, priority, owner, confidence = "Feature Question", "Low", "Onboarding Specialist", 0.78
    elif _contains(text, r"add teachers|teaching staff|classroom|roster|setup"):
        category, priority, owner, confidence = "Setup & Onboarding", "Low", "Onboarding Specialist", 0.82
    elif _contains(text, r"having some problems|having some issues|need assistance|earliest convenience"):
        category, priority, owner, confidence = "Setup & Onboarding", "Medium", "Onboarding Specialist", 0.35

    secondary = infer_secondary(text, category)
    if len(issues) >= 3 and priority not in ("Critical", "High"):
        priority = "High"
    if priority == "Critical":
        owner = "Escalation Manager"

    supporting_team = get_supporting_team(category, priority, owner)
    esc_type = get_escalation_type(category, priority, text)
    needs_review = priority == "Critical" or confidence < 0.4 or bool(error)

    if error:
        review_reason = "AI unavailable or invalid response — rule-based safety layer used. Manual review recommended."
    elif priority == "Critical":
        review_reason = "Critical priority: requires immediate human oversight before action."
    elif confidence < 0.4:
        review_reason = "Low confidence — message too vague to route automatically."
    else:
        review_reason = None

    return {
        "category": category,
        "secondary_category": secondary,
        "priority": priority,
        "sla": SLA_BY_PRIORITY[priority],
        "owner": owner,
        "supporting_team": supporting_team,
        "escalation_type": esc_type,
        "confidence": confidence,
        "needs_human_review": needs_review,
        "human_review_reason": review_reason,
        "safe_mode": "Draft only - do not auto-send" if needs_review else "Auto-route approved",
        "draft_reply": make_reply(category, priority, owner, sender_name, supporting_team),
        "issues_detected": issues,
        "weekly_impact": None,
        "pii_masked": True,
        "original_sender": sender_name,
        "automation_mode": "Rules-assisted fallback" if error else "Rules-only",
        "llm_used": False,
        **({"debug_error": error} if error and DEBUG_LLM else {}),
    }

# ---------------------------------------------------------------------------
# LLM call and JSON parsing
# ---------------------------------------------------------------------------

def call_llm(prompt: str) -> str:
    """Calls Groq with a short timeout for live-demo latency."""
    if not client:
        raise RuntimeError("GROQ_API_KEY not set")

    _debug("[LLM] Calling Groq", {"model": MODEL_NAME, "timeout": LLM_TIMEOUT_SECONDS})
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise Brightwheel onboarding triage assistant. "
                    "Return valid JSON only. Never contradict the SLA in the draft reply. "
                    "If priority is High, reply timeframe must be same business day. "
                    "If priority is Critical, reply timeframe must be within 1 hour. "
                    "If priority is Medium, reply timeframe must be within 24 hours. "
                    "If priority is Low, reply timeframe must be within 48 hours."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=1200,
    )
    _debug("[LLM] Groq response received")
    return response.choices[0].message.content


def clean_json(raw: str) -> dict:
    """Extracts JSON safely from LLM output."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1].replace("json", "", 1).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(raw[start:end + 1])

# ---------------------------------------------------------------------------
# Post-processing guardrails: NO rule_based_triage call here
# ---------------------------------------------------------------------------

def _is_hallucinated_instruction(draft: str) -> bool:
    """Flags product-step hallucinations in draft replies."""
    return _contains(
        draft,
        r"click|navigate|go to|select|tab|button|dashboard|settings|log in to your|follow the prompts|classrooms tab|upload the file"
    )


def post_process(result: dict, message_body: str, sender_name: str = "there") -> dict:
    """
    Groq-first post-processing.

    Important: this function does NOT call rule_based_triage().
    It only normalizes the LLM output, enforces routing consistency,
    and repairs unsafe draft replies. Rule-based triage is used only if
    Groq fails, JSON parsing fails, or USE_AI=false.
    """
    text = message_body.lower()

    category = result.get("category") if result.get("category") in VALID_CATEGORIES else "Setup & Onboarding"
    priority = result.get("priority") if result.get("priority") in VALID_PRIORITIES else "Medium"

    # Small deterministic safety corrections without using the full fallback engine.
    if category == "Privacy & Security":
        priority = "Critical"
    if priority == "Critical":
        owner = "Escalation Manager"
    elif category == "Wrong Team":
        owner = result.get("owner") if result.get("owner") in ("Sales Team", "HR Team") else "Sales Team"
    else:
        owner = result.get("owner") or DEFAULT_OWNER_BY_CATEGORY.get(category) or "Onboarding Specialist"

    supporting_team = get_supporting_team(category, priority, owner)
    sla = SLA_BY_PRIORITY[priority]

    secondary_category = result.get("secondary_category")
    if secondary_category not in VALID_CATEGORIES:
        secondary_category = None

    confidence = result.get("confidence", 0.75)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.75

    needs_human_review = bool(result.get("needs_human_review", False))
    human_review_reason = result.get("human_review_reason")
    if priority == "Critical":
        needs_human_review = True
        human_review_reason = human_review_reason or "Critical priority: requires immediate human oversight before action."

    safe_mode = "Draft only - do not auto-send" if needs_human_review else "Auto-route approved"

    issues_detected = result.get("issues_detected")
    if not isinstance(issues_detected, list) or not issues_detected:
        issues_detected = detect_issues(message_body)

    draft = result.get("draft_reply", "") or ""
    bad_reply = (
        not draft.strip()
        or len(draft.split()) > 120
        or _is_hallucinated_instruction(draft)
        or (priority == "Critical" and "1 hour" not in draft.lower())
        or (priority == "Critical" and "24 hours" in draft.lower())
        or (priority == "High" and ("2 business days" in draft.lower() or "48 hours" in draft.lower()))
    )

    if bad_reply:
        draft = make_reply(category, priority, owner, sender_name, supporting_team)

    return {
        "category": category,
        "secondary_category": secondary_category,
        "priority": priority,
        "sla": sla,
        "owner": owner,
        "supporting_team": supporting_team,
        "escalation_type": result.get("escalation_type") or get_escalation_type(category, priority, text),
        "confidence": confidence,
        "needs_human_review": needs_human_review,
        "human_review_reason": human_review_reason,
        "safe_mode": safe_mode,
        "draft_reply": draft,
        "issues_detected": issues_detected,
        "weekly_impact": None,
        "pii_masked": True,
        "original_sender": sender_name,
        "automation_mode": "AI-assisted",
        "llm_used": True,
        "model": MODEL_NAME,
    }

# ---------------------------------------------------------------------------
# Follow-up detection helper
# ---------------------------------------------------------------------------

def detect_followup(message_body: str, sender_email: str, messages: list) -> dict | None:
    """
    Detects if a message is a follow-up to a previous one.
    Matches follow-up language AND sender email against known messages.
    Note: works for dataset messages loaded at startup.
    Production would require persistent storage for new messages.
    """
    followup_patterns = r"following up|follow up|just checking|any update|heard back|haven't heard|no response|still waiting|checking in|following up on"

    if not _contains(message_body, followup_patterns):
        return None

    previous = [m for m in messages if str(m.get("sender_email", "")).strip().lower() == sender_email.strip().lower()]

    if not previous:
        return None

    return {
        "is_followup": True,
        "original_message_id": str(previous[0].get("message_id", "")),
        "original_subject": str(previous[0].get("subject", ""))
    }

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def triage_message(message_body: str, sender_name: str = "there", sender_email: str = "") -> dict:
    """Main function called by Flask: raw message -> structured triage dict."""
    start = time.perf_counter()

    _debug("[TRIAGE] start", {"USE_AI": USE_AI, "key_exists": bool(GROQ_API_KEY), "model": MODEL_NAME})

    if not USE_AI:
        result = rule_based_triage(message_body, sender_name, error=None)
        result["latency_ms"] = round((time.perf_counter() - start) * 1000)
        return result

    scrubbed = scrub_pii(message_body)
    prompt = f"{TAXONOMY}\n\nMessage:\n{scrubbed}\n\nReturn ONLY valid JSON."

    try:
        raw = call_llm(prompt)
        _debug("[LLM] raw response:", raw)
        parsed = clean_json(raw)
        _debug("[LLM] parsed JSON keys:", list(parsed.keys()))
        result = post_process(parsed, message_body, sender_name)
    except Exception as exc:
        _debug("[LLM] failed; using fallback:", repr(exc))
        result = rule_based_triage(message_body, sender_name, error=str(exc))
        if DEBUG_LLM:
            result["debug_error"] = str(exc)
            if "raw" in locals():
                result["raw_llm_response"] = raw

    result["latency_ms"] = round((time.perf_counter() - start) * 1000)
    return result

# ---------------------------------------------------------------------------
# Weekly impact calculator
# ---------------------------------------------------------------------------

def calculate_weekly_impact(total_messages: int = 200) -> dict:
    """Estimates weekly capacity saved based on Brightwheel's prompt numbers."""
    manual_minutes = total_messages * 4
    ai_minutes = (total_messages * 20) / 60
    saved_minutes = manual_minutes - ai_minutes
    return {
        "total_messages_per_week": total_messages,
        "manual_triage_hours": round(manual_minutes / 60, 1),
        "ai_triage_hours": round(ai_minutes / 60, 1),
        "hours_saved_per_week": round(saved_minutes / 60, 1),
        "efficiency_gain_percent": round((saved_minutes / manual_minutes) * 100, 1),
    }