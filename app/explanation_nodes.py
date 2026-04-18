# Explanation Nodes
# These helpers turn model outputs into short plain-English explanations for the user.
# Groq is used when configured, with a local fallback message if Groq is unavailable.

import json
import urllib.error
import urllib.request

from config import (
    ADVISORY_DESCRIPTIONS,
    ADVISORY_LABELS,
    GRADE_LABELS,
    GROQ_API_URL,
    GROQ_MODEL,
    LABELS,
    RAW_NUMERIC_COLUMNS,
    config_value,
)


# Return the Groq settings used by the explanation step.
def explanation_settings_from(secrets):
    api_key = config_value(secrets, "GROQ_API_KEY")
    return {
        "api_key": api_key,
        "api_url": config_value(secrets, "GROQ_API_URL") or GROQ_API_URL,
        "model": config_value(secrets, "GROQ_MODEL") or GROQ_MODEL,
    }


# Build a compact property summary so Groq receives only useful context.
def summarize_inputs(flow):
    if not flow:
        return {}

    summary = {LABELS[col]: flow["numeric_inputs"].get(col) for col in RAW_NUMERIC_COLUMNS}
    summary["Furnishing Status"] = flow.get("furnishing")
    summary["Neighborhood"] = flow.get("neighborhood")
    return summary


# Build a grounded JSON prompt so the LLM can explain only confirmed facts.
def build_grounded_explanation_context(flow, result, comparables=None):
    grade = int(result["grade"])
    return {
        "property_inputs": summarize_inputs(flow),
        "model_output": {
            "predicted_price": round(result["price"], 2),
            "raw_grade_class": grade,
            "advisory_recommendation": ADVISORY_LABELS.get(grade, str(grade)),
            "advisory_display": GRADE_LABELS.get(grade, str(grade)),
            "advisory_meaning": ADVISORY_DESCRIPTIONS.get(grade, ""),
            "confidence": round(result["confidence"], 4),
        },
        "comparable_properties": comparables or [],
        "source_rules": {
            "Prompt": "Value extracted from the user's prompt.",
            "Default": "Value filled from training-data defaults because the user did not provide it.",
            "Edited": "Value changed by the user before prediction.",
        },
        "allowed_claims": [
            "Use only the JSON values in this request.",
            "Mention comparable properties only when they are present in comparable_properties.",
            "Describe likely reasons, not exact feature importance.",
        ],
    }


# Ask Groq for a constrained explanation of the prediction.
def explain_with_groq(flow, result, settings, comparables=None):
    prompt = build_grounded_explanation_context(flow, result, comparables)
    system_prompt = (
        "You are the Explanation Node for a student real-estate ML project. "
        "Use only the JSON facts provided by the user message. Do not invent market trends, legal advice, "
        "financial advice, exact feature importance, or facts from outside the JSON. "
        "If a claim is uncertain, write 'Based on the model output' instead of stating it as fact. "
        "Respond with exactly four bullets and these bold labels only: "
        "**Summary:**, **Market Context:**, **Recommendation:**, **Risk Warning:**. "
        "Keep each bullet simple and under 35 words."
    )
    payload = {
        "model": settings["model"],
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(prompt)},
        ],
    }
    request = urllib.request.Request(
        settings["api_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings['api_key']}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "property-price-prediction-streamlit/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Groq explanation failed with status {exc.code}: {detail[:300]}") from exc

    return body["choices"][0]["message"]["content"].strip()


# Build a deterministic explanation when Groq is unavailable or the request fails.
def fallback_explanation(result, reason):
    grade = int(result["grade"])
    display_grade = GRADE_LABELS.get(grade, str(grade))
    recommendation = ADVISORY_LABELS.get(grade, str(grade))
    return (
        f"- **Summary:** Based on the model output, the predicted price is Rs {result['price']:,.0f} with {result['confidence']:.1%} confidence.\n"
        f"- **Market Context:** The advisory class is {display_grade}, using the trained tabular model and confirmed property values.\n"
        f"- **Recommendation:** Treat this as **{recommendation}** in the project advisory context, then verify with real market research.\n"
        f"- **Risk Warning:** Explanation fallback used because {reason}. This is educational output, not financial advice."
    )


# Return either a grounded Groq explanation or a clear fallback explanation.
def generate_explanation(flow, result, settings, comparables=None):
    if not settings.get("api_key"):
        return fallback_explanation(result, "GROQ_API_KEY is not configured")

    try:
        return explain_with_groq(flow, result, settings, comparables)
    except Exception as exc:
        return fallback_explanation(result, f"Groq explanation failed: {exc}")
