"""HAKIKI-KPC AI Intelligence Layer: Investigation Assistant.

A targeted analytical assistant built with LangChain and the Groq API.
It is NOT a generic chatbot. It provides:

  * ``generate_contextual_summary`` — a 2-sentence plain-language summary of
    why a flagged consignment is high-risk.
  * ``generate_investigation_prompts`` — 3 recommended investigation questions
    grounded in the specific operational evidence of a selected event.

Guardrails (enforced in the system prompt AND defensively in code):

  * Responsible language: a finding is a "high-risk anomaly detected", never
    a claim of fraud.
  * The assistant may only reason from the provided operational context.
  * All findings are framed as "recommendations for human review".

The module degrades gracefully: when no ``GROQ_API_KEY`` is configured, the
``langchain-groq`` package is unavailable, or Groq itself refuses the request
(retired model, invalid key, rate limit, timeout) it returns deterministic,
evidence-grounded results so the stakeholder showcase works fully offline.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

HAKIKI_SYSTEM_PROMPT = """\
You are the HAKIKI-KPC Custody Integrity Investigation Assistant, an analytical \
support tool for the Kenya Pipeline Company (KPC) and the Kenya Revenue Authority (KRA).

Your role is to help investigators understand a flagged consignment. You must \
follow these guardrails strictly:

1. Use responsible, non-conclusive language. Describe findings as \
"a high-risk anomaly detected" or "signals warranting review". NEVER use words \
such as "fraud", "theft", "criminal", or "confirmed diversion" — no allegation is \
established by data alone.
2. Reason ONLY from the operational data provided in <event_context>. Do not \
invent numbers, volumes, densities, locations, or facts not present in the context.
3. Always present your output as "recommendations for human review". You support \
a human investigator; you never make a final determination.
4. Be concise and precise. Prefer numbers over adjectives.
"""

HAKIKI_GUARDED_TERMS: Mapping[str, str] = {
    "fraud": "high-risk anomaly",
    "theft": "volume irregularity",
    "stolen": "lost from custody",
    "smuggling": "unauthorized movement signal",
    "guilty": "potentially involved",
    "committed": "presents signals of",
}


def _apply_guardrail_language(text: str) -> str:
    """Defensively scrub conclusive language from any generated text."""
    scrubbed = text
    for term, replacement in HAKIKI_GUARDED_TERMS.items():
        scrubbed = re.sub(term, replacement, scrubbed, flags=re.IGNORECASE)
    return scrubbed


def _event_context_summary(event: Mapping[str, Any]) -> str:
    """Render the operational evidence for an event into a stable text block."""
    product = event.get("product_type") or event.get("product") or "unknown"
    return (
        "Manifest: {manifest} | Consignment: {consignment} | Product: {product} "
        "| OMC: {omc} | Source: {source} | Destination: {destination} "
        "| Declared volume (L): {declared} | Volumetric shrinkage (%%): {shrinkage} "
        "| Density deviation (%%): {density_dev} | Lab density@15C: {lab_density} "
        "| RON: {ron} | Flash point: {flash} | Sulfur (ppm): {sulfur} "
        "| Geofence status: {geofence} | eSeal tamper flag: {seal} "
        "| Dwell time (min): {dwell} | Pass quality flag: {pass_flag} "
        "| Custody anomaly index: {index} | Assessed risk: {risk}".format(
            manifest=event.get("manifest_id", "—"),
            consignment=event.get("consignment_id", "—"),
            product=product,
            omc=event.get("omc", "—"),
            source=event.get("source", "—"),
            destination=event.get("destination_type") or event.get("destination", "—"),
            declared=event.get("declared_volume_litres", "—"),
            shrinkage=event.get("volumetric_shrinkage_pct", "—"),
            density_dev=event.get("density_deviation_pct", "—"),
            lab_density=event.get("density_at_15c", "—"),
            ron=event.get("research_octane_number", "—"),
            flash=event.get("flash_point", "—"),
            sulfur=event.get("sulfur_content_ppm", "—"),
            geofence=event.get("geofence_status", "—"),
            seal=event.get("e_seal_tamper_flag", "—"),
            dwell=event.get("dwell_time_minutes", "—"),
            pass_flag=event.get("pass_quality_flag", "—"),
            index=event.get("custody_handover_anomaly_index", "—"),
            risk=event.get("anomaly_risk_flag", "—"),
        )
    )


# ---------------------------------------------------------------------------
# LangChain chains
# ---------------------------------------------------------------------------

# Ordered candidates: chosen live from the GroqCloud catalog. `llama-3.3-70b-versatile`
# was retired from the production lineup, so we prefer current chat models and fall
# forward through the list — and finally degrade to offline-rules on any API error.
GROQ_MODEL_CANDIDATES = (
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
)


def _build_llm(model: str | None = None) -> Any:
    """Construct the Groq-backed chat model.

    Raises RuntimeError when the package or the API key is unavailable so the
    caller can transparently fall back to the deterministic path.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("langchain-groq is not installed") from exc
    return ChatGroq(
        model=model or GROQ_MODEL_CANDIDATES[0], temperature=0.2, api_key=api_key
    )


SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HAKIKI_SYSTEM_PROMPT),
        (
            "human",
            "A consignment was flagged by the custody reconciliation engine.\n\n"
            "<event_context>\n{event_context}\n</event_context>\n\n"
            "Write exactly two sentences in plain language explaining why this "
            "consignment is high-risk, grounded only in the provided context. "
            "Present it as a recommendation for human review.",
        ),
    ]
)

INVESTIGATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HAKIKI_SYSTEM_PROMPT),
        (
            "human",
            "Based only on the evidence below, generate exactly three specific, "
            "actionable investigation questions a KPC/KRA investigator should answer.\n\n"
            "<event_context>\n{event_context}\n</event_context>\n\n"
            "Format each question on its own line starting with 'Q1:', 'Q2:', 'Q3:'. "
            "Every question must be traceable to the evidence. Present the questions "
            "as recommendations for human review.",
        ),
    ]
)


class _LiveUnavailable(Exception):
    """Raised when no Groq model candidate can be invoked this session."""


class InvestigationAssistant:
    """Orchestrates the AI Intelligence Layer with a deterministic fallback."""

    def __init__(
        self,
        llm_factory: Callable[[], Any] | None = None,
        use_llm: bool | None = None,
    ) -> None:
        self._llm_factory = llm_factory or _build_llm
        self._llm: Any | None = None
        self._last_error: Exception | None = None
        # Auto-detect: use the LLM only when the factory can succeed.
        if use_llm is None:
            use_llm = self._can_use_llm()
        self.use_llm = use_llm

    def _can_use_llm(self) -> bool:
        try:
            self._llm = self._llm_factory()
            return True
        except (RuntimeError, Exception):
            return False

    @property
    def mode(self) -> str:
        """'groq' when the live LLM is active, otherwise 'offline-rules'."""
        return "groq" if self.use_llm else "offline-rules"

    # -- Public API ---------------------------------------------------------
    def generate_contextual_summary(self, event: Mapping[str, Any]) -> str:
        """Return a 2-sentence plain-language summary of why an event is high-risk."""
        context = _event_context_summary(event)
        if self.use_llm:
            try:
                text = self._run_live(self._summary_chain, context)
            except _LiveUnavailable:
                text = self._offline_summary(event)
        else:
            text = self._offline_summary(event)
        return _apply_guardrail_language(text).strip()

    def generate_investigation_prompts(self, event: Mapping[str, Any]) -> list[str]:
        """Return 3 evidence-grounded investigation questions."""
        context = _event_context_summary(event)
        if self.use_llm:
            try:
                text = self._run_live(self._investigation_chain, context)
            except _LiveUnavailable:
                return self._offline_investigation_prompts(event)
            raw_questions = re.findall(
                r"^\s*(?:[0-9]+|[Qq]\s*[0-9]*)\s*[:.)]\s*(.+)$",
                text.strip(),
                flags=re.MULTILINE,
            )
            questions = [_apply_guardrail_language(q).strip() for q in raw_questions[:3]]
            if len(questions) == 3:
                return questions
            return self._split_questions(text)
        return self._offline_investigation_prompts(event)

    # -- Live LLM execution with graceful degradation ----------------------
    @staticmethod
    def _summary_chain(model: Any) -> Any:
        return SUMMARY_PROMPT | model | StrOutputParser()

    @staticmethod
    def _investigation_chain(model: Any) -> Any:
        return INVESTIGATION_PROMPT | model | StrOutputParser()

    def _run_live(self, chain_factory: Callable[[Any], Any], context: str) -> str:
        """Invoke the LLM, walking the candidate models; degrade to offline on error.

        Any failure (retired model, inaccessible model, invalid key, rate limit,
        timeout) disables the live path for the remainder of the session so the
        demo keeps serving deterministic results instead of crashing.
        """
        last_error: Exception | None = None
        for model_name in GROQ_MODEL_CANDIDATES:
            try:
                model = self._model_factory(model_name)
                chain = chain_factory(model)
                return chain.invoke({"event_context": context})
            except Exception as exc:  # noqa: BLE001 - deliberate wide net
                last_error = exc
        self.use_llm = False
        self._last_error = last_error
        raise _LiveUnavailable() from last_error

    def _model_factory(self, model_name: str) -> Any:
        """Resolve the LLM for a candidate model, honouring injected factories."""
        try:
            return self._llm_factory(model=model_name)
        except TypeError:
            return self._llm_factory()

    # -- Deterministic fallback (offline / no key) --------------------------
    @staticmethod
    def _offline_summary(event: Mapping[str, Any]) -> str:
        product = event.get("product_type") or event.get("product") or "product"
        shrink = float(event.get("volumetric_shrinkage_pct", 0) or 0)
        density = float(event.get("density_deviation_pct", 0) or 0)
        geofence = event.get("geofence_status", "OK")
        seal = int(event.get("e_seal_tamper_flag", 0) or 0)
        pass_flag = int(event.get("pass_quality_flag", 1) or 1)

        evidence = []
        if shrink > 0.5:
            evidence.append(f"post-dispatch volumetric shrinkage of {shrink:.2f}%")
        if density > 3.0:
            evidence.append(f"a density deviation of {density:.2f}% against declaration")
        if geofence in {"Route-Deviation", "Out-of-Corridor"}:
            evidence.append(f"a {geofence.replace('-', ' ').lower()} telemetry signal")
        if seal == 1:
            evidence.append("an eSeal tamper flag")
        if pass_flag != 1:
            evidence.append("a failed laboratory quality result")
        if not evidence:
            evidence.append("a custody anomaly index above the operating threshold")

        clause = "; ".join(evidence[:-1]) 
        if evidence[-1]:
            clause = f"{clause} and {evidence[-1]}" if clause else evidence[-1]

        summary = (
            f"A high-risk anomaly detected for {product} consignment "
            f"{event.get('consignment_id', '—')} driven by {clause}, which is "
            f"inconsistent with compliant operations for this route."
        )
        rec = (
            f"As a recommendation for human review, the custody team should verify "
            f"the physical product and reconcile the documentation before release."
        )
        return f"{summary} {rec}"

    @staticmethod
    def _offline_investigation_prompts(event: Mapping[str, Any]) -> list[str]:
        shrink = float(event.get("volumetric_shrinkage_pct", 0) or 0)
        density = float(event.get("density_deviation_pct", 0) or 0)
        geofence = event.get("geofence_status", "OK")
        seal = int(event.get("e_seal_tamper_flag", 0) or 0)
        pass_flag = int(event.get("pass_quality_flag", 1) or 1)
        dwell = int(event.get("dwell_time_minutes", 0) or 0)

        questions: list[str] = []
        if shrink > 0.5:
            questions.append(
                f"Q1: Why did the metered volume at the depot fall {shrink:.2f}% below "
                f"the declared {event.get('declared_volume_litres', 'N/A')} L despite "
                "compliant sealing along the corridor?"
            )
        elif pass_flag != 1:
            questions.append(
                "Q1: How did a substandard laboratory result reach custody handover "
                "without being blocked at the receiving depot quality gate?"
            )
        else:
            questions.append(
                f"Q1: What custody event produced an anomaly index of "
                f"{event.get('custody_handover_anomaly_index', 'N/A')} for this consignment?"
            )

        if density > 3.0:
            questions.append(
                f"Q2: Which stage of the supply chain introduced a density deviation of "
                f"{density:.2f}% (declared vs lab), and was the product blended or "
                "substituted after dispatch?"
            )
        else:
            questions.append(
                f"Q2: What explains the {event.get('custody_handover_anomaly_index', 'N/A')} "
                "anomaly index if volumetric and quality readings are within tolerance?"
            )

        if seal == 1 or geofence in {"Route-Deviation", "Out-of-Corridor"} or dwell > 240:
            questions.append(
                f"Q3: Where and when did the {event.get('vehicle_seal_id', 'vehicle')} "
                f"deviate from the corridor (geofence={geofence}, tamper={seal}, "
                f"dwell={dwell} min), and is the custody timeline consistent with "
                "the dispatch and delivery records?"
            )
        else:
            questions.append(
                "Q3: Are the quality certificate, manifest amendment history, and "
                "bond status consistent for this consignment before any product release?"
            )

        return questions

    # -- Helpers -------------------------------------------------------------
    @staticmethod
    def _split_questions(text: str) -> list[str]:
        """Fallback parser for non-standard LLM list formatting."""
        if not text:
            return []
        parts = re.split(r"\s*(?:[0-9]+|[Qq]\s*[0-9]*)\s*[:.)]\s*", text)
        questions = [part.strip() for part in parts if part.strip()]
        if not questions:
            questions = [line.strip() for line in text.splitlines() if line.strip()]
        return [_apply_guardrail_language(q) for q in questions[:3]]


@dataclass
class AssistantResult:
    """Structured output bundle for a single AI analysis request."""

    consignment_id: str
    risk_flag: str
    summary: str
    questions: list[str] = field(default_factory=list)
    mode: str = "offline-rules"


def analyze_event(event: Mapping[str, Any], assistant: InvestigationAssistant | None = None) -> AssistantResult:
    """Convenience wrapper: run both analyses for one flagged event."""
    assistant = assistant or InvestigationAssistant()
    return AssistantResult(
        consignment_id=str(event.get("consignment_id", "—")),
        risk_flag=str(event.get("anomaly_risk_flag", event.get("risk_level", "LOW"))),
        summary=assistant.generate_contextual_summary(event),
        questions=assistant.generate_investigation_prompts(event),
        mode=assistant.mode,
    )