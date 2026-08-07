"""
Hugging Face Qwen maintenance analyzer for the Aircraft Maintenance Platform.

This module extracts text from the aircraft maintenance manual PDF using pypdfium2,
constructs a prompt with the statistics and manual text, and calls the Qwen 2.5
model hosted on Hugging Face Serverless Inference API to generate a structured report.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import pypdfium2 as pdfium
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


class QwenAircraftMaintenanceAnalyzer:
    """
    Generate AI maintenance reports using Hugging Face Qwen and a manual PDF.
    """

    def __init__(
        self,
        hf_client: InferenceClient,
        model_id: str = "Qwen/Qwen2.5-72B-Instruct",
        manual_pdf_path: str | Path = "",
        temperature: float = 0.2,
        max_tokens: int = 2000,
    ) -> None:
        self.hf_client = hf_client
        self.model_id = model_id
        self.manual_pdf_path = Path(manual_pdf_path)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract_manual_text(self) -> str:
        """Extract text from the aircraft maintenance manual PDF using pypdfium2."""
        if not self.manual_pdf_path.exists():
            raise FileNotFoundError(
                f"Maintenance manual not found: {self.manual_pdf_path}"
            )

        if self.manual_pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Maintenance manual must be a PDF file. "
                f"Received: {self.manual_pdf_path}"
            )

        logger.info("Extracting text from maintenance manual: %s", self.manual_pdf_path)
        
        pdf = pdfium.PdfDocument(str(self.manual_pdf_path))
        manual_text = ""
        for i, page in enumerate(pdf):
            textpage = page.get_textpage()
            page_text = textpage.get_text_bounded()
            manual_text += f"--- PAGE {i+1} ---\n{page_text}\n"
            
        return manual_text

    def build_prompt(self, engineering_analytics: dict[str, Any], manual_text: str) -> str:
        """Build the model prompt containing the extracted manual and analytics JSON."""
        analytics_json = json.dumps(
            engineering_analytics,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        return f"""
You are a Senior Aircraft Maintenance Engineer.

Your task is to generate a professional aircraft maintenance engineering report using two inputs:

1. The Aircraft Maintenance Manual provided below.
2. Engineering Analytics JSON provided below.

Important rules:
- Compare every engineering parameter in the analytics JSON against the thresholds, safe operating limits, risk matrix, decision trees, inspection procedures, failure modes, and maintenance actions defined in the manual.
- Use only thresholds and maintenance procedures found in the manual.
- Do not invent thresholds, limits, failure modes, or maintenance actions.
- If a required threshold or procedure is unavailable in the manual, state that explicitly in the JSON output.
- Prioritize maintenance actions when multiple actions apply.
- Determine whether the aircraft status is one of: SAFE, MONITOR, MAINTENANCE REQUIRED, GROUND AIRCRAFT.
- Produce a final flight decision for the operations dashboard. The decision must clearly state whether the aircraft may fly now, may fly with monitoring, or must not fly until maintenance is completed.
- Return JSON only. Do not include Markdown, prose outside JSON, or code fences.

--- AIRCRAFT MAINTENANCE MANUAL ---
{manual_text}

--- ENGINEERING ANALYTICS JSON ---
{analytics_json}

Return exactly one JSON object with this schema:
{{
  "aircraft": "string",
  "aircraft_model": "string",
  "health_status": "SAFE | MONITOR | MAINTENANCE REQUIRED | GROUND AIRCRAFT",
  "risk_level": "LOW | MEDIUM | HIGH | CRITICAL | UNKNOWN",
  "safe_for_next_flight": true,
  "final_flight_decision": {{
    "decision": "CLEARED_TO_FLY | FLY_WITH_MONITORING | MAINTENANCE_REQUIRED_BEFORE_FLIGHT | GROUND_AIRCRAFT",
    "can_fly_now": true,
    "ui_statement": "string",
    "required_before_next_flight": "string",
    "decision_rationale": "string"
  }},
  "overall_summary": "string",
  "threshold_violations": [
    {{
      "parameter": "string",
      "observed_value": "number or string",
      "manual_threshold": "string",
      "severity": "LOW | MEDIUM | HIGH | CRITICAL | UNKNOWN",
      "manual_reference": "string",
      "explanation": "string"
    }}
  ],
  "root_cause": {{
    "most_likely_cause": "string",
    "supporting_evidence": ["string"],
    "manual_reference": "string"
  }},
  "maintenance_actions": [
    {{
      "priority": 1,
      "action": "string",
      "reason": "string",
      "manual_reference": "string"
    }}
  ],
  "inspection_checklist": [
    {{
      "step": 1,
      "inspection_item": "string",
      "acceptance_criteria": "string",
      "manual_reference": "string"
    }}
  ],
  "work_order": {{
    "title": "string",
    "aircraft_id": "string",
    "work_order_type": "INSPECTION | REPAIR | MONITORING | GROUNDING",
    "priority": "LOW | MEDIUM | HIGH | CRITICAL",
    "tasks": ["string"],
    "required_parts_or_tools": ["string"],
    "estimated_maintenance_category": "string"
  }},
  "confidence": {{
    "score": 0.0,
    "rationale": "string",
    "missing_information": ["string"]
  }}
}}
""".strip()

    def analyze(self, engineering_analytics: dict[str, Any]) -> dict[str, Any]:
        """Generate a structured AI maintenance report from analytics and PDF."""
        manual_text = self.extract_manual_text()
        prompt = self.build_prompt(engineering_analytics, manual_text)

        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]

        try:
            logger.info("Invoking Qwen model: %s", self.model_id)
            response = self.hf_client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as exc:
            logger.error("Failed to query Hugging Face API: %s", exc)
            raise RuntimeError(
                f"Unable to invoke Hugging Face model '{self.model_id}': {exc}"
            ) from exc

        response_text = response.choices[0].message.content
        if not response_text:
            raise ValueError("Hugging Face returned an empty response")

        return self._parse_json_response(response_text.strip())

    @staticmethod
    def _parse_json_response(response_text: str) -> dict[str, Any]:
        """Parse and validate the JSON-only model response, stripping markdown blocks if present."""
        cleaned = response_text
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Model returned non-JSON response: %s", response_text)
            raise ValueError("Hugging Face response was not valid JSON") from exc

        if not isinstance(parsed, dict):
            raise ValueError("Hugging Face response JSON must be an object")

        return parsed
