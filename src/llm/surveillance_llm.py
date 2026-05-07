import json
from typing import Any, Dict, Optional

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.config.logger import logger
from src.config.models.surveillance_metadata import SurveillanceMetadata
from src.config.settings import LangChainSettings
from src.prompts.prompt_template import PROMPT_v1, REPORT_PROMPT_v1


class SurveillanceLLM:
    """
    Enhanced LangChain-based LLM for surveillance data analysis.

    Features:
    - LangChain PromptTemplate integration
    - Pydantic output parsing with validation
    - Built-in retry logic with exponential backoff
    - Factory pattern for consistent initialization
    - Support for structured JSON outputs
    """

    def __init__(self, settings: Optional[LangChainSettings] = None) -> None:
        """
        Initialize the SurveillanceLLM with enhanced LangChain features.

        :param settings: Pydantic settings for LangChain configuration.
        :return: None
        :raise: Exception if initialization fails.
        """
        try:
            self.settings = settings or LangChainSettings()

            # Initialize LangChain LLM with new package
            self.llm = OllamaLLM(
                base_url=self.settings.ollama_base_url,
                model=self.settings.ollama_model,
                temperature=self.settings.ollama_temperature,
                # timeout=self.settings.ollama_timeout,
            )

            # Initialize prompt template and output parser (lazy)
            self.prompt_template = None
            self.output_parser = None
            self.chain = None

            logger.debug(
                f"Initialized SurveillanceLLM with model: {self.settings.ollama_model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize SurveillanceLLM: {e}")
            raise

    def _ensure_chain_initialized(self) -> None:
        """Lazily initialize the chain if not already created."""
        if self.chain is None:
            self.prompt_template = self._create_prompt_template()
            self.output_parser = PydanticOutputParser(
                pydantic_object=SurveillanceMetadata
            )
            self.chain = self.prompt_template | self.llm | self.output_parser
            logger.debug("Initialized LangChain prompt/parser chain")

    @staticmethod
    def _create_prompt_template() -> PromptTemplate:
        """Create LangChain PromptTemplate for surveillance analysis."""
        template = PROMPT_v1

        return PromptTemplate(
            input_variables=["tags", "format_instructions"],
            template=template,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((Exception,)),
        reraise=True,
    )
    def analyze_surveillance_element(
        self, element: Dict[str, Any]
    ) -> SurveillanceMetadata:
        """
        Analyze a surveillance element using LangChain with retry logic.

        :param element: OSM element dictionary with tags.
        :return: Parsed and validated SurveillanceMetadata.
        :raise: Exception if analysis fails after retries.
        """
        try:
            # Ensure chain is initialized
            self._ensure_chain_initialized()

            tags = element.get("tags", {})
            tags_json = json.dumps(tags, ensure_ascii=False, indent=2)

            logger.debug(f"Analyzing surveillance element with tags: {tags}")

            # Get format instructions from parser
            format_instructions = self.output_parser.get_format_instructions()

            # Use the chain to process the input
            result = self.chain.invoke(
                {"tags": tags_json, "format_instructions": format_instructions}
            )

            # Create the complete metadata object
            metadata = SurveillanceMetadata.from_raw(element, result.model_dump())

            logger.debug(
                f"Successfully analyzed element {element.get('id', 'unknown')}"
            )
            return metadata

        except ValidationError as e:
            logger.warning(
                f"Validation error for element {element.get('id', 'unknown')}: {e}"
            )
            # Fallback: create metadata with validation errors
            return SurveillanceMetadata.from_raw(element, {"schema_errors": str(e)})
        except Exception as e:
            logger.error(
                f"Failed to analyze element {element.get('id', 'unknown')}: {e}"
            )
            raise

    def analyze_surveillance_elements_batch(self, elements: list) -> list:
        """
        Analyze a list of surveillance elements in a single Ollama batch call.

        Issues one ``self.chain.batch(...)`` invocation. Returns a list of
        dicts aligned with the input order:

        - On success at index ``i``: the ``model_dump(exclude_none=True)`` of
          ``SurveillanceMetadata.from_raw(element_i, llm_result_i)``.
        - On a per-result failure (validation error, parse error, transport
          error for that specific request): ``{"error": str(exc)}`` at that
          index. Other indices in the batch are unaffected.

        A failure of the batch call as a whole (network down, model not
        loaded, etc.) propagates to the caller, which is expected to wrap
        the call in a try/except per chunk.

        :param elements: List of OSM element dicts. Empty list returns ``[]``.
        :return: List of analysis dicts aligned with ``elements``.
        """
        if not elements:
            return []

        self._ensure_chain_initialized()
        format_instructions = self.output_parser.get_format_instructions()

        inputs = [
            {
                "tags": json.dumps(el.get("tags", {}), ensure_ascii=False, indent=2),
                "format_instructions": format_instructions,
            }
            for el in elements
        ]

        logger.debug(f"Analyzing {len(elements)} surveillance elements in one batch")
        # ``return_exceptions=True`` keeps the batch atomic from the chain's
        # perspective: a malformed LLM response on one prompt becomes an
        # Exception at that index, leaving the others as parsed objects.
        results = self.chain.batch(inputs, return_exceptions=True)

        out: list = []
        for element, result in zip(elements, results):
            eid = element.get("id", "unknown")
            if isinstance(result, Exception):
                logger.warning(f"Batch result error for element {eid}: {result}")
                out.append({"error": str(result)})
                continue
            try:
                metadata = SurveillanceMetadata.from_raw(element, result.model_dump())
                out.append(metadata.model_dump(exclude_none=True))
            except ValidationError as e:
                logger.warning(f"Validation error for element {eid}: {e}")
                out.append({"error": str(e)})
            except Exception as e:
                logger.warning(f"from_raw failed for element {eid}: {e}")
                out.append({"error": str(e)})
        return out

    def generate_city_report(
        self,
        stats: Dict[str, Any],
        sample: list,
    ) -> str:
        """
        Generate a fixed-section markdown city report from the
        analyzer's summary stats and a small sample of sensitive
        cameras.

        Sections (enforced by the prompt): ``Overview``, ``Operators``,
        ``Privacy mix``, ``Sensitivity``, ``Hotspots``, ``Caveats``.

        :param stats: The dict returned by ``compute_statistics``.
            ``Counter`` values are normalised to plain ``dict``\\ s
            for the prompt.
        :param sample: List of enriched camera dicts (each with
            ``analysis``) for cameras flagged sensitive. The method
            extracts the relevant fields and truncates to 10 entries.
        :return: Markdown report as a single string.
        :raise: ``RuntimeError`` if the LLM call fails. Callers should
            wrap this in a try/except so a report failure does not
            abort the analyzer run.
        """
        stats_summary = self._summarize_stats_for_report(stats)
        sample_block = self._format_sensitive_sample(sample)

        prompt_template = PromptTemplate(
            input_variables=["stats_summary", "sensitive_sample"],
            template=REPORT_PROMPT_v1,
        )
        chain = prompt_template | self.llm

        logger.info("Generating city report via LLM")
        try:
            result = chain.invoke(
                {"stats_summary": stats_summary, "sensitive_sample": sample_block}
            )
        except Exception as e:
            logger.error(f"City report generation failed: {e}")
            raise RuntimeError(f"City report generation failed: {e}") from e

        return str(result).strip()

    @staticmethod
    def _summarize_stats_for_report(stats: Dict[str, Any]) -> str:
        """
        Render a compact, prompt-friendly summary of the stats dict.

        ``Counter`` values are emitted as ``most_common(top_n)`` lists
        so the LLM has the same prioritised view the charts use.
        """

        def _top(counter_like: Any, n: int = 5) -> list:
            if not counter_like:
                return []
            if hasattr(counter_like, "most_common"):
                return list(counter_like.most_common(n))
            # Plain dict fallback (post-JSON round-trip)
            items = sorted(counter_like.items(), key=lambda kv: kv[1], reverse=True)
            return items[:n]

        total = stats.get("total", 0)
        sensitive = stats.get("sensitive_count", 0)
        public = stats.get("public_count", 0)
        private = stats.get("private_count", 0)
        unknown_privacy = max(total - public - private, 0)

        lines = [
            f"- total_cameras: {total}",
            f"- sensitive: {sensitive}",
            f"- public: {public}",
            f"- private: {private}",
            f"- unknown_privacy: {unknown_privacy}",
            f"- top_operators: {_top(stats.get('operator_counts'))}",
            f"- top_camera_types: {_top(stats.get('camera_type_counts'))}",
            f"- top_zones: {_top(stats.get('zone_counts'))}",
            f"- top_zones_by_sensitive: {_top(stats.get('zone_sensitivity_counts'))}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _format_sensitive_sample(sample: list, limit: int = 10) -> str:
        """
        Render up to ``limit`` sensitive cameras as compact bullet lines.

        Pulls ``operator``, ``zone``, and ``sensitive_reason`` from each
        element's ``analysis`` dict. Elements without those fields are
        skipped silently. Returns ``"(none)"`` if the sample is empty
        so the prompt's "explicit no data" path is exercised.
        """
        rows = []
        for el in sample:
            if len(rows) >= limit:
                break
            analysis = el.get("analysis", {}) if isinstance(el, dict) else {}
            if not isinstance(analysis, dict) or not analysis.get("sensitive"):
                continue
            op = analysis.get("operator") or "—"
            zone = analysis.get("zone") or "—"
            reason = analysis.get("sensitive_reason") or "—"
            rows.append(f"- operator={op}, zone={zone}, reason={reason}")
        return "\n".join(rows) if rows else "(none)"

    def generate_response(self, prompt: str, **kwargs: Any) -> str:
        """
        Generate a response from the LLM (backward compatibility method).

        :param prompt: The text prompt to send to the LLM.
        :param kwargs: Additional parameters.
        :return: The generated response text from the LLM.
        :raise: RuntimeError if the LLM request fails.
        """
        try:
            logger.debug(f"Generating response for prompt: {prompt!r}")

            # Handle kwargs by creating temporary LLM if needed
            if kwargs:
                temp_llm = OllamaLLM(
                    base_url=self.settings.ollama_base_url,
                    model=self.settings.ollama_model,
                    temperature=kwargs.get(
                        "temperature", self.settings.ollama_temperature
                    ),
                    # timeout=kwargs.get("timeout", self.settings.ollama_timeout),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["temperature", "timeout"]
                    },
                )
                response = temp_llm.invoke(prompt)
            else:
                response = self.llm.invoke(prompt)

            logger.debug("Successfully generated LLM response")
            return str(response).strip()

        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            raise RuntimeError(f"LLM generation error: {e}") from e

    def generate_batch(self, prompts: list, **kwargs: Any) -> list:
        """
        Generate responses for multiple prompts in batch.

        :param prompts: List of text prompts to send to the LLM.
        :param kwargs: Additional parameters.
        :return: List of generated response texts.
        :raise: RuntimeError if batch generation fails.
        """
        try:
            logger.debug(f"Generating batch responses for {len(prompts)} prompts")

            if kwargs:
                temp_llm = OllamaLLM(
                    base_url=self.settings.ollama_base_url,
                    model=self.settings.ollama_model,
                    temperature=kwargs.get(
                        "temperature", self.settings.ollama_temperature
                    ),
                    # timeout=kwargs.get("timeout", self.settings.ollama_timeout),
                    **{
                        k: v
                        for k, v in kwargs.items()
                        if k not in ["temperature", "timeout"]
                    },
                )
                responses = temp_llm.batch(prompts)
            else:
                responses = self.llm.batch(prompts)

            results = [str(response).strip() for response in responses]
            logger.debug(f"Successfully generated {len(results)} batch responses")
            return results

        except Exception as e:
            logger.error(f"Batch generation error: {e}")
            raise RuntimeError(f"Batch generation error: {e}") from e


def create_surveillance_llm(
    settings: Optional[LangChainSettings] = None,
) -> SurveillanceLLM:
    """
    Factory function for creating SurveillanceLLM instances with consistent configuration.

    :param settings: Optional LangChain settings. If None, uses default settings.
    :return: Configured SurveillanceLLM instance.
    :raise: Exception if LLM creation fails.
    """
    try:
        llm = SurveillanceLLM(settings)
        logger.info(f"Created SurveillanceLLM with model: {llm.settings.ollama_model}")
        return llm
    except Exception as e:
        logger.error(f"Failed to create SurveillanceLLM: {e}")
        raise


# Backward compatibility alias
LangChainLLM = SurveillanceLLM
