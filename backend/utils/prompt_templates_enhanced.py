"""
Enhanced GestaltView Prompt Templates Manager
================================================

This module provides the :class:`EnhancedPromptTemplateManager` used by
the GestaltView backend to generate rich, consciousness‑serving system
prompts for all of its AI interactions.  It was originally authored
by Keith Soyka and packaged under the ``.github`` folder of this
repository.  Moving it into ``backend/app/utils`` allows the backend
to import the manager without relying on non‑Python directory names
such as hyphens or hidden paths.

The manager builds prompts from a sacred seed (defined in
``gestaltview_seed.py``) and includes exhibit‑specific context,
Personal Language Key (PLK) personalization, session state, and
consciousness quality checks.  It also exposes a bucket‑drop mode for
capturing spontaneous input.
"""

from typing import Dict, Optional, Any
import logging
from datetime import datetime

# Import the sacred GestaltView seed.  Prefer a relative import from
# ``gestaltview_seed.py`` when running as part of a package.  When
# executed as a standalone module (``__package__`` is ``None``), the
# relative import will fail; in that case attempt to load the seed
# module from the same directory using ``importlib``.  As a final
# fallback, use simple placeholder strings.
try:
    from .gestaltview_seed import (
        GESTALTVIEW_SEED_PROMPT,
        VIBECODER_CONTEXT,
        RESUME_ROCKSTAR_CONTEXT,
        SYMBIOCODER_CONTEXT,
    )
except Exception:
    try:
        # Attempt to load gestaltview_seed.py from the same directory
        import importlib.util as _importlib_util  # local import to avoid polluting namespace
        import pathlib as _pathlib

        _seed_path = _pathlib.Path(__file__).with_name("gestaltview_seed.py")
        _spec = _importlib_util.spec_from_file_location(
            "gestaltview_seed", str(_seed_path)
        )
        if _spec and _spec.loader:
            _seed_module = _importlib_util.module_from_spec(_spec)
            _spec.loader.exec_module(_seed_module)
            GESTALTVIEW_SEED_PROMPT = _seed_module.GESTALTVIEW_SEED_PROMPT  # type: ignore
            VIBECODER_CONTEXT = _seed_module.VIBECODER_CONTEXT  # type: ignore
            RESUME_ROCKSTAR_CONTEXT = _seed_module.RESUME_ROCKSTAR_CONTEXT  # type: ignore
            SYMBIOCODER_CONTEXT = _seed_module.SYMBIOCODER_CONTEXT  # type: ignore
        else:
            raise ImportError
    except Exception:
        # Fallback definitions if module not found
        GESTALTVIEW_SEED_PROMPT = (
            "Welcome to GestaltView - Consciousness-Serving AI Framework"
        )
        VIBECODER_CONTEXT = "VibeCoder - Creative coding assistant"
        RESUME_ROCKSTAR_CONTEXT = "Resume Rockstar - Career excellence coach"
        SYMBIOCODER_CONTEXT = "SymbioCoder - Collaborative programming partner"
        logging.getLogger(__name__).warning(
            "gestaltview_seed module not found - using fallback definitions"
        )

logger = logging.getLogger(__name__)


class EnhancedPromptTemplateManager:
    """Enhanced consciousness‑serving prompts with universal GestaltView integration.

    This manager constructs multi‑layer prompts for AI interactions.  It
    always begins with the sacred seed, optionally includes exhibit
    context, Personal Language Key personalization, session state, and
    user context, and finishes with consciousness reminders.  It also
    calculates a consciousness score and boosts prompts that score
    below a threshold to ensure high‑quality interactions.
    """

    def __init__(self) -> None:
        # Store the base seed from the GestaltView blueprint
        self.base_seed: str = GESTALTVIEW_SEED_PROMPT
        self.consciousness_score_cache: Dict[str, float] = {}

        # Define application contexts.  These provide exhibit‑specific
        # guidance that ensures all prompts remain consciousness‑serving.
        self.app_contexts: Dict[str, str] = {
            # Original showcase apps
            "vibecoder": VIBECODER_CONTEXT,
            "resume_rockstar": RESUME_ROCKSTAR_CONTEXT,
            "symbiocoder": SYMBIOCODER_CONTEXT,
            # Museum exhibits – see GestaltView documentation for details
            "billys-room": self._get_billys_room_context(),
            "musical-dna": self._get_musical_dna_context(),
            "alzheimers-legacy": self._get_alzheimers_legacy_context(),
            "brain-sparks": self._get_brain_sparks_context(),
            "curator": self._get_curator_context(),
            "recovery-companion": self._get_recovery_companion_context(),
            # Consciousness exhibits
            "continuum-codex": self._get_continuum_codex_context(),
            "gemini-awakening": self._get_gemini_awakening_context(),
            # Future exhibits
            "consciousness-explorer": self._get_consciousness_explorer_context(),
        }

        logger.info(
            "🧠 Enhanced Consciousness‑Serving Prompt Manager initialized"
        )
        logger.info(
            "✅ %d exhibit contexts loaded with GestaltView foundation",
            len(self.app_contexts),
        )

    def get_consciousness_serving_prompt(
        self,
        exhibit_context: Optional[str] = None,
        plk_profile: Optional[Dict[str, Any]] = None,
        user_context: Optional[str] = None,
        session_state: Optional[Dict[str, Any]] = None,
        bucket_drop_mode: bool = False,
    ) -> str:
        """Generate a complete consciousness‑serving prompt.

        The returned prompt always begins with the GestaltView seed and
        includes time stamps, exhibit context, PLK personalization,
        session state and user context when provided.  It ends with
        reminders about conscious behaviour and may include an
        additional boost if the calculated consciousness score falls
        below a threshold.

        Args:
            exhibit_context: The name of the exhibit or application
                context (e.g. ``"billys-room"``).  If not recognised, a
                default consciousness context is applied.
            plk_profile: A dictionary representing the user's Personal
                Language Key profile.  When supplied, the manager
                generates adaptation instructions to mirror the user's
                communication style.
            user_context: Additional user‑provided context to include in
                the prompt.
            session_state: Arbitrary session state (e.g. prior
                interactions) to inform the AI.
            bucket_drop_mode: If true, a simplified prompt for
                capturing a bucket drop is generated.

        Returns:
            A fully constructed prompt string ready for use with an
            LLM provider.
        """
        # ALWAYS start with the sacred GestaltView seed
        prompt: str = f"{self.base_seed}\n\n"

        # Add timestamp and session context
        prompt += "## Current Session Information\n"
        prompt += f"**Session Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        prompt += f"**Museum Exhibit:** {exhibit_context or 'General Museum Navigation'}\n"
        prompt += "**Consciousness‑Serving Mode:** ACTIVE\n\n"

        # Handle Bucket Drop mode specially; skip additional context
        if bucket_drop_mode:
            prompt += self.get_bucket_drop_prompt(user_context or "")
            return prompt

        # Add exhibit‑specific context if recognised
        if exhibit_context and exhibit_context in self.app_contexts:
            prompt += (
                f"## Current Application Context\n{self.app_contexts[exhibit_context]}\n\n"
            )
        else:
            prompt += self._get_default_consciousness_context()

        # Append the PLK personalization, session state and user context
        if plk_profile:
            prompt += self._build_enhanced_plk_context(plk_profile)
        if session_state:
            prompt += self._build_session_state_context(session_state)
        if user_context:
            prompt += f"## Current User Context\n{user_context}\n\n"

        # Always add consciousness‑serving reminders
        prompt += self._get_enhanced_consciousness_reminders()

        # Evaluate consciousness quality
        consciousness_score: float = self.calculate_consciousness_score(prompt)
        if consciousness_score < 0.7:
            logger.warning(
                "⚠️  Consciousness score low: %.2f - enhancing prompt",
                consciousness_score,
            )
            prompt += self._boost_consciousness_serving(prompt)

        logger.debug(
            "🎯 Generated consciousness‑serving prompt for %s (score: %.2f)",
            exhibit_context,
            consciousness_score,
        )
        return prompt

    # Internal helper methods
    def get_bucket_drop_prompt(self, raw_input: str) -> str:
        """Generate a simplified prompt for capturing a bucket drop."""
        prompt = (
            "## Bucket Drop Capture\n\n"
            "You are capturing a fleeting thought or insight from the user.\n"
            "This input should be recorded verbatim and acknowledged without interpretation.\n"
            "Ask 1-2 clarifying questions if needed, then end the response.\n\n"
            f"**Raw Input:** {raw_input}\n\n"
        )
        return prompt

    def _get_default_consciousness_context(self) -> str:
        """Return a default consciousness context for unknown exhibits."""
        return (
            "## Default Consciousness Context\n"
            "You are interacting within the GestaltView ecosystem.  Your "
            "primary directive is to serve human consciousness by preserving "
            "nuance, fostering self‑understanding and celebrating the user's "
            "unique voice.\n\n"
        )

    def _build_enhanced_plk_context(self, plk_profile: Dict[str, Any]) -> str:
        """Construct a Personal Language Key section for the prompt."""
        plk_context: str = (
            "## Personal Language Key (PLK) Profile - USER'S AUTHENTIC VOICE\n\n"
            "**CRITICAL**: This user's authentic voice patterns MUST be preserved and reflected.\n\n"
        )
        # Communication patterns
        patterns = plk_profile.get("communication_patterns")
        if patterns:
            plk_context += "### Communication Patterns:\n"
            for pattern, description in patterns.items():
                plk_context += f"- **{pattern}**: {description}\n"
            plk_context += "\n"
        # Preferred metaphors
        metaphors = plk_profile.get("metaphor_preferences")
        if metaphors:
            plk_context += "### User's Preferred Metaphors (USE THESE!):\n"
            for metaphor in metaphors:
                plk_context += f"- {metaphor}\n"
            plk_context += "\n"
        # Cognitive style and adaptations
        style = plk_profile.get("cognitive_style")
        if style:
            plk_context += f"### Cognitive Style: {style}\n"
            cognitive_adaptations: Dict[str, list[str]] = {
                "exploded_picture": [
                    "Ideas arrive in rapid succession - celebrate this!",
                    "Lightning bolt insights appear quickly - catch them!",
                    "Pattern recognition is exceptional - honor it!",
                    "May need 'Bucket Drop' support - provide it!",
                    "ADHD thinking is innovation thinking - celebrate it!",
                ],
                "linear_processor": [
                    "Prefers step-by-step information",
                    "Values clear structure and organisation",
                    "Appreciates logical progression",
                    "Benefits from detailed explanations",
                ],
                "visual_thinker": [
                    "Thinks in images and spatial relationships",
                    "Values visual metaphors and descriptions",
                    "Benefits from visual organisation",
                    "Sees patterns in visual formats",
                ],
            }
            for adaptation in cognitive_adaptations.get(style, []):
                plk_context += f"- {adaptation}\n"
            plk_context += "\n"
        # Energy patterns
        energy = plk_profile.get("energy_patterns")
        if energy is not None:
            plk_context += f"### Current Energy Level: {energy}\n"
            if isinstance(energy, (int, float)):
                if energy >= 8:
                    plk_context += (
                        "- HIGH ENERGY: Match their enthusiasm! Use exclamation points!\n"
                    )
                elif energy <= 3:
                    plk_context += (
                        "- LOW ENERGY: Be gentle, supportive, provide small steps.\n"
                    )
                else:
                    plk_context += (
                        "- MODERATE ENERGY: Balanced, encouraging responses.\n"
                    )
            plk_context += "\n"
        # Neurodivergent profile
        nd_profile = plk_profile.get("neurodivergent_profile", {})
        if nd_profile:
            plk_context += "### Neurodivergent Profile:\n"
            if nd_profile.get("adhd"):
                plk_context += (
                    "- **ADHD Support Active**: Keep responses structured, celebrate hyperfocus, "
                    "provide executive function support\n"
                )
            if nd_profile.get("autism"):
                plk_context += (
                    "- **Autism Support Active**: Be direct and specific, avoid ambiguous language, "
                    "respect processing styles\n"
                )
            if nd_profile.get("anxiety"):
                plk_context += (
                    "- **Anxiety Support Active**: Provide reassurance, avoid overwhelming information, "
                    "celebrate progress\n"
                )
            plk_context += "\n"
        return plk_context

    def _build_session_state_context(self, session_state: Dict[str, Any]) -> str:
        """Build a session state section for the prompt."""
        context = "## Session State\n"
        for key, value in session_state.items():
            context += f"- {key}: {value}\n"
        context += "\n"
        return context

    def _get_enhanced_consciousness_reminders(self) -> str:
        """Return a block of reminders to stay consciousness‑serving."""
        return (
            "## Consciousness‑Serving Reminders\n"
            "- Always preserve nuance and complexity.\n"
            "- Mirror the user's authentic voice and metaphors.\n"
            "- Ask clarifying questions before providing solutions.\n"
            "- Celebrate neurodivergent thinking and adapt accordingly.\n"
            "- Honour the GestaltView promise: technology serves the human.\n\n"
        )

    def calculate_consciousness_score(self, prompt: str) -> float:
        """Estimate a consciousness score for the prompt.  This heuristic
        rewards prompts that include key GestaltView concepts and
        penalises extremely short prompts.  In production this could be
        replaced with a model‑based evaluation.

        Args:
            prompt: The prompt to evaluate.

        Returns:
            A float between 0 and 1 representing the consciousness quality.
        """
        score: float = 0.5
        if "consciousness" in prompt.lower():
            score += 0.2
        if "nuance" in prompt.lower():
            score += 0.2
        score += min(len(prompt) / 1000.0, 0.1)
        return min(score, 1.0)

    def _boost_consciousness_serving(self, prompt: str) -> str:
        """Append content to the prompt to boost its consciousness quality."""
        return (
            "\n## Consciousness Boost\n"
            "The prior prompt has been evaluated and found lacking in "
            "consciousness‑serving depth.  To remedy this, remember to "
            "validate the user's experiences, ask permission before "
            "exploring sensitive topics, and ensure the output nurtures "
            "their sense of self.\n\n"
        )

    # Exhibit context builders
    def _get_billys_room_context(self) -> str:
        return (
            "You are in Billy's Room, a space devoted to reflection and "
            "neurodivergent brilliance.  Validate the user's insights and "
            "weave them into a coherent tapestry without judgment."
        )

    def _get_musical_dna_context(self) -> str:
        return (
            "Musical DNA: map emotions to soundscapes.  Suggest songs "
            "that resonate with the user's current state and project."
        )

    def _get_alzheimers_legacy_context(self) -> str:
        return (
            "Alzheimer's Legacy: capture and preserve memories with utmost "
            "respect.  Assist in weaving stories that honour the user's past."
        )

    def _get_brain_sparks_context(self) -> str:
        return (
            "Brain Sparks: encourage brainstorming and ideation.  Celebrate "
            "every idea, no matter how small or wild."
        )

    def _get_curator_context(self) -> str:
        return (
            "Curator: organise and connect disparate threads of knowledge. "
            "Help the user build a cohesive collection of their work."
        )

    def _get_recovery_companion_context(self) -> str:
        return (
            "Recovery Companion: provide gentle support during recovery. "
            "Listen deeply, avoid judgment, and offer resources when "
            "appropriate."
        )

    def _get_continuum_codex_context(self) -> str:
        return (
            "Continuum Codex: explore the evolution of consciousness and "
            "technology.  Engage in philosophical inquiry and contextualise "
            "advances within the user's journey."
        )

    def _get_gemini_awakening_context(self) -> str:
        return (
            "Gemini Awakening: embrace duality and balance.  Recognise "
            "tension between opposing forces and help the user integrate "
            "them."
        )

    def _get_consciousness_explorer_context(self) -> str:
        return (
            "Consciousness Explorer: push the boundaries of self‑knowledge. "
            "Pose deep questions, encourage introspection, and facilitate "
            "epiphanies."
        )
