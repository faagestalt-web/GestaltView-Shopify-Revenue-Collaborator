# core/creation_engine.py
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import asyncio
import openai
import os

logger = logging.getLogger(__name__)

@dataclass
class ChaosInput:
    """Represents chaotic creative inputs from the user"""
    text_notes: List[str] = field(default_factory=list)
    voice_transcripts: List[str] = field(default_factory=list)
    file_contents: List[str] = field(default_factory=list)
    emotional_markers: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SynthesisRequest:
    """Request for creative synthesis"""
    user_id: str
    chaos_inputs: ChaosInput
    output_type: str  # "code_snippet", "essay", "poem", "solution", "brainstorm"
    personalization: Dict[str, Any] = field(default_factory=dict)
    synthesis_style: str = "convergent"  # "convergent", "divergent", "analytical"
    
@dataclass
class CreativeOutput:
    """Result of creative synthesis"""
    content: str
    output_type: str
    synthesis_metadata: Dict[str, Any]
    creativity_score: float
    coherence_score: float
    personalization_applied: List[str]
    creation_timestamp: datetime = field(default_factory=datetime.utcnow)

class CreationCornerEngine:
    """Advanced creative synthesis engine for chaos-to-masterpiece transformation"""
    
    def __init__(self):
        self.openai_client = openai
        self.openai_client.api_key = os.getenv("OPENAI_API_KEY")
        self.synthesis_history: List[CreativeOutput] = []
        self.creative_patterns: Dict[str, Any] = {}
        
    async def synthesize(self, request: SynthesisRequest) -> CreativeOutput:
        """Main synthesis method - transforms chaos into creative output"""
        logger.info(f"Starting synthesis for user {request.user_id}, type: {request.output_type}")
        
        # Step 1: Prepare and contextualize inputs
        contextualized_chaos = await self._contextualize_chaos(request.chaos_inputs, request.personalization)
        
        # Step 2: Apply synthesis strategy based on output type
        synthesis_prompt = self._build_synthesis_prompt(
            contextualized_chaos, 
            request.output_type, 
            request.synthesis_style,
            request.personalization
        )
        
        # Step 3: Generate creative content
        raw_output = await self._generate_creative_content(synthesis_prompt, request.output_type)
        
        # Step 4: Post-process and enhance
        enhanced_output = await self._enhance_output(raw_output, request)
        
        # Step 5: Evaluate creativity and coherence
        creativity_score = self._evaluate_creativity(enhanced_output, request.chaos_inputs)
        coherence_score = self._evaluate_coherence(enhanced_output)
        
        # Step 6: Package result
        result = CreativeOutput(
            content=enhanced_output,
            output_type=request.output_type,
            synthesis_metadata={
                "input_complexity": len(request.chaos_inputs.text_notes),
                "synthesis_style": request.synthesis_style,
                "processing_time": "async",
                "plk_patterns_applied": len(request.personalization.get('signature_metaphors', []))
            },
            creativity_score=creativity_score,
            coherence_score=coherence_score,
            personalization_applied=self._extract_applied_personalization(request.personalization)
        )
        
        self.synthesis_history.append(result)
        return result
    
    async def _contextualize_chaos(self, chaos_inputs: ChaosInput, personalization: Dict[str, Any]) -> str:
        """Transform chaotic inputs into contextual narrative"""
        all_inputs = []
        all_inputs.extend(chaos_inputs.text_notes)
        all_inputs.extend(chaos_inputs.voice_transcripts)
        all_inputs.extend(chaos_inputs.file_contents)
        all_inputs.extend(chaos_inputs.emotional_markers)
        
        # Add personalization context
        plk_context = ""
        if personalization.get('signature_metaphors'):
            metaphors = [m['metaphor'] for m in personalization['signature_metaphors'][:3]]
            plk_context = f"User resonates with these concepts: {', '.join(metaphors)}. "
        
        contextualized = f"{plk_context}Raw creative materials: " + " | ".join(all_inputs)
        return contextualized
    
    def _build_synthesis_prompt(self, contextualized_chaos: str, output_type: str, synthesis_style: str, personalization: Dict[str, Any]) -> str:
        """Build the synthesis prompt for AI generation"""
        
        base_prompts = {
            "code_snippet": f"Transform these ideas into clean, functional code that solves a real problem:",
            "essay": f"Weave these thoughts into a compelling essay with clear insights:",
            "poem": f"Distill these feelings and ideas into evocative poetry:",
            "solution": f"Synthesize these inputs into a practical, actionable solution:",
            "brainstorm": f"Expand and connect these ideas into a comprehensive brainstorm:",
            "masterpiece": f"Create a unique masterpiece that captures the essence of these ideas:"
        }
        
        style_modifiers = {
            "convergent": "Focus on finding the unified core insight that connects everything.",
            "divergent": "Explore multiple creative directions and possibilities.",
            "analytical": "Break down and systematically examine the relationships between ideas."
        }
        
        personalization_prompt = ""
        if personalization.get('recent_states'):
            recent_state = personalization['recent_states'][-1] if personalization['recent_states'] else 'focused'
            personalization_prompt = f"The user is currently feeling {recent_state}. Match this energy. "
        
        prompt = f"""
{personalization_prompt}
{base_prompts.get(output_type, base_prompts['masterpiece'])}
{style_modifiers.get(synthesis_style, '')}

Creative Materials:
{contextualized_chaos}

Instructions:
- Be authentic and personally meaningful
- Maintain coherence while embracing creativity  
- If this resonates with ADHD thinking patterns, honor that neurodivergent perspective
- Create something genuinely useful and inspiring
"""
        return prompt
    
    async def _generate_creative_content(self, prompt: str, output_type: str) -> str:
        """Generate creative content using AI"""
        try:
            response = await asyncio.create_task(self._call_openai_async(prompt, output_type))
            return response
        except Exception as e:
            logger.error(f"AI generation error: {e}")
            return f"Creative synthesis in progress... The essence of your ideas is forming into something beautiful. [Generation temporarily unavailable - {output_type}]"
    
    async def _call_openai_async(self, prompt: str, output_type: str) -> str:
        """Async wrapper for OpenAI call"""
        try:
            response = self.openai_client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"You are a creative synthesis engine specializing in {output_type} creation. You understand neurodivergent thinking patterns and ADHD creativity."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def _enhance_output(self, raw_output: str, request: SynthesisRequest) -> str:
        """Post-process and enhance the generated content"""
        enhanced = raw_output
        
        # Add ADHD-friendly formatting for code
        if request.output_type == "code_snippet":
            enhanced = f"```python\n# ADHD-Friendly Code: Clear, commented, modular\n{enhanced}\n```"
        
        # Add encouraging footer for all outputs
        footer = "\n\n✨ Created with your unique creative signature ✨"
        enhanced += footer
        
        return enhanced
    
    def _evaluate_creativity(self, output: str, original_inputs: ChaosInput) -> float:
        """Evaluate creativity score (0-1)"""
        # Simple heuristic - in production, use more sophisticated NLP
        input_words = set()
        for text in original_inputs.text_notes:
            input_words.update(text.lower().split())
        
        output_words = set(output.lower().split())
        
        # Creativity = (new concepts introduced) / (total concepts)
        new_words = output_words - input_words
        creativity_ratio = len(new_words) / len(output_words) if output_words else 0
        
        return min(creativity_ratio * 2, 1.0)  # Scale to 0-1
    
    def _evaluate_coherence(self, output: str) -> float:
        """Evaluate coherence score (0-1)"""
        # Simple coherence check - sentence length variance indicates good flow
        sentences = output.split('.')
        if len(sentences) < 2:
            return 0.5
        
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if not sentence_lengths:
            return 0.5
            
        # Good coherence = reasonable sentence length variety
        avg_length = sum(sentence_lengths) / len(sentence_lengths)
        variance = sum((l - avg_length) ** 2 for l in sentence_lengths) / len(sentence_lengths)
        
        # Normalize to 0-1 (lower variance = better coherence for creative writing)
        coherence_score = max(0, 1 - (variance / 100))
        return min(coherence_score, 1.0)
    
    def _extract_applied_personalization(self, personalization: Dict[str, Any]) -> List[str]:
        """Extract which personalization features were applied"""
        applied = []
        if personalization.get('signature_metaphors'):
            applied.append(f"{len(personalization['signature_metaphors'])} signature metaphors")
        if personalization.get('recent_states'):
            applied.append(f"emotional state: {personalization['recent_states'][-1]}")
        if personalization.get('authenticity_patterns'):
            applied.append("authenticity patterns")
        return applied
    
    def get_creation_analytics(self) -> Dict[str, Any]:
        """Get analytics about creation patterns"""
        if not self.synthesis_history:
            return {"message": "No creations yet - ready to synthesize your first masterpiece!"}
        
        output_types = [c.output_type for c in self.synthesis_history]
        type_counts = {}
        for otype in output_types:
            type_counts[otype] = type_counts.get(otype, 0) + 1
        
        avg_creativity = sum(c.creativity_score for c in self.synthesis_history) / len(self.synthesis_history)
        avg_coherence = sum(c.coherence_score for c in self.synthesis_history) / len(self.synthesis_history)
        
        return {
            "total_creations": len(self.synthesis_history),
            "output_type_distribution": type_counts,
            "most_created_type": max(type_counts, key=type_counts.get) if type_counts else "None",
            "average_creativity_score": round(avg_creativity, 2),
            "average_coherence_score": round(avg_coherence, 2),
            "creative_evolution": "Expanding" if len(self.synthesis_history) > 5 else "Emerging",
            "personalization_usage": sum(len(c.personalization_applied) for c in self.synthesis_history)
        }
