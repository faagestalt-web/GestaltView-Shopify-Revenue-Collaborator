"""
plk_engine.py – Simplified Personal Language Key (PLK) analysis
================================================================

This module implements a very lightweight version of the PLK concept.
It is provided as a reference and starting point for more advanced
implementations.  The real PLK engine in GestaltView analyzes a
user’s language to detect personal phrases, metaphors, tone and
cognitive patterns.  For demonstration we simply count word
frequencies and return a resonance score based on how often the
text repeats certain user-defined signature phrases.

Usage:

    from plk_engine import PersonalLanguageKey

    plk = PersonalLanguageKey(profile={'signature_phrases': ['ADHD is my jazz', 'flow state']})
    score, matches = plk.resonance_score('This planner helps you find your flow state and harness your ADHD as jazz.')
    print(score)  # e.g. 0.5
    print(matches)  # ['flow state', 'ADHD is my jazz']

You can extend this module to perform sentiment analysis, metaphor
detection, embedding similarity and more.
"""

from collections import Counter
from typing import Dict, List, Tuple


class PersonalLanguageKey:
    """A simplified representation of a user’s Personal Language Key."""

    def __init__(self, profile: Dict[str, List[str]] | None = None) -> None:
        # The profile contains signature phrases and words that reflect the
        # user’s authentic voice.  In practice this would be learned from
        # the user’s input over time.  Here we accept it as input.
        self.profile = profile or {'signature_phrases': []}

    def resonance_score(self, text: str) -> Tuple[float, List[str]]:
        """
        Compute a simple resonance score for the given text.  The score
        is the fraction of signature phrases found in the text.

        :param text: The text to analyze
        :return: A tuple (score, matches) where score is a float between
                 0 and 1 and matches is a list of phrases that were
                 detected.
        """
        if not text:
            return 0.0, []
        text_lower = text.lower()
        matches = []
        for phrase in self.profile.get('signature_phrases', []):
            if phrase.lower() in text_lower:
                matches.append(phrase)
        score = len(matches) / max(1, len(self.profile.get('signature_phrases', [])))
        return score, matches

    def analyze_text(self, text: str) -> Dict[str, any]:
        """
        Perform an analysis of the text, returning basic statistics and
        the resonance score.

        :param text: The text to analyze
        :return: A dictionary containing word counts, total words, and
                 resonance data.
        """
        words = [w for w in text.split() if w.isalpha()]
        counts = Counter(words)
        score, matches = self.resonance_score(text)
        return {
            'total_words': len(words),
            'unique_words': len(counts),
            'signature_matches': matches,
            'resonance_score': score,
            'word_frequencies': dict(counts),
        }


if __name__ == '__main__':
    # Example usage when run directly
    example_profile = {'signature_phrases': ['adhd is my jazz', 'flow state', 'beautiful tapestry']}
    plk = PersonalLanguageKey(example_profile)
    sample = input('Enter a sentence to analyze: ')
    analysis = plk.analyze_text(sample)
    print('Analysis:', analysis)
