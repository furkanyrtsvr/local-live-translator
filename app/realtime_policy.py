"""Content-preserving controls for keeping synthesized speech near the source clock."""
import re


def bounded_speed(predicted_seconds, target_seconds, base_speed, maximum_speed):
    if target_seconds is None or target_seconds <= 0:
        return base_speed
    return min(maximum_speed, max(base_speed, predicted_seconds / target_seconds))


def phrase_budget(source_seconds, phrase_words, total_words, backlog_seconds=0):
    # Recover small accumulated drift by making upcoming speech a little shorter.
    # Never delete sentences or ask the translator to summarise them.
    catchup = min(source_seconds * .20, max(0, backlog_seconds - .5) * .20)
    return max(.35, (source_seconds-catchup) * phrase_words/max(1, total_words))


class TranscriptOverlap:
    def __init__(self):
        self.previous = []
        self.language = ''

    def apply(self, text, language, has_overlap):
        words = text.split()
        normalise = lambda w: re.sub(r'[^\w]', '', w.casefold())
        current = [normalise(word) for word in words]
        remove = 0
        if has_overlap and language == self.language:
            for count in range(1, min(8, len(self.previous), len(current))+1):
                if self.previous[-count:] == current[:count]:
                    remove = count
        self.previous = current
        self.language = language
        return ' '.join(words[remove:])
