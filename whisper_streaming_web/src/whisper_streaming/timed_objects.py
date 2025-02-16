from dataclasses import dataclass
from typing import Optional

@dataclass
class TimedText:
    start: Optional[float]
    end: Optional[float]
    text: str

@dataclass
class ASRToken(TimedText):
    def with_offset(self, offset: float) -> "ASRToken":
        """Return a new token with the time offset added."""
        return ASRToken(self.start + offset, self.end + offset, self.text)

@dataclass
class Sentence(TimedText):
    pass

@dataclass
class Transcript(TimedText):
    pass