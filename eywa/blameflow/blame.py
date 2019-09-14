
class BlameType:
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    CORRECTIVE = "CORRECTIVE"


class Blame(object):

    def __init__(self, blame_type, expected=None, confidence=1.0):
        assert blame_type in [BlameType.POSITIVE, BlameType.NEGATIVE, BlameType.CORRECTIVE]
        self.blame_type = blame_type
        self.expected = expected
        if blame_type != BlameType.CORRECTIVE and expected is not None:
            raise Exception("expected arg should be provided only for corrective blame.")
        assert confidence >= 0. and confidence <= 1.0
        if confidence < 1.0 and blame_type == BlameType.POSITIVE:
            raise Exception("Positive blame should always have 100% confidence.")
        self.confidence = confidence
        self.parent = None
        self.node = None
        self.node_updated = False
        self.forks = []

    def fork(self, blame_type=None, expected=None, confidence=1.0):
        if blame_type is None:
            blame_type = self.blame_type
        if blame_type == BlameType.CORRECTIVE and expected is None:
            expected = self.expected
        b = Blame(blame_type, expected, confidence=confidence)
        b.parent = self
        self.forks.append(b)
        return b
