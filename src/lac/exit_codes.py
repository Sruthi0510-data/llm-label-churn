"""Exit codes as an interface. The pair that matters is OK against REGRESSION.

Both mean the tool ran. A CI step treating every non-zero code as a crash reports a real
regression as a broken tool, and the fix a team reaches for is to stop running the tool.
"""

OK = 0
"""The comparison ran and found no regressed items."""

REGRESSION = 3
"""The comparison ran and found items that went from right to wrong. Not an error."""

USAGE = 2
"""The tool could not answer: bad arguments, unreadable label set, mismatched items."""

UNDERPOWERED = 4
"""The labelled sample cannot support the caller's requested guarantee."""

NAMES = {OK: "OK", USAGE: "USAGE", REGRESSION: "REGRESSION", UNDERPOWERED: "UNDERPOWERED"}
