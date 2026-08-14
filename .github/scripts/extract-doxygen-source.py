# SPDX-License-Identifier: CC0-1.0

"""Extract exact source lines from a pinned Doxygen *_source.html page."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path


class DoxygenSourceParser(HTMLParser):
    """Collect text inside Doxygen's one-`div.line`-per-source-line markup."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._line_depth = 0
        self._parts: list[str] = []
        self.lines: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag != "div":
            return
        if self._line_depth:
            self._line_depth += 1
            return
        classes = dict(attrs).get("class", "") or ""
        if "line" in classes.split():
            self._line_depth = 1
            self._parts.clear()

    def handle_endtag(self, tag: str) -> None:
        if tag != "div" or not self._line_depth:
            return
        self._line_depth -= 1
        if not self._line_depth:
            self.lines.append("".join(self._parts).replace("\N{NO-BREAK SPACE}", " "))
            self._parts.clear()

    def handle_data(self, data: str) -> None:
        if self._line_depth:
            self._parts.append(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()

    extractor = DoxygenSourceParser()
    extractor.feed(arguments.input.read_text(encoding="utf-8"))
    extractor.close()
    if not extractor.lines:
        raise SystemExit("Doxygen page contained no div.line source records")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8", newline="") as output:
        output.write("\r\n".join(extractor.lines))
        output.write("\r\n")


if __name__ == "__main__":
    main()
