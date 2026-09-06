"""Write the CHANGELOG section for a release tag to a file."""

import re
import sys
from pathlib import Path

CHANGELOG = Path("CHANGELOG.md")


def section_for(version: str, changelog: str) -> str:
    pattern = rf"^## +{re.escape(version)}(?:\s.*)?$"
    lines = changelog.splitlines()
    for index, line in enumerate(lines):
        if re.match(pattern, line):
            body = []
            for following in lines[index + 1:]:
                if following.startswith("## "):
                    break
                body.append(following)
            return "\n".join(body).strip()
    raise SystemExit(f"No '## {version}' section found in {CHANGELOG}.")


def main() -> None:
    tag, destination = sys.argv[1], Path(sys.argv[2])
    version = tag[1:] if tag.startswith("v") else tag
    destination.write_text(section_for(version, CHANGELOG.read_text(encoding="utf-8")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
