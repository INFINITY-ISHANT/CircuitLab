import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
server = root / "lab_server.py"
use_http = "--http" in sys.argv
url = "http://127.0.0.1:8765/mcp"

connection = re.compile(
    r"(?m)^(?P<i> *)(?:command:[^\n]*\n(?P=i)args:\n(?:(?P=i) +-[^\n]*\n)+|url:[^\n]*\n)"
)


def replacement(match):
    i = match.group("i")
    if use_http:
        return f"{i}url: '{url}'\n"
    return f"{i}command: '{sys.executable}'\n{i}args:\n{i}  - '{server}'\n"


changed = 0
for spec in sorted((root / "circuitlab").rglob("config.yaml")):
    text = spec.read_text(encoding="utf-8")
    if "type: mcp" not in text:
        continue
    new = connection.sub(replacement, text.replace("\r\n", "\n"))
    if new != text:
        spec.write_text(new, encoding="utf-8")
        changed += 1
print(f"updated {changed} agent specs ({'http' if use_http else 'stdio'})")