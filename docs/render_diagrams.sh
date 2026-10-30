#!/usr/bin/env bash
# Re-render docs/diagrams/*.mmd into docs/img/*.png.
#
#   npm i -g @mermaid-js/mermaid-cli
#   ./render_diagrams.sh
#
# If mmdc cannot find a browser, point CHROME_PATH at any local Chromium/Chrome:
#   CHROME_PATH=/usr/bin/chromium ./render_diagrams.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p img

cfg=$(mktemp)
trap 'rm -f "$cfg"' EXIT
if [ -n "${CHROME_PATH:-}" ]; then
  printf '{"executablePath": "%s", "args": ["--no-sandbox"]}\n' "$CHROME_PATH" > "$cfg"
else
  printf '{"args": ["--no-sandbox"]}\n' > "$cfg"
fi

for f in diagrams/*.mmd; do
  name=$(basename "$f" .mmd)
  mmdc -i "$f" -o "img/$name.png" -c mermaid.config.json -p "$cfg" -b white -s 2 -q
  echo "rendered img/$name.png"
done
