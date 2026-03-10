from __future__ import annotations

import json

from app.dependencies import get_indexer_service


def main() -> None:
    result = get_indexer_service().rebuild_index()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

