"""Legacy entrypoint retained for compatibility.

The historical implementation fitted StandardScaler on the complete dataset
before splitting, so it is not a supported evaluation or artifact-generation
path. Use the canonical exporter instead.
"""

from scripts.train_and_export_model import main


if __name__ == "__main__":
    raise SystemExit(main())
