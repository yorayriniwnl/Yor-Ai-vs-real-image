"""Legacy entrypoint retained for compatibility.

The historical version of this file fitted StandardScaler before the train/test
split. That leaked holdout distribution information into preprocessing. Model
artifacts must now be regenerated through the canonical leakage-safe exporter.
"""

from scripts.train_and_export_model import main


if __name__ == "__main__":
    raise SystemExit(main())
