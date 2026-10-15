"""Shared schema validation: every config key must match schemas/<module_name>.toml (required or [OPTIONAL])."""

from pathlib import Path

import toml

SCHEMAS_DIR = Path("schemas")


class ConfigValidationError(AssertionError):
    """Raised when a config's keys don't match its schema."""


def validate_module_config(module_name: str, module_config: dict, schemas_dir: Path = SCHEMAS_DIR):
    """Validates one module's config table against schemas/<module_name>.toml."""
    schema_path = schemas_dir / f"{module_name}.toml"
    if not schema_path.exists():
        raise ConfigValidationError(
            f"Malformed module! Schema {schema_path} does not exist."
        )

    schema = toml.load(schema_path)
    optionals = list(schema.get("OPTIONAL", {}).keys())

    schema_keys = set(schema[module_name].keys())
    config_keys = set(module_config.keys())

    missing = [k for k in schema_keys - config_keys if k not in optionals]
    extra = [k for k in config_keys - schema_keys if k not in optionals]

    if missing:
        raise ConfigValidationError(
            f"Malformed config ([{module_name}]): missing required keys {missing}"
        )
    if extra:
        raise ConfigValidationError(
            f"Malformed config ([{module_name}]): unknown keys {extra} "
            "(not in schema, not optional)"
        )


def validate_config_file(path: Path, schemas_dir: Path = SCHEMAS_DIR):
    """Validates every module table in a config.toml file."""
    exp_toml = toml.load(path)
    for module_name, module_config in exp_toml.items():
        try:
            validate_module_config(module_name, module_config, schemas_dir=schemas_dir)
        except ConfigValidationError as e:
            raise ConfigValidationError(f"{e} (file: {path})") from e


def write_config(path: Path, content: str):
    """Writes a generated config.toml, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[OK] Config written to {path}")
