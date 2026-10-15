def wandb_block(module_name, run_name, enabled, project, mode="online", entity=None, group=None):
    """Returns a `[<module_name>.wandb]` TOML snippet when enabled, else "" (the module
    then runs with local-only plots/metrics, see experiment_tracker.py). Pure function of
    its arguments (no reliance on any module-level globals) so every gen_configs*.py
    generator can import and reuse it with its own toggles."""
    if not enabled:
        return ""
    lines = [
        f"\n[{module_name}.wandb]",
        "enabled = true",
        f'project = "{project}"',
        f'mode = "{mode}"',
        f'run_name = "{run_name}"',
    ]
    if entity:
        lines.append(f'entity = "{entity}"')
    if group:
        lines.append(f'group = "{group}"')
    return "\n".join(lines) + "\n"
