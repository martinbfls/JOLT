.PHONY: help setup lint gen-configs backup backup-dry restore

help:
	@echo "Targets:"
	@echo "  setup             Install runtime dependencies (requirements.txt)"
	@echo "  lint              Run ruff (see pyproject.toml)"
	@echo "  gen-configs       Regenerate every campaign's configs"
	@echo "  backup            Back up experiments/ to a private HF dataset (HF_REPO=user/name to override)"
	@echo "  backup-dry        Show what backup would upload"
	@echo "  restore           Download the backed-up experiments/"

setup:
	pip install -r requirements.txt

lint:
	uv tool run ruff check .

gen-configs:
	python3 -m modules.generate_labels_trigger_joint.gen_configs
	python3 -m modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign

backup:
	python3 scripts/backup_results.py $(BACKUP_ARGS)

backup-dry:
	python3 scripts/backup_results.py --dry-run $(BACKUP_ARGS)

restore:
	python3 scripts/backup_results.py --restore $(BACKUP_ARGS)
