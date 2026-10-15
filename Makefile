.PHONY: help setup lint gen-configs

help:
	@echo "Targets:"
	@echo "  setup             Install runtime dependencies (requirements.txt)"
	@echo "  lint              Run ruff (see pyproject.toml)"
	@echo "  gen-configs       Regenerate every campaign's configs"

setup:
	pip install -r requirements.txt

lint:
	uv tool run ruff check .

gen-configs:
	python3 -m modules.generate_labels_trigger_joint.gen_configs
	python3 -m modules.generate_labels_trigger_joint.gen_configs_paper_main_campaign
