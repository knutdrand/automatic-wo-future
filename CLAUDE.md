# Claude Code Configuration for automatic-model

## Allowed Commands

The following commands can be run without prompting for user approval:

- `uv run chap*` - All chap CLI commands (evaluate, evaluate2, export-metrics, etc.)
- `uv run uvicorn*` - Running the uvicorn server
- `pkill` - For stopping background processes
- `/Users/knutdr/Sources/automatic-model/scripts/*.sh` - Project shell scripts

## Project Overview

This project develops a CHAP-compatible disease prediction model using:
- **chapkit**: ML service framework
- **chap-core**: Evaluation and backtesting
- **darts**: Time series forecasting library

## Workflow

1. Start the model server: `./scripts/start_server.sh`
2. Run evaluation: Use `chap evaluate2` from chap-core directory
3. Export metrics: Use `chap export-metrics`
4. Track improvements in `METRICS_LOG.md`
