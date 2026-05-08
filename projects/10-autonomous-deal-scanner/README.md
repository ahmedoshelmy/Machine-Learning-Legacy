# Autonomous Deal Scanner

## Overview
Capstone multi-agent system with serverless deployment and automated notifications.

## Concepts
- Multi-agent orchestration (CrewAI / AutoGen)
- Agent roles and responsibilities
- Tool integration and API calls
- Information aggregation
- Serverless deployment (Modal)
- Notification systems (email, Slack)
- Scheduling and automation

## Project Structure
```
10-autonomous-deal-scanner/
├── src/
│   ├── main.py                  # Orchestration
│   ├── agents/
│   │   ├── market_analyzer.py
│   │   ├── deal_evaluator.py
│   │   ├── news_monitor.py
│   │   └── notifier.py
│   ├── tools/
│   │   ├── api_integrations.py
│   │   ├── data_fetcher.py
│   │   └── evaluator.py
│   ├── deployment/
│   │   ├── modal_handler.py
│   │   └── scheduler.py
│   └── notifications/
│       ├── email_sender.py
│       └── slack_integration.py
├── config/
│   └── agents_config.yaml
├── data/
│   ├── alerts/
│   └── history/
├── requirements.txt
└── README.md
```

## Status
`[ ] Not Started`

## Stack
- Python 3.10+
- OpenAI API
- CrewAI / AutoGen
- Modal (serverless)
- LangChain
- APScheduler (scheduling)
- Various market APIs (Alpha Vantage, NewsAPI, etc.)
