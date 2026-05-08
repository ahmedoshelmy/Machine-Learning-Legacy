# AI Projects

A comprehensive learning and development repository for advanced AI systems, including LLM-based agents, RAG systems, machine learning pipelines, and agentic applications.

## 📚 Repository Structure

```
ai-projects/
├── courses/                              # Learning materials and courses
│   ├── datacamp/
│   ├── deep-learning-specialization/
│   ├── practical-deep-learning/
│   └── knowledge-graphs-deeplearningai/
│
├── projects/                             # Active and planned projects
│   ├── 01-website-summarizer/           # Basic API integration
│   ├── 02-ai-sales-brochure-generator/  # Prompt chaining
│   ├── 03-airline-assistant/            # Function calling + DB
│   ├── 04-multimodal-assistant/         # DALL-E 3 + Whisper
│   ├── 05-meeting-minutes-generator/    # Audio → Text → Summaries
│   ├── 06-python-to-cpp-rust-converter/ # Code translation
│   ├── 07-production-rag-system/        # Enterprise RAG
│   ├── 08-price-prediction-engine/      # XGBoost + Neural Networks
│   ├── 09-fine-tuned-llama/             # QLoRA fine-tuning
│   └── 10-autonomous-deal-scanner/      # Multi-agent capstone
│
├── tools-and-utilities/                  # Reusable code and tools
│   ├── openai-api/                      # OpenAI API utilities
│   ├── mcp-server/                      # MCP server implementations
│   └── pyspark/                         # PySpark examples
│
└── README.md                             # This file
```

## 🎯 Projects Overview

### Foundation Level (1-3)
| # | Project | Focus | Key Skills |
|---|---------|-------|------------|
| 1 | Website Summarizer | API basics | Web scraping, prompt engineering |
| 2 | AI Sales Brochure | Prompt chains | Multi-step workflows, JSON output |
| 3 | Airline Assistant | Function calling | Tool use, database operations |

### Integration Level (4-5)
| # | Project | Focus | Key Skills |
|---|---------|-------|------------|
| 4 | Multimodal Assistant | Multi-input | Image + audio + text |
| 5 | Meeting Minutes | Audio pipeline | Transcription, summarization |

### Advanced Level (6-8)
| # | Project | Focus | Key Skills |
|---|---------|-------|------------|
| 6 | Code Converter | Logic refactoring | Code analysis, performance |
| 7 | Production RAG | Enterprise systems | Vectors, retrieval, re-ranking |
| 8 | Price Predictor | ML pipelines | XGBoost, neural networks |

### Expert Level (9-10)
| # | Project | Focus | Key Skills |
|---|---------|-------|------------|
| 9 | Fine-tuned Llama | Model customization | QLoRA, domain adaptation |
| 10 | Deal Scanner | Multi-agent systems | Orchestration, deployment |

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key
- Git

### Setup
```bash
cd ai-projects
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# For each project:
cd projects/[project-name]
pip install -r requirements.txt
```

## 📖 Progression Path

**Recommended order for learning:**
1. Start with projects 1-3 for fundamentals
2. Move to 4-5 for integration skills
3. Progress to 6-8 for advanced techniques
4. Culminate with 9-10 for expert-level work

## 🛠️ Tools and Utilities

### OpenAI API Utilities
Shared code for OpenAI integrations:
```python
from tools_and_utilities.openai_api import ChatClient, EmbeddingClient
```

### MCP Server
Model Context Protocol implementations for extending capabilities.

### PySpark Examples
Distributed computing patterns and examples.

## 📋 Project Status

Each project has a README with:
- ✅ Project overview
- ✅ Key concepts to learn
- ✅ Recommended stack
- ✅ Project structure
- ⬜ Status tracker (`[ ] Not Started`, `[~] In Progress`, `[x] Complete`)

## 🎓 Course Materials

All course materials are organized by provider:

- **DataCamp** - Structured data science courses
- **Deep Learning Specialization** - Andrew Ng's DL courses
- **Practical Deep Learning** - Fast.ai courses
- **Knowledge Graphs (DeepLearning.ai)** - RAG and KG courses

## 🔗 Key Resources

- [OpenAI API Docs](https://platform.openai.com/docs)
- [LangChain Docs](https://python.langchain.com)
- [CrewAI Docs](https://docs.crewai.com)
- [Modal Docs](https://modal.com/docs)

## 🤝 Contributing

To add a new project:
1. Create a numbered folder: `projects/NN-project-name/`
2. Include a `README.md` with overview and structure
3. Create subdirectories: `src/`, `tests/`, `data/` as needed
4. Add `requirements.txt` for dependencies

## 📝 Notes

This repository follows a **progressive learning model**:
- Each project builds on previous concepts
- Projects increase in complexity and scope
- Real-world application of LLM, ML, and data engineering skills

---

**Last Updated:** May 8, 2026 
