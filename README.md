# Working Memory Simulator for LLMs

**Simulating Human Working Memory Constraints in Large Language Models**

A computational framework for investigating how working-memory constraints affect large language model (LLM) reasoning on multi-step mathematical and logical tasks.

## Project Overview

This dissertation implements a **Working Memory (WM) Simulator**: a control layer that imposes human-like memory constraints on LLMs and evaluates how different memory limitations and forgetting strategies influence reasoning performance.

### Key Features

- **Token-Limited Buffer**: Configurable active memory with explicit capacity limits
- **Forgetting Policies**: Four distinct strategies for managing memory overflow:
  - **Recency**: Drop oldest chunks first (baseline)
  - **Importance**: Drop least important chunks (based on heuristics)
  - **Random**: Random chunk selection (control condition)
  - **Summarising Recency**: Compress older chunks before dropping
- **Summarisation Modes**: Light and heavy compression of older information
- **Complete Logging**: Detailed traces of reasoning steps, memory evolution, trim events, and token usage
- **Reproducible Experiments**: Seeds, configuration files, and resumable runs

## Repository Structure

```
wm-prototype/
├── src/                          # Core implementation
│   ├── buffer.py                # Token-limited working memory buffer
│   ├── engine.py                # Multi-step reasoning orchestrator
│   ├── experiment_runner.py     # Experimental setup and result logging
│   ├── policies.py              # Forgetting policy implementations
│   ├── summariser.py            # Text compression (rule-based + cognitive mode)
│   ├── llm_client.py            # OpenAI API wrapper
│   ├── dataset.py               # Dataset loading + answer matching
│   └── tokenizer.py             # Token counting utility
├── data/
│   ├── gsm8k_200.jsonl          # GSM8K benchmark subset
│   └── full_run_*.jsonl         # Generated run outputs
├── configs/
│   ├── core_main_matrix.json    # Primary core-matrix configuration
│   └── extension_cognitive_modes.json  # Cognitive extension configuration
├── docs/                        # Dissertation chapters and references
├── requirements.txt             # Python dependencies
└── .env                         # Local API key (not committed)
```

## Quick Start

### 1. Setup

```bash
# Clone/navigate to repository
cd wm-prototype

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

### 2. Local Smoke Check (No API)

```bash
python -c "import compileall; print(compileall.compile_dir('src', quiet=1))"
```

Expected output:
```
True
```

### 3. API Smoke Check (Single Call)

```bash
python -c "from src.llm_client import LLMClient; llm=LLMClient('gpt-4o-mini'); print(llm.complete('Reply with exactly: OK'))"
```

### 4. Run One Task Through the Pipeline

Create a small script/notebook that uses `ExperimentRunner.run_single_task(...)` with:
- a task row from `data/gsm8k_200.jsonl`
- buffer size (e.g., 64)
- policy (`recency`, `random`, `importance`, `summarising_recency`)
- summarisation level (`light`, `heavy`, or `cognitive` for summarising policy)

## Implementation Details

### Buffer (`src/buffer.py`)

- **Max Tokens**: Configurable capacity in tokens
- **Pinned Problem**: Problem statement (starting with "PROBLEM:") is never dropped
- **Chunk-Based**: Discrete text segments for memory management

```python
buffer = Buffer(max_tokens=128, token_counter=TokenCounter())
buffer.add("PROBLEM: solve 2x + 3 = 11")
buffer.add("Step 1: subtract 3 from both sides")
buffer.trim_until_fits()  # Drop oldest if over capacity
```

### Forgetting Policies (`src/policies.py`)

Each policy implements `apply(buffer: Buffer)`:

1. **RecencyPolicy**: Drop oldest chunks when over capacity
2. **RandomPolicy**: Randomly select chunks to drop
3. **ImportancePolicy**: Drop least important (based on numbers, keywords)
4. **SummarisingRecencyPolicy**: Compress old chunks before dropping

### Reasoning Engine (`src/engine.py`)

Multi-step LLM reasoning loop:

```
Initialise: Add problem to buffer
For each step (1 to max_steps):
  1. Build prompt with current buffer
  2. Call LLM (get thought, answer, done signal)
  3. Add LLM response to buffer
  4. Apply forgetting policy
  5. Log trim events & value tracking
  6. If LLM signals DONE → return answer
Return None (if max steps reached without DONE)
```

### Comprehensive Logging

Each experiment run produces JSONL with:

```json
{
  "task_id": "gsm8k_1",
  "question": "...",
  "expected": "5",
  "predicted": "5",
  "correct": true,
  "buffer_size": 128,
  "policy": "summarising_recency",
  "summarisation_level": "light",
  "steps": 4,
  "usage_total": 1250,
  "response_time": 2.5,
  "duration": 3.2,
  "trace": [
    {
      "thought": "Let me work through this step by step...",
      "answer": "intermediate_answer",
      "done": false,
      "introduced_values": ["5", "3"],
      "step_usage": 312
    },
    ...
  ],
  "trim_events": [
    {
      "step": 3,
      "tokens_before": 145,
      "tokens_after": 120,
      "dropped_chunk_details": [
        {
          "content": "Step 1: ...",
          "dropped_introduced_values": [{"value": "10", "introduced_step": 1}]
        }
      ]
    }
  ],
  "trim_count": 1,
  "config_key": "id=gsm8k_1|bs=128|pol=summarising_recency|sum=light|win=3"
}
```

## Testing

This workspace snapshot does not currently include a `tests/` directory. Recommended checks are:

```bash
# Syntax/compile check
python -c "import compileall; print(compileall.compile_dir('src', quiet=1))"

# Import check
python -c "import importlib; [importlib.import_module(m) for m in ['src.buffer','src.dataset','src.engine','src.experiment_runner','src.llm_client','src.policies','src.summariser','src.tokenizer']]; print('imports ok')"
```

## Ethics & Compliance

✅ **No human participants** — all experiments on public datasets (GSM8K subset)  
✅ **Public data only** — no personal or sensitive information  
✅ **Secure storage** — university-approved OneDrive with backup  
✅ **Reproducible** — full code, configs, and documentation  
✅ **Transparent limitations** — all results clearly contextualized  


## Literature Review

The project is grounded in cognitive science and LLM research:

- **Human WM**: Bounded capacity (~4 chunks), executive control, strategic forgetting
- **LLMs**: Large context windows, no central executive, context-overload failure
- **Gap**: No prior work systematically imposes human-like WM constraints on LLMs


## Key Results (Full Core Matrix)

From the completed core matrix (`data/full_run_core_matrix_results.jsonl`; 2,600 runs = 200 GSM8K tasks × 13 conditions):

| Slice | Accuracy |
|------|----------|
| Overall | 0.7415 |
| Buffer 64 | 0.7138 |
| Buffer 128 | 0.6888 |
| Buffer 256 | **0.8060** |
| Full-context baseline | **0.8700** |

**Findings:**
- Best constrained aggregate performance occurs at buffer 256
- Constrained policy effects are modest (random/summarising_recency ≈ 0.7383)
- Completion reliability matters: 280/2600 rows have `predicted = null`

## Dependencies

```
openai>=1.0.0
tiktoken
numpy
pandas
jsonlines
matplotlib
seaborn
python-dotenv
tqdm
jupyter
```

## Research Questions

**RQ1**: How do different working-memory constraints (buffer size, summarisation, policy) affect LLM reasoning accuracy?

**RQ2**: Do different forgetting policies produce distinct reasoning behaviours (error patterns, memory traces)?

**RQ3**: Is there an optimal memory capacity for multi-step reasoning under bounded constraints?

## Hypothesis

> Moderate working-memory constraints may promote more selective and structured reasoning, whereas overly restrictive constraints will degrade performance due to insufficient retained information.

## Recommended Next Steps

1. **Run primary matrix first**: `configs/core_main_matrix.json` via `run_full_core_matrix.py`
2. **Run cognitive extension second**: `configs/extension_cognitive_modes.json` via `run_extension_cognitive.py`
3. **Generate reproducible tables/plots**: `python analyze_results.py --input data/full_run_core_matrix_results.jsonl --out-dir results` then `python plot_results.py --results-dir results`
4. **Inspect qualitative traces**: `python visualize_buffer.py --input <results.jsonl> --task-id <id>`

## Citation

If you use this framework in research, please cite:

```bibtex
@unpublished{dissertation2026,
  title={Simulating Human Working Memory Constraints in Large Language Models},
  author={Rishi Ranjan},
  school={University of Surrey},
  year={2026},
  note={Undergraduate Dissertation}
}
```

## License

This project is provided for academic and research purposes.

## Contact

For questions or feedback, contact the project author or supervisor.

---
