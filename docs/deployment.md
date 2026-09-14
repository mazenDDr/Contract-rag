# Deployment

| What | Where |
|---|---|
| Recorded answers: all 70 test questions, graded, with citations and failure causes | <https://mazenddr.github.io/Contract-rag/demo/> |
| The field guide and the tour | <https://mazenddr.github.io/Contract-rag/> |
| The live system: model, API and page | the all-in-one Docker image, on any machine (below) |

## Why the online demo is recorded

Hosting the live system needs a machine that can hold a 4B model, about 16 GB of memory. Free hosts are too small. Hugging Face Docker Spaces, which the all-in-one image was built for, now need a paid plan: a push returns 402, "hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription". So the public demo replays the evaluation run instead. It's built by `scripts/build_site.py` from `runs/matrix-v3` and `runs/failure-analysis-v3`, and it shows the served setup's real answers, grades, statement checks, cited excerpts and failure categories. It costs nothing and answers instantly, but it can't take new questions.

## The all-in-one image

`deploy/space/Dockerfile` puts everything in one container:

- Ollama with `qwen3.5:4b` baked in, so a cold start doesn't download it. Only Ollama's CPU libraries are copied, not its CUDA libraries.
- The API (`api/main.py --root-path /api`).
- The Streamlit page.
- nginx on port 7860, routing `/api/` to the API and everything else to the page.

`deploy/space/start.sh` starts all four and exits if any of them stops, so a supervisor restarts the whole container rather than running half of it. The data the served setup needs is baked in: parsed documents, fixed chunks and their BM25 index, 13.7 MB.

```bash
PYTHONPATH=src .venv/bin/python scripts/push_space.py     # assemble build/space/ (needs the built data)
docker build -t contract-rag-space build/space
docker run -p 7860:7860 -e APP_API_KEY=choose-a-key contract-rag-space
```

| Path | What it serves |
|---|---|
| `/` | The Streamlit page: pick a contract, ask, open each citation |
| `/api/docs` | The FastAPI service's interactive docs |
| `/api/health` | Open health check |
| `/api/contracts`, `/api/ask` | Need the `X-API-Key` header |

The same image fits a Hugging Face Space on a paid plan. `APP_API_KEY=... PYTHONPATH=src .venv/bin/python scripts/push_space.py --push OWNER/NAME` creates the Space, sets its key secret and uploads the build.

**Local check** (Docker Desktop on the laptop, CPU only in the container):
- the Kubient × Associated Press termination question was answered correctly, citing [5] on page 1;
- search took 3.4 ms and the answer 47 s;
- `/api/contracts` without a key returned 401;
- `/`, `/api/docs` and `/api/health` returned 200;
- every request wrote one JSON log line.

## Why the image runs the 4B model, although it is slow on CPU

Without a GPU, almost all of an answer's time goes into reading the prompt: 8 excerpts of up to 512 tokens, about 4,400 tokens in all.

**Speed.** One real question (q0078), CPU only, on an Apple M4 Pro:

| Model | 2 threads | 4 threads |
|---|---|---|
| `qwen3.5:4b` | 164 s | 121 s |
| `qwen3.5:2b` | 48 s | 44 s |

**Quality.** The smaller sizes were run through the same evaluation as the 4B. The setup is the served one (fixed chunks, BM25, contract scope, top 8), with the same 70 test questions, the `gemma4:12b` grader and the `contradiction-only-v5` rubric:

| Model | Correctness | Faithfulness | Citation validity | Abstention accuracy | Run |
|---|---|---|---|---|---|
| `qwen3.5:4b` | **0.75** [0.64, 0.84] | **0.62** | **0.87** | 0.94 | `runs/matrix-v3` |
| `qwen3.5:2b` | 0.53 [0.41, 0.63] | 0.08 | 0.14 | 0.94 | `runs/matrix-deploy-2b` |
| `qwen3.5:0.8b` | 0.32 [0.22, 0.42] | 0.03 | 0.06 | 0.80 | `runs/matrix-deploy-0.8b` |

The 2B answered with no citation at all in 52 of its 70 answers. Citing the clause is the point of the project, so the image serves the 4B that every reported result was measured with.

To reproduce the small-model runs:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_matrix.py --config configs/matrix-deploy.yaml --run-dir runs/matrix-deploy-2b
# 0.8B: the same config with generator.model: qwen3.5:0.8b and report_path under runs/matrix-deploy-0.8b/
```

## The API image without a model

The root `Dockerfile` builds the API alone (610 MB, no model), for running next to an existing Ollama server. See `docs/api.md`.
