# Defcon33 Model Security Lab — AI/ML Attacks & Defenses

[![Releases](https://img.shields.io/github/v/release/Rxcl536/defcon33-model-security-lab?label=Releases&style=for-the-badge)](https://github.com/Rxcl536/defcon33-model-security-lab/releases)

![Lab Visual](https://images.unsplash.com/photo-1535223289827-42f1e9919769?auto=format&fit=crop&w=1650&q=80)

AI/ML model security experiments and exploit demos inspired by DEF CON 33 presentations. This repo collects practical examples and supporting tooling for research into pickle RCE, TorchScript exploitation, ONNX injection, model poisoning, and integrated LLM attacks using PromptMap2. Use the releases link above to download and run the packaged artifacts; the release assets must be downloaded and executed to reproduce the demos.

Table of Contents
- Overview
- Key topics & badges
- What you get
- Demos and attack catalog
- Architecture and data flow
- Quick start (download & run)
- Repro steps and examples
- Detection and mitigation notes
- Research methodology
- Contributing
- License and credits
- Releases and downloads

Overview
This repository documents methods to compromise machine learning models and pipelines, with reproducible demos and small utilities. The work targets common deployment surfaces: serialized artifacts, scriptable model formats, model conversion pipelines, and prompt chains in LLM systems. Each demo pairs an exploit with an analysis and suggested mitigations. The artifacts mirror real-world patterns used in production and research.

Key topics & badges
- Topics: ai-security, defcon, llm-security, machine-learning, model-security, pickle, prompt-injection, pytorch, safetensors, security-research
- Platform badges:
  - [![PyPI](https://img.shields.io/pypi/v/defcon33-model-security-lab?style=flat-square)](https://pypi.org/)
  - [![Issues](https://img.shields.io/github/issues/Rxcl536/defcon33-model-security-lab?style=flat-square)](https://github.com/Rxcl536/defcon33-model-security-lab/issues)

What you get
- Step-by-step demos for:
  - Pickle-based remote code execution (RCE) triggered via model deserialization.
  - TorchScript exploitation: crafted script modules that execute unintended operations when loaded or run.
  - ONNX injection: manipulation of graph inputs, initializers, or exporter metadata to alter runtime behavior.
  - Model poisoning: gradients and data pipeline attacks that degrade or backdoor model behavior.
  - Integrated LLM attacks using PromptMap2: prompt chaining and context injection against multi-model systems.
- Small utilities to inspect serialized models, verify safetensors usage, and compare conversion artifacts.
- Reproducible data and minimal models used in presentations and tests.
- Release bundles that contain pre-built artifacts and scripts for offline reproduction. Download and execute the release assets from the Releases page.

Demos and attack catalog
1. Pickle RCE
   - Target: Python model artifacts saved with `pickle`/`joblib`.
   - Vector: crafted `__reduce__` payloads embedded in model object state.
   - Impact: code execution during unpacking or model initialization.
   - Mitigation: avoid untrusted pickle, prefer safetensors or strict JSON schemas, validate artifact origins.
2. TorchScript exploitation
   - Target: `torch.jit` scripted modules and traced graphs.
   - Vector: malicious module constants or attributes that run side effects on load or forward.
   - Impact: data exfiltration, local filesystem access, process spawning.
   - Mitigation: verify model provenance, sandbox model execution, use restricted execution runtime.
3. ONNX injection
   - Target: ONNX graph files and conversion tools.
   - Vector: crafted nodes, initializer tensors, or metadata with unexpected types or shapes; malicious custom ops.
   - Impact: altered inference, hidden I/O channels, crash or memory corruption in runtimes.
   - Mitigation: sanitize conversion steps, validate graph schema, run on hardened runtimes.
4. Model poisoning
   - Target: training pipelines and federated updates.
   - Vector: label flipping, targeted backdoor triggers, gradient replacement.
   - Impact: targeted misclassification, stealthy backdoors, degraded model utility.
   - Mitigation: robust aggregation, anomaly detection on updates, hold-out validation sets.
5. Integrated LLM attacks with PromptMap2
   - Target: multi-LLM orchestration and prompt routing systems.
   - Vector: prompt injection, context poisoning, nested prompt maps to escalate privileges or leak secrets.
   - Impact: model reveals sensitive context, performs unauthorized operations via integrated agents.
   - Mitigation: prompt sanitization, instruction filtering, least-privilege orchestration.

Architecture and data flow
- The lab uses a modular layout:
  - /artifacts: serialized models in different formats (pickle, torchscript, onnx, safetensors).
  - /demos: attacker scripts and victim loaders.
  - /tools: inspectors, converters, and validators.
  - /docs: slides, notes, and mapping of DEF CON presentation content to demos.
- Data flow example: attacker crafts a malicious serialized artifact -> attacker uploads to a model registry or shares via download link -> victim downloads and loads the artifact with an unsafe loader -> the loader triggers the payload or modified behavior -> attacker achieves objective.

Quick start (download & run)
- Visit the releases page. The release assets contain packaged demos and ready-to-run scripts. The release files must be downloaded and executed to reproduce the scenarios.
- Release link: https://github.com/Rxcl536/defcon33-model-security-lab/releases
- Example quick steps:
  - Download the release bundle labeled `defcon33-lab-v1.zip` from the Releases page.
  - Extract the bundle.
  - From the extracted folder, run `python demo_picklerce.py` or `python demo_torchscript_exploit.py` as documented in `README` inside the bundle.
- The releases badge at the top links to the same page for convenience.

Repro steps and examples
- Pickle RCE (conceptual)
  - Create a malicious payload object with a custom `__reduce__` that calls `os.system` or spawns a reverse shell.
  - Save the object with `pickle.dump`.
  - Load with `pickle.load` in a victim script that expects a model. The load call triggers payload execution.
  - Example loader flow: victim calls `model = pickle.load(open('model.pkl','rb'))` and then `model.predict(data)`.
- TorchScript exploit (conceptual)
  - Script a small module that stores a file path in a constant and calls a side-effect function during `forward`.
  - Export with `torch.jit.script` and save with `torch.jit.save`.
  - Victim loads the module with `torch.jit.load('bad.pt')`. If the module uses side-effect code on load or first forward, it executes.
- ONNX injection (conceptual)
  - Modify ONNX initializer arrays to include attacker-controlled values that change control-flow behavior in downstream runtime.
  - Insert a custom operator node that runtime maps to a vulnerable native plugin.
- PromptMap2 integrated chain
  - Build a map of role-to-prompt templates. Introduce a crafted context that injects instructions into a chained LLM workflow.
  - The orchestrator concatenates prompts without sanitization, exposing hidden directives to downstream models.

Detection and mitigation notes
- Validate artifacts:
  - Use cryptographic signing for model artifacts and verify signatures before load.
  - Prefer formats with limited interpretive behavior; safetensors removes code path risks in many cases.
- Isolate execution:
  - Run models inside restricted containers, with limited network and filesystem access.
  - Use syscall filters or sandboxed runtimes for heavy untrusted workloads.
- Monitor anomalies:
  - Track model accuracy drift, unusual resource usage, and outbound network calls during inference.
  - Maintain provenance metadata for training data and model updates.
- Pipeline hardening:
  - Implement gating checks on model conversion and export steps.
  - Run automated validators on ONNX graphs and TorchScript modules.

Research methodology
- Reproducibility: each demo includes a minimal model or dataset and an attacker script. The release bundles include pre-built artifacts to shorten setup time.
- Minimalism: demos use minimal dependencies to isolate the attack vector. They aim to show the core idea and failure mode.
- Mapping to DEF CON material: slides, lab notes, and presentation excerpts accompany each demo in /docs. The lab expands on examples shown during the DEF CON 33 talks.
- Ethical use: the lab targets research and defensive testing on owned or permitted systems. The artifacts make it easy to test detection and mitigation techniques.

Contributing
- Please open issues for bug reports or experiment ideas.
- Suggested workflow:
  - Fork the repo.
  - Add a new demo in `/demos` with clear README and a small set of artifacts.
  - Provide a short threat model and suggested mitigations.
  - Create PR with tests and docs updates.
- Maintain coding style: keep examples small, clear, and well commented. Use deterministic seeds for training artifacts where used.

License and credits
- The repository uses an OSI-compatible license in the root LICENSE file.
- Credits:
  - DEF CON presenters and slides that inspired the experiments.
  - Open-source projects: PyTorch, ONNX, Transformers, and related tooling used for demos.
  - Community contributors who provided reproductions and hardening suggestions.

Releases and downloads
- The releases page packages reproducible assets for offline use. Download and execute the release artifacts to run the demos.
- Release page: https://github.com/Rxcl536/defcon33-model-security-lab/releases
- If the release link is unavailable, check the repository "Releases" section on GitHub for tagged builds and release notes.

Images and visuals
- The repo includes slide exports and architecture PNGs under `/docs/images`. Use them as visual aids when presenting attack chains.
- Example image sources:
  - Unsplash network/security imagery for cover illustrations.
  - Local diagram exports for flow charts and threat models.

Appendix: artifact checklist
- Model artifacts
  - `model_pickle_demo.pkl` — Pickle-serialized object demonstrating RCE pattern.
  - `torchscript_bad.pt` — TorchScript module with side-effect code.
  - `onnx_injected.onnx` — ONNX file with injected initializers/nodes.
  - `safetensors_sample.safetensors` — Safe tensor sample used for baseline comparison.
- Tools
  - `inspect_model.py` — Lightweight inspector for artifact metadata (format, size, ops).
  - `convert_and_check.py` — Conversion helper that verifies schema after conversion.
  - `promptmap2_demo/` — LLM orchestration examples using PromptMap2 patterns.

Contact and further reading
- See the /docs folder for slides, references, and extended notes.
- Open issues for follow-up experiments and reproducibility requests.