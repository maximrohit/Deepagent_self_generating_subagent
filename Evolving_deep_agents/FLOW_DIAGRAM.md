# Evolving Deep Agents — Flow Diagrams

Self-evolving multi-agent system: warm-boot registry → topic analysis → TODO plan →
execute (reuse / create / invoke) → draft → validate → finalize.

---

## 1. System overview

```mermaid
flowchart TB
    subgraph Boot["Warm boot (main.py)"]
        M[main.py] --> CFG[config.py + .env]
        CFG --> REG[SubAgentRegistry.warm_boot]
        REG --> SA["sub_agents/*.py + manifest.json"]
        REG --> DL[dynamic_loader.py]
        M --> FM[FlowManifest.load]
        FM --> FL["flows/manifest.json"]
        M --> G[build_supervisor_graph]
    end

    subgraph Runtime["Per user task"]
        U[User question] --> INV[app.invoke]
        INV --> SG[Supervisor StateGraph]
        SG --> OUT[Final answer]
    end

    Boot --> Runtime

    subgraph Shared["Shared utilities"]
        LLM["utility/llm.py — Model ChatOllama"]
        TAV["utility/tavily_tools.py — search tools"]
        GRD["utility/guardrails.py — honesty / macro rules"]
    end

    SG -.-> LLM
    SA -.-> LLM
    SA -.-> TAV
    SG -.-> GRD
```

---

## 2. Supervisor graph (runtime)

```mermaid
flowchart TD
    START([START]) --> ANALYZE[analyze_topic_node]
    ANALYZE --> PLAN[plan_todos_node]

    ANALYZE -.- A1["Entities + entity types"]
    ANALYZE -.- A2["MACRO domain comics/finance/fitness/general"]
    ANALYZE -.- A3["DO / DON'T lists"]
    ANALYZE -.- A4["Check agent + flow manifests"]
    ANALYZE -.- A5["Reuse vs CREATE_NEEDED"]

    PLAN --> EXEC[execute_todos_node]
    PLAN -.- P1["TODOs grounded in topic briefing"]
    PLAN -.- P2["Sanitize: no finance agents on comics"]

    EXEC --> DRAFT[draft_synthesize_node]
    DRAFT --> VAL[validate_query_node]

    VAL -->|PASS or iter == max| FIN[finalize_node]
    VAL -->|FAIL and iter &lt; max| PLAN

    FIN --> END([END — final answer])

    subgraph ExecDetail["Inside execute_todos_node (per TODO)"]
        E1{Agent assigned<br/>and domain OK?}
        E1 -->|yes| E2[Invoke compiled sub-agent]
        E1 -->|no / CREATE_NEEDED| E3[create_sub_agent_with_critic_loop]
        E3 -->|created| E2
        E3 -->|aborted| E4[Fallback primary_deep_agent<br/>never wrong-domain finance]
        E4 --> E2
    end

    EXEC -.-> ExecDetail
```

---

## 3. Sub-agent creation pipeline (`code_generator.py`)

```mermaid
flowchart TD
    IN[CREATE_NEEDED: macro domain name] --> S1[1. design_capability_spec]
    S1 --> S1b{Missing niche?}
    S1b -->|fully covered| REUSE[Register reuse-only flow — no new .py]
    S1b -->|missing| S1c[Macro-name enforce<br/>batman → comics_lore_agent]

    S1c --> S1d[1b. research_prompt_brief<br/>Tavily + LLM on entities/attrs<br/>NOT the one-shot user ask]
    S1d --> LOOP

    subgraph LOOP["Self-correct loop max 3"]
        S2[2. generate_sub_agent_code<br/>embeds researched system_prompt_core]
        S2r[Deterministic repair<br/>Model import / deepagents / tools]
        S3[3. validate_contract]
        S4[4. critique_sub_agent]
        S4n[Normalize critic<br/>ignore bogus primary overlap]
        S2 --> S2r --> S3 --> S4 --> S4n
        S4n -->|FAIL| S2
        S4n -->|PASS| OK
    end

    OK[OK] --> S6[6. persist_and_register]
    S6 --> S6b{Import/register OK?}
    S6b -->|yes| S7[7. register_composed_flow]
    S6b -->|no| DEL[Delete broken file — raise]
    S7 --> DONE[Hot-registered agent + flow]
```

---

## 4. Module map

```mermaid
flowchart LR
    subgraph Entry
        main.py
    end

    subgraph Orchestration
        graph.py
        state.py
        code_generator.py
    end

    subgraph Persistence
        registry.py
        dynamic_loader.py
        flow_manifest.py
        sub_agents["sub_agents/"]
        flows["flows/"]
    end

    subgraph Utility
        llm["utility/llm.py"]
        tavily["utility/tavily_tools.py"]
        guard["utility/guardrails.py"]
        config.py
    end

    main.py --> graph.py
    main.py --> registry.py
    main.py --> flow_manifest.py
    graph.py --> state.py
    graph.py --> code_generator.py
    graph.py --> registry.py
    graph.py --> flow_manifest.py
    code_generator.py --> registry.py
    code_generator.py --> flow_manifest.py
    registry.py --> dynamic_loader.py
    registry.py --> sub_agents
    flow_manifest.py --> flows
    graph.py --> llm
    code_generator.py --> llm
    sub_agents --> llm
    sub_agents --> tavily
    graph.py --> guard
```

---

## 5. Data artifacts

| Path | Role |
|------|------|
| `sub_agents/*.py` | Persisted specialist agents (`build_agent`, `AGENT_*`) |
| `sub_agents/manifest.json` | Agent name / description / capabilities index |
| `flows/manifest.json` | Multi-agent workflows + problem coverage |
| `.env` | `TAVILY_API_KEY`, Ollama settings via `config.py` |

---

## 6. End-to-end example (comics question)

```mermaid
sequenceDiagram
    participant U as User
    participant M as main.py
    participant A as analyze_topic
    participant P as plan_todos
    participant E as execute_todos
    participant C as code_generator
    participant R as registry
    participant F as finalize

    U->>M: Batman vs Super Commando Dhruv?
    M->>A: task
    A->>A: domain=comics, entities=[Batman, Dhruv]
    A->>A: CREATE_NEEDED comics_lore_agent
    A->>P: topic briefing
    P->>E: TODOs unassigned + CREATE_NEEDED
    E->>C: create comics_lore_agent
    C->>C: prompt research (entities/attrs) → codegen → repair → validate → critic
    C->>R: persist + hot-register
    E->>R: invoke comics_lore_agent
    E->>F: draft → validate → final
    F->>U: answer with/without prep
```
