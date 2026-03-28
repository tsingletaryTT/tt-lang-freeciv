# Exploration Session Notes

## 2026-03-25: Project Pivot

**Original Plan:** Build Hermes agent to implement TT-Lang + FreeCiv demo
**Reality Check:** Hermes better suited for simpler tasks, ran into repeated setup loops
**New Plan:** Claude directly explores and implements

### What We Built for Hermes

Created comprehensive skills:
- ttl_build (8KB) - Build instructions
- ttl_run_sim (7KB) - Run kernels in simulator  
- ttl_translate (11.6KB) - GPU → TT-Lang translation
- ttl_profile (8KB) - Performance profiling
- ttl_debug (9.7KB) - Debugging workflows

**Outcome:** Great reference documentation, but Hermes not ideal for exploratory implementation

### New Approach

**Strengths we're leveraging:**
- Claude: Better at exploration, connecting concepts, implementing novel solutions
- Hermes: Good for repetitive tasks, following clear checklists
- TT-Lang: Well-suited for parallel game algorithms
- FreeCiv: Open source, extensible, visual feedback

**Next Steps:**
1. Clone FreeCiv
2. Explore both codebases in parallel
3. Design map generation kernel
4. Implement and test
5. Integrate with FreeCiv

## Environment Status

**TT-Lang:** ~/code/tt-lang/
- Built in simulator-only mode
- Simulator binary: bin/ttlang-sim
- Tested and working

**70B Model:** Running at localhost:8000
- Llama-3.3-70B-Instruct
- Available for consultation

**Hermes:** Configured at ~/.local/bin/hermes-tt
- Available as tool, not primary agent
- Skills preserved for reference

## Key Insights

**TT-Lang Architecture:**
- Three concurrent threads: compute, reader, writer
- Synchronization via dataflow buffers (circular buffers)
- Context managers handle pop/push automatically
- Use `grid="auto"` for automatic parallelization

**FreeCiv:** (to be explored)
- C codebase
- Server/client architecture
- Extensible AI system
- Procedural map generation

## What's in ~/code/tt-lang-hermes-spoke/

Reference materials:
- skills/ - 5 comprehensive TT-Lang skill guides
- HANDOFF.md - Complete handoff documentation
- PLAN.md - Original 6-phase plan
- HOW_TO_USE_HERMES.md - Hermes usage guide

**Status:** Archived for reference, focus now on ~/tt-lang-freeciv/
