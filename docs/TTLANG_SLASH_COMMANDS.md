# TT-Lang Slash Commands Integration

**Date:** 2026-03-25

## Overview

The official TT-Lang slash commands are now installed at `~/.claude/commands/`. These provide specialized workflows for kernel development.

## Installed Commands

```
~/.claude/commands/
├── ttl-help.md         # List all commands
├── ttl-import.md       # Translate CUDA/Triton/PyTorch → TT-Lang
├── ttl-export.md       # Export TT-Lang → TT-Metal C++
├── ttl-optimize.md     # Profile and optimize performance
├── ttl-profile.md      # Per-line cycle counts
├── ttl-simulate.md     # Run simulator and suggest improvements
├── ttl-test.md         # Generate test cases
└── ttl-bug.md          # File bug reports with reproducers
```

## Most Useful for Our Project

### 1. `/ttl-simulate` - Run Simulator

**Purpose:** Test our height map kernel and get suggestions for improvements.

**Usage:**
```
/ttl-simulate kernels/height_map_simple.py
```

**What it does:**
- Runs the kernel in ttlang-sim
- Validates correctness
- Suggests improvements

### 2. `/ttl-profile` - Performance Profiling

**Purpose:** Understand performance characteristics of our kernel.

**Usage:**
```
/ttl-profile kernels/height_map_simple.py
```

**What it does:**
- Reports per-line cycle counts
- Identifies compute-bound vs memory-bound sections
- Shows DRAM bandwidth utilization
- Analyzes thread balance

**Note:** Requires hardware or `--hw` flag for simulator profiling. Since we're using simulator-only mode, profiling may be limited.

### 3. `/ttl-test` - Generate Tests

**Purpose:** Create comprehensive test cases for our kernel.

**Usage:**
```
/ttl-test kernels/height_map_simple.py "test edge cases for different map sizes"
```

**What it does:**
- Generates test runner script
- Creates test cases for edge conditions
- Validates correctness across inputs

### 4. `/ttl-optimize` - Performance Optimization

**Purpose:** Profile and suggest optimizations.

**Usage:**
```
/ttl-optimize kernels/height_map_simple.py
```

**What it does:**
- Profiles the kernel
- Identifies bottlenecks
- Suggests specific optimizations
- May implement optimizations if requested

## Commands Less Relevant to Us

### `/ttl-import` - Translation

**Purpose:** Translate CUDA/Triton/PyTorch kernels to TT-Lang.

**Why we don't need it yet:**
- We wrote our kernel from scratch in TT-Lang
- No existing GPU code to translate

**Future use:** If we want to translate FreeCiv's existing CPU map generation code, this would be useful.

### `/ttl-export` - C++ Export

**Purpose:** Export TT-Lang kernel to production TT-Metal C++ code.

**Why we don't need it yet:**
- We're in prototype/demo phase
- Python TT-Lang is sufficient for now

**Future use:** For production deployment or performance comparison.

### `/ttl-bug` - Bug Reporting

**Purpose:** File bug reports with reproducers.

**When to use:**
- If we hit compiler crashes
- Unexpected behavior in simulator
- Runtime errors we can't explain

## Local vs Remote Execution

**Important:** These commands are designed for remote execution (SSH to a machine with TT-Lang hardware). Since we're working locally, we can:

1. **Option A: Use commands directly**
   - Commands will work with local file paths
   - May need to adjust remote.conf if tools complain

2. **Option B: Run tools manually**
   - The commands are just wrappers around tools
   - We can call tools directly:
     ```bash
     cd ~/code/tt-lang
     source build/env/activate
     ./bin/ttlang-sim ~/tt-lang-freeciv/kernels/height_map_simple.py
     ```

3. **Option C: Use commands as documentation**
   - Read the .md files for guidance
   - Implement the workflows manually
   - Best for understanding what's happening

## Current Setup Status

### ✅ Installed
- All 8 slash commands in `~/.claude/commands/`
- Helper tools in `~/.claude/commands/tools/`
- Basic remote.conf configuration

### ⚠️ Limitations
- Remote tools configured for local execution only
- Some commands expect hardware (profiling)
- We're in simulator-only mode

### 💡 Recommended Workflow

For our current phase (prototyping, exploration):

1. **Develop kernels** - Write/modify TT-Lang code directly
2. **Test manually** - Run `ttlang-sim` directly
3. **Use commands for guidance** - Read command .md files for best practices
4. **Use commands when appropriate** - `/ttl-test`, `/ttl-simulate` once kernels are more stable

## Examples for Our Project

### Example 1: Test Current Kernel

```bash
# Manual approach (what we've been doing)
cd ~/code/tt-lang
source build/env/activate
./bin/ttlang-sim ~/tt-lang-freeciv/kernels/height_map_simple.py

# Command approach (once we fix remote.conf)
/ttl-simulate kernels/height_map_simple.py
```

### Example 2: Generate Test Suite

Once our kernel is more complete:
```
/ttl-test kernels/height_map_simple.py "Test various map sizes (64x64, 128x128, 256x256, 512x512) and validate output range [0, 1000]"
```

### Example 3: Profile Performance

When we're ready to optimize:
```
/ttl-profile kernels/height_map_simple.py
```

### Example 4: Import FreeCiv CPU Code

If we want to translate FreeCiv's existing map generation:
```
# First extract relevant C code to a standalone function
# Then use ttl-import to get TT-Lang version
/ttl-import freeciv_map_gen_extracted.c
```

## Documentation Reference

**Command details:** Each .md file in `~/.claude/commands/` contains:
- Description and usage
- Prerequisites and constraints
- Step-by-step process
- Expected output

**Most comprehensive:** `ttl-import.md` (72KB) - Contains:
- TT-Lang programming model explanation
- Three-thread architecture details
- Concept mapping tables (CUDA → TT-Lang)
- Kernel structure templates
- Common patterns and examples

**Quick reference:** `ttl-help.md` - Lists all commands with brief descriptions

## Next Steps

1. **Fix remote.conf for local execution** (if we want to use the commands)
2. **Try `/ttl-simulate`** on our current kernel
3. **Generate tests** with `/ttl-test` once kernel is more complete
4. **Profile performance** when we move beyond passthrough implementation
5. **Consider importing** FreeCiv's map generation code with `/ttl-import`

## Resources

- Command files: `~/.claude/commands/*.md`
- Helper tools: `~/.claude/commands/tools/`
- TT-Lang examples: `~/code/tt-lang/examples/`
- Our kernels: `~/tt-lang-freeciv/kernels/`

---

**Status:** Commands installed, documentation created, ready to use for kernel development!
