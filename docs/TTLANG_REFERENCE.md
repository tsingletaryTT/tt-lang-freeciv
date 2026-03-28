# TT-Lang Translation Skill

**Skill Name:** ttl_translate
**Purpose:** Translate GPU kernels (CUDA, Triton) or PyTorch/TTNN code to TT-Lang
**Status:** Complete

## Overview

This skill helps you translate GPU kernels and PyTorch/TTNN operations into TT-Lang kernels that run on Tenstorrent hardware.

**Common use case:** Fusing multiple TTNN operations into a single TT-Lang kernel for better performance.

## TT-Lang Programming Model

### Three-Thread Architecture

Every TT-Lang kernel has exactly **three threads** that run concurrently:

1. **Compute thread** (`@ttl.compute()`): Performs math operations on tiles in L1 memory
2. **Reader thread** (`@ttl.datamovement()`): Loads data from DRAM to circular buffers
3. **Writer thread** (`@ttl.datamovement()`): Writes data from circular buffers to DRAM

These threads **synchronize via circular buffers** (dataflow buffers, DFBs).

### Basic Kernel Template

```python
import ttl
import ttnn

@ttl.kernel(grid=(1, 1))  # Grid size: (cols, rows) of cores
def my_kernel(input1, input2, output):
    # Create dataflow buffers (circular buffers for synchronization)
    in1_dfb = ttl.make_dataflow_buffer_like(input1, shape=(1, 1), buffer_factor=2)
    in2_dfb = ttl.make_dataflow_buffer_like(input2, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(output, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        # Use context managers (automatically handles pop/push)
        with in1_dfb.wait() as a, in2_dfb.wait() as b:
            with out_dfb.reserve() as o:
                result = a + b  # Your computation here
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        # Read from DRAM to circular buffers
        with in1_dfb.reserve() as blk:
            tx = ttl.copy(input1[0, 0], blk)
            tx.wait()
        with in2_dfb.reserve() as blk:
            tx = ttl.copy(input2[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        # Write from circular buffers to DRAM
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, output[0, 0])
            tx.wait()

# Call kernel directly (no need to return ttl.Program)
# my_kernel(tensor1, tensor2, output_tensor)
```

## Translation Patterns

### Pattern 1: Fusing TTNN Operations

**Before (TTNN - multiple ops, multiple round trips):**
```python
x = ttnn.exp(input)
y = ttnn.add(x, bias)
z = ttnn.relu(y)
```

**After (TT-Lang - single fused kernel):**
```python
@ttl.kernel(grid=(1, 1))
def fused_kernel(input, bias, output):
    input_dfb = ttl.make_dataflow_buffer_like(input, shape=(1, 1), buffer_factor=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(output, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with input_dfb.wait() as inp, bias_dfb.wait() as b:
            with out_dfb.reserve() as o:
                # Chain operations in single compute function
                result = ttl.math.relu(ttl.math.exp(inp) + b)
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        with input_dfb.reserve() as blk:
            tx = ttl.copy(input[0, 0], blk)
            tx.wait()
        with bias_dfb.reserve() as blk:
            tx = ttl.copy(bias[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, output[0, 0])
            tx.wait()
```

### Pattern 2: Simple Element-Wise Operation

**CUDA-style thinking:**
```c
__global__ void add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}
```

**TT-Lang equivalent:**
```python
@ttl.kernel(grid=(1, 1))
def add_kernel(a, b, c):
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with a_dfb.wait() as av, b_dfb.wait() as bv:
            with c_dfb.reserve() as cv:
                cv.store(av + bv)

    @ttl.datamovement()
    def dm_read():
        with a_dfb.reserve() as blk:
            tx = ttl.copy(a[0, 0], blk)
            tx.wait()
        with b_dfb.reserve() as blk:
            tx = ttl.copy(b[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with c_dfb.wait() as blk:
            tx = ttl.copy(blk, c[0, 0])
            tx.wait()
```

### Pattern 3: Reduction Operation

For operations like sum, max, mean across dimensions:

```python
@ttl.kernel(grid="auto")
def reduce_sum_kernel(input, output):
    # Use multiple cores if grid="auto"
    in_dfb = ttl.make_dataflow_buffer_like(input, shape=(1, 1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(output, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with in_dfb.wait() as inp:
            with out_dfb.reserve() as o:
                # Use TTL math operations
                result = ttl.math.reduce_sum(inp, dim=1)
                o.store(result)

    @ttl.datamovement()
    def dm_read():
        # Read tiles to process
        with in_dfb.reserve() as blk:
            tx = ttl.copy(input[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, output[0, 0])
            tx.wait()
```

## Concept Mapping

### GPU → TT Hardware

| GPU Concept | TT-Lang Equivalent | Notes |
|------------|-------------------|-------|
| Thread block | Core in grid | `grid=(cols, rows)` |
| Shared memory | L1 memory | Accessed via DFBs |
| Global memory | DRAM | Accessed via ttl.copy() |
| `__syncthreads()` | DFB synchronization | wait()/reserve() |
| Thread ID | Implicit in grid | Access via grid coordinates |

### CUDA → TT-Lang

| CUDA Pattern | TT-Lang Pattern |
|--------------|----------------|
| `blockIdx.x, threadIdx.x` | Grid position (implicit in data indexing) |
| `__shared__ float[]` | Dataflow buffer (DFB) |
| `atomicAdd()` | Not directly supported (use different approach) |
| `for (i = 0; i < n; i++)` | Loop in compute function |

### PyTorch → TT-Lang

| PyTorch Op | TT-Lang Equivalent |
|------------|-------------------|
| `torch.add(a, b)` | `a + b` in compute |
| `torch.exp(x)` | `ttl.math.exp(x)` |
| `torch.relu(x)` | `ttl.math.relu(x)` |
| `torch.matmul(a, b)` | `ttl.math.matmul(a, b)` |
| `torch.sum(x, dim=1)` | `ttl.math.reduce_sum(x, dim=1)` |

## Translation Workflow

### Step 1: Analyze Source Code

**Questions to answer:**
- What are the inputs and outputs?
- What computation is performed?
- Are there multiple operations that can be fused?
- What's the data flow pattern?

### Step 2: Design TT-Lang Structure

**Decisions:**
1. **Grid size**: How many cores? Start with `(1, 1)` or use `grid="auto"`
2. **Dataflow buffers**: One per input/output tensor
3. **Buffer factor**: Usually 2 (double buffering)
4. **Tile shape**: Match input tensor layout

### Step 3: Write Compute Function

**Map operations:**
- Basic math: `+`, `-`, `*`, `/` work directly
- Special functions: Use `ttl.math.*` (exp, relu, sin, cos, etc.)
- Reductions: Use `ttl.math.reduce_*`

**Example:**
```python
@ttl.compute()
def compute():
    with input_dfb.wait() as inp:
        with output_dfb.reserve() as out:
            # Your computation here
            result = ttl.math.exp(inp) + 1.0
            out.store(result)
```

### Step 4: Write Data Movement

**Reader pattern (standard template):**
```python
@ttl.datamovement()
def dm_read():
    with input_dfb.reserve() as blk:
        tx = ttl.copy(input_tensor[row, col], blk)
        tx.wait()
```

**Writer pattern (standard template):**
```python
@ttl.datamovement()
def dm_write():
    with output_dfb.wait() as blk:
        tx = ttl.copy(blk, output_tensor[row, col])
        tx.wait()
```

### Step 5: Test in Simulator

```bash
cd /home/ttuser/code/tt-lang
source build/env/activate
./bin/ttlang-sim your_kernel.py
```

### Step 6: Iterate

**Common issues:**
- **Shape mismatches**: Verify tensor shapes match DFB configuration
- **Synchronization errors**: Check wait()/reserve() pairs
- **Unsupported ops**: Check examples/ for similar patterns

## Available Math Operations

**Basic arithmetic:**
- `+`, `-`, `*`, `/` (element-wise)

**TT-Lang math functions:**
- `ttl.math.exp(x)`
- `ttl.math.log(x)`
- `ttl.math.sqrt(x)`
- `ttl.math.relu(x)`
- `ttl.math.sigmoid(x)`
- `ttl.math.tanh(x)`
- `ttl.math.sin(x)`, `ttl.math.cos(x)`
- `ttl.math.abs(x)`
- `ttl.math.pow(x, y)`

**Reductions:**
- `ttl.math.reduce_sum(x, dim=...)`
- `ttl.math.reduce_max(x, dim=...)`
- `ttl.math.reduce_mean(x, dim=...)`

**Matrix operations:**
- `ttl.math.matmul(a, b)`

## Key Constraints

### What TT-Lang CAN'T Do (or not yet)

- **No atomic operations** (atomicAdd, atomicMax, etc.)
- **No dynamic memory allocation** in kernel
- **No recursion** in compute functions
- **No dynamic branching** (some `if` supported, but limited)

### Workarounds

**Instead of atomics:**
- Use separate kernels for accumulation
- Restructure algorithm to avoid contention

**Instead of dynamic branching:**
- Use masking operations
- Split into multiple kernels if necessary

## Examples to Reference

**Location:** `/home/ttuser/code/tt-lang/examples/`

**Key examples:**
1. `eltwise_add.py` - Simple addition
2. `matmul.py` - Matrix multiplication
3. `tutorial/single_node_single_tile_block.py` - Basic structure
4. `tutorial/` directory - Progressive complexity

## Integration with Hermes

When asked to translate a kernel, Hermes should:

1. **Understand source code**:
   - Identify inputs, outputs, operations
   - Note data dependencies

2. **Generate TT-Lang skeleton**:
   - Start from template
   - Create DFBs for each input/output
   - Map operations to compute function

3. **Test immediately**:
   ```bash
   cd /home/ttuser/code/tt-lang && source build/env/activate && ./bin/ttlang-sim translated_kernel.py
   ```

4. **Iterate on errors**:
   - Parse error messages
   - Suggest fixes based on error type
   - Reference examples/ for patterns

5. **Validate correctness**:
   - Compare output to original implementation
   - Test edge cases
   - Verify performance expectations

## Quick Reference

**Kernel skeleton:**
```python
import ttl

@ttl.kernel(grid=(1, 1))
def kernel_name(in1, in2, out):
    # Create DFBs
    in1_dfb = ttl.make_dataflow_buffer_like(in1, shape=(1,1), buffer_factor=2)
    in2_dfb = ttl.make_dataflow_buffer_like(in2, shape=(1,1), buffer_factor=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1,1), buffer_factor=2)

    @ttl.compute()
    def compute():
        with in1_dfb.wait() as a, in2_dfb.wait() as b:
            with out_dfb.reserve() as o:
                o.store(a + b)  # Your op here

    @ttl.datamovement()
    def dm_read():
        with in1_dfb.reserve() as blk:
            ttl.copy(in1[0,0], blk).wait()
        with in2_dfb.reserve() as blk:
            ttl.copy(in2[0,0], blk).wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            ttl.copy(blk, out[0,0]).wait()
```

## References

- **Comprehensive guide**: `/home/ttuser/code/tt-lang/claude-slash-commands/ttl-import.md` (1848 lines)
- **Examples**: `/home/ttuser/code/tt-lang/examples/`
- **Programming guide**: `/home/ttuser/code/tt-lang/docs/sphinx/programming-guide.md`

---

**Status:** ✅ Complete and ready for use
