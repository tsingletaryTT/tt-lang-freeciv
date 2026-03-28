# TT-Lang FreeCiv Integration Plan

**Updated Approach:** Claude directly explores and implements (not Hermes)

## Phase 1: TT-Lang Environment ✅

Already complete:
- Built in simulator mode at ~/code/tt-lang/
- Tested with examples/eltwise_add.py → PASSED!
- Virtual environment ready

## Phase 2: Explore Codebases (NEW FOCUS)

**TT-Lang Exploration:**
- [x] Understand kernel structure (3-thread model)
- [ ] Study examples/ directory patterns
- [ ] Identify available math operations
- [ ] Understand dataflow buffer semantics

**FreeCiv Exploration:**
- [ ] Clone repository
- [ ] Find AI decision-making code
- [ ] Find map generation system
- [ ] Find pathfinding implementation
- [ ] Identify integration hooks

## Phase 3: Implement First Kernel

**Target:** Map generation (simplest)

**Steps:**
1. Design: Perlin noise terrain generation
2. Implement: TT-Lang kernel
3. Test: Simulator validation
4. Verify: Output correctness

**Why this first?**
- No game state dependency
- Clear input/output (seed → terrain array)
- Easy to visualize
- Builds confidence

## Phase 4: FreeCiv Integration

**Approach:**
- Python bridge layer
- Call TT-Lang kernel from Python
- Convert output to FreeCiv format
- Test in FreeCiv game

## Phase 5: Demo & Documentation

**Deliverables:**
- Working map generation demo
- Performance comparison (CPU vs TT-Lang)
- Documentation
- Potential: Add AI or pathfinding kernel

## Success Criteria

**Minimum Viable Demo:**
1. ✅ TT-Lang environment working
2. ⏳ One kernel implemented and tested
3. ⏳ FreeCiv can use kernel output
4. ⏳ Visual demo (generated map in FreeCiv)
5. ⏳ Documentation of approach

**Time Estimate:** 4-8 hours for MVP
