# tiny-gpu Learning Plan

Seven projects, ordered so each one sets up the next. For every phase:

1. **Learn** – Claude explains the concepts listed under *What you need to know* (ask as many questions as you want).
2. **Predict** – before writing code, you say what you expect to change (cycle count, signals, results).
3. **Build** – you write the code. Claude gives hints, not solutions, unless you ask for one.
4. **Verify** – run the tests, read the trace in `test/logs/`, compare against your prediction.
5. **Review** – Claude reviews your diff and asks a few check-your-understanding questions.

Work on one git branch per phase (`git switch -c phase-1-kernels`) so every step is easy to diff and undo.

Run tests from PowerShell (WSL toolchain):

```bash
wsl -e bash -lc 'export PATH=$HOME/tinygpu-venv/bin:$HOME/.local/bin:$PATH; cd "/mnt/c/Users/youse/R projects/micro/tiny-gpu/tiny-gpu" && make test_matadd'
```

`make test_foo` runs `test/test_foo.py`, so a new test file is picked up automatically.

## Baseline (record every change here)

| Change | matadd cycles | matmul cycles | Notes |
|---|---|---|---|
| Original | 179 | 492 | 2 cores, 4 threads/block, 1-cycle memory |
| | | | |

---

## Phase 1 – Your first kernel by hand: vector subtraction

**Goal:** understand the instruction encoding by hand-assembling a kernel.

**What you need to know**
- The SIMT model: one program, many threads; `i = blockIdx * blockDim + threadIdx`.
- The 16-bit instruction formats (from `src/decoder.sv`):
  - `opcode[15:12] rd[11:8] rs[7:4] rt[3:0]` – ADD, SUB, MUL, DIV, LDR, STR, CMP
  - `opcode[15:12] rd[11:8] imm[7:0]` – CONST
  - `opcode[15:12] nzp[11:9] _ target[7:0]` – BRnzp (absolute jump target)
- Special registers: R13 = `%blockIdx`, R14 = `%blockDim`, R15 = `%threadIdx` (read-only).
- How dispatch splits 8 threads into 2 blocks of 4 over 2 cores.

**You build:** `test/test_vecsub.py` – `C[i] = A[i] - B[i]`, copied from `test_matadd.py` and edited.

**Done when:** the test passes, including one case where `B[i] > A[i]`. Predict the result first: what does an 8-bit subtraction give when it goes below zero?

**Stretch:** change the thread count to 6 (not a multiple of 4). Which threads run, and why? (Look at `thread_count` in `dispatch.sv` and `core.sv`.)

---

## Phase 2 – A Python assembler

**Goal:** stop hand-writing binary; learn what an assembler actually does.

**What you need to know**
- Tokenizing and parsing a line: mnemonic, registers, immediates (`#5`), special-register names.
- Two-pass assembly: pass 1 records label addresses, pass 2 encodes (needed for `BRn LOOP`).
- Bit packing with shifts and masks: `(op << 12) | (rd << 8) | (rs << 4) | rt`.
- Range checking: registers 0–15, immediates 0–255, writes to R13–R15 rejected.

**You build**
- `test/helpers/assembler.py` with `assemble(source: str) -> list[int]`.
- A small helper (for example `run_kernel(dut, program, data, threads)`) so tests stop copy-pasting the cycle loop.

**Done when:** assembling the source text of matadd and matmul gives exactly the same integers as the hand-written lists, checked with an assert. Then convert `test_vecsub` to use it.

**Stretch:** a disassembler (int → text) and use it in the trace log.

---

## Phase 3 – Kernels that need thought: dot product and ReLU

**Goal:** hit the limits of this hardware on purpose. Both kernels expose a real GPU problem.

### 3a. Dot product

**What you need to know**
- Reductions: combining many threads' results into one value.
- Synchronization: threads in a block run in lockstep, and every `STR` finishes (the WAIT state) before the next instruction. Different blocks get no ordering guarantee at all.
- Why real GPUs have shared memory, barriers (`__syncthreads`) and atomics.

**You build:** `test_dot.py` – each thread stores `A[i]*B[i]`, then one thread loops and sums the products. First do it with a single block (4 threads), then work out why the same approach is unsafe across 2 blocks.

**Done when:** the result is correct for 4 elements, and you can explain the multi-block race in your own words.

### 3b. ReLU – `out[i] = max(x[i], 0)`

**What you need to know**
- Two's complement: storing signed values in 8 bits (0xFF = -1).
- How `CMP` builds NZP flags in `alu.sv`, and whether that comparison is signed or unsigned. Find out by experiment.
- **Branch divergence:** what the scheduler does when threads disagree on a branch (`src/scheduler.sv` line ~104 picks the *last thread's* `next_pc`).

**You build:** `test_relu.py` using `CMP` + `BRn`, with input data that mixes positive and negative values *within one block*.

**Done when:** you have a test that **fails** and you can explain exactly why. You will fix it twice: branch-free in Phase 4 and properly in Phase 7. Keep this test.

---

## Phase 4 – Add an instruction (`MAX`, then `AND` / `SHL`)

**Goal:** your first hardware change, end to end.

**What you need to know**
- How control signals flow: decoder → registers/ALU/PC → write-back in UPDATE.
- The opcode map: `1010`–`1110` are free.
- **Gotcha:** `decoded_alu_arithmetic_mux` is 2 bits and all four values (ADD/SUB/MUL/DIV) are already taken. You must widen it to 3 bits in `decoder.sv`, `core.sv` and `alu.sv`. Missing one is a classic width bug, and finding it is part of the exercise.
- Signed vs unsigned comparison in SystemVerilog (`$signed()`), if you want a signed `MAX`.

**You build:** `MAX rd, rs, rt` (signed), plus assembler support and a unit test. Then rewrite ReLU as `MAX R, x, R_zero` with no branch at all.

**Done when:** `test_relu` passes with mixed signs, and matadd/matmul still pass (a regression check).

**Stretch:** `AND`, `OR`, `SHL`, `SHR`; an immediate-form `ADDI`.

---

## Phase 5 – A data cache

**Goal:** understand memory latency and caching, and measure the win.

**What you need to know**
- **First problem:** the Python memory model answers in 1 cycle, so a cache cannot help yet. You need to add configurable latency to `test/helpers/memory.py` (for example 10 cycles) and record the new baselines.
- Direct-mapped cache: index, tag, valid bit; hit vs miss; 8-bit addresses → small tables.
- Write policy: write-through (simple) vs write-back, and how stores interact with cached lines.
- Where to put it: per-core in front of the LSUs, or shared in front of `controller.sv`. Trade-offs: coherence, ports, contention.
- Valid/ready handshakes, so a hit can answer without going to memory.

**You build:** `src/cache.sv`, a direct-mapped, write-through cache, wired between the LSUs and the data memory controller.

**Done when:** all tests pass and matmul cycles drop measurably at 10-cycle memory latency. Explain *why* matmul benefits and matadd barely does (data reuse).

**Stretch:** hit/miss counters exposed to the test; 2-way set-associative.

---

## Phase 6 – Pipelining

**Goal:** overlap work instead of running every state one after another.

**What you need to know**
- Cycles per instruction today: count FETCH→DECODE→REQUEST→WAIT→EXECUTE→UPDATE in the trace.
- Pipelining concepts: stages, throughput vs latency, pipeline registers.
- Hazards: **control** (a branch makes the prefetched instruction wrong → flush), **data** (an instruction reading a register still being written).
- Start small: *prefetch* – while the current instruction executes, the fetcher already loads `PC+1`.

**You build:** step 1, a prefetch buffer in `fetcher.sv` with scheduler changes to skip FETCH on a hit; flush on taken branches. Step 2 (optional), overlap DECODE with the previous UPDATE.

**Done when:** every test still passes, cycles drop, and you can point to a cycle in the trace where a flush happens.

---

## Phase 7 – Branch divergence

**Goal:** make data-dependent `if`s correct. This is how real GPUs handle them.

**What you need to know**
- Active masks: one bit per thread; inactive threads don't write registers or memory.
- Per-thread `next_pc` already exists in `pc.sv`; the scheduler just ignores all but one.
- Reconvergence: when both sides of a branch finish, threads must rejoin. Approaches: SIMT stack (push mask + reconvergence PC) or the simpler "run the minimum PC among active threads" scheme.
- Why divergence costs performance: both paths run back to back.

**You build:** an active mask in the scheduler/core, gating `enable` for registers, ALU, LSU and PC writes, plus a reconvergence scheme (start with min-PC).

**Done when:** the *original branching* ReLU from Phase 3b passes with mixed signs in one block, plus a new test with an if/else where both sides write different values.

---

## Phase 8 – Warp scheduling (latency hiding)

**Goal:** keep the ALUs busy while one group of threads waits on memory.

**What you need to know**
- Warps: a core holds several resident blocks (warps), each with its own PC, registers, NZP and state.
- The warp scheduler: each cycle, pick a warp that is *ready* (not in WAIT); round-robin vs oldest-first.
- Cost: register-file size multiplies with warp count, which is why real GPUs have huge register files. Occupancy.
- Dispatch changes: hand a core more than one block at a time.

**You build:** `WARPS_PER_CORE` parameter; per-warp state; a scheduler that switches warps when the current one is in WAIT; dispatch that fills all warp slots.

**Done when:** all tests pass; with memory latency from Phase 5 (and the cache off), cycles drop compared with `WARPS_PER_CORE=1`. Fill in the baseline table and explain the result.

---

## Progress

- [x] Phase 1 – Vector subtraction (hand-assembled) – 179 cycles, same as matadd
  - Found & fixed: partial last block hung forever (scheduler read `next_pc` from idle slot 3 → now slot 0)
  - Test has timeout, wraparound data, and a 0xAA sentinel check on idle outputs
- [ ] Phase 2 – Assembler + test helper
- [ ] Phase 3a – Dot product
- [ ] Phase 3b – ReLU (failing on purpose)
- [ ] Phase 4 – New instructions (`MAX`, ReLU fixed branch-free)
- [ ] Phase 5 – Memory latency + data cache
- [ ] Phase 6 – Pipelining (prefetch)
- [ ] Phase 7 – Branch divergence
- [ ] Phase 8 – Warp scheduling
