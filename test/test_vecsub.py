import cocotb 
from cocotb.triggers import RisingEdge
from .helpers.setup import setup
from .helpers.memory import Memory
from .helpers.format import format_cycle
from .helpers.logger import logger

@cocotb.test()
async def test_vecsub(dut):
    #program memory
    program_memory = Memory(dut=dut, addr_bits=8, data_bits=16, channels=1, name="program")
    program = [
                0b0101000011011110, # MUL R0, %blockIdx, %blockDim
                0b0011000000001111, # ADD R0, R0, %threadIdx         ; i = blockIdx * blockDim + threadIdx
                0b1001000100000000, # CONST R1, #0                   ; baseA (matrix A base address)
                0b1001001000001000, # CONST R2, #8                   ; baseB (matrix B base address)
                0b1001001100010000, # CONST R3, #16                  ; baseC (matrix C base address)
                0b0011010000010000, # ADD R4, R1, R0                 ; addr(A[i]) = baseA + i
                0b0111010001000000, # LDR R4, R4                     ; load A[i] from global memory
                0b0011010100100000, # ADD R5, R2, R0                 ; addr(B[i]) = baseB + i
                0b0111010101010000, # LDR R5, R5                     ; load B[i] from global memory
                0b0100011001000101, # SUB R6, R4, R5                 ; C[i] = A[i] - B[i]
                0b0011011100110000, # ADD R7, R3, R0                 ; addr(C[i]) = baseC + i
                0b1000000001110110, # STR R7, R6                     ; store C[i] in global memory
                0b1111000000000000, # RET                             ; end of kernel

            ]
    # Data Memory
    data_memory = Memory(dut=dut, addr_bits=8, data_bits=8, channels=4, name="data")
    data = [
            1, 2, 3, 4, 5, 6, 7, 8, # Matrix A (1 x 8)
            2, 4, 6, 8, 10, 12, 14, 16,  # Matrix B (1 x 8)
            170, 170, 170, 170, 170, 170,170, 170 # Matrix C (1 x 8)
    ]
      

    # Device Control
    threads = 6

    await setup(
        dut=dut,
        program_memory=program_memory,
        program=program,
        data_memory=data_memory,
        data=data,
        threads=threads
    )
    
    data_memory.display(24)
    cycles = 0
    while dut.done.value != 1 :
        if cycles >= 1000:
            raise TimeoutError("Simulation timed out")
        data_memory.run()
        program_memory.run()
    
        await cocotb.triggers.ReadOnly()
        format_cycle(dut, cycles)
            
        await RisingEdge(dut.clk)
        cycles += 1
    
    logger.info(f"Completed in {cycles} cycles")
    data_memory.display(24)
    
    expected_results = [(a - b) % 256 for a, b in zip(data[0:threads], data[8:8+threads])] 
    for i, expected in enumerate(expected_results):
        result = data_memory.memory[i + 16]
        assert result == expected, f"Result mismatch at index {i}: expected {expected}, got {result}"
    for i in range(threads,8):
        result = data_memory.memory[i + 16]
        assert result == data[16+i], f"Result mismatch at index {i}: expected {data[16+i]}, got {result}"