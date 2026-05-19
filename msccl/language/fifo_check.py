"""Compile-time check for NCCL send-FIFO pressure.

NCCL maintains a per-(channel, peer, direction) ring-buffer between the GPU
kernel and the network proxy with NCCL_STEPS=8 slots. If a kernel posts more
than 8 slots on the same connection before the peer's matching primitive can
drain anything, both sides deadlock silently.

This pass walks the lowered MSCCL IR per (rank, TB) and tracks pendingSend
per (channel, sendPeer). Any recv-side opcode in the TB acts as a yield point
that zeros all pendingSend buckets (the symmetric peer reaches its matching
send and starts draining). If pendingSend > 8 after more than one primitive
contributed since the last drain, the algorithm is rejected.

Single-primitive overflow is OK -- the runtime stripes within a single primitive.
"""
from collections import defaultdict
from msccl.language.ir import Instruction, Buffer

NCCL_STEPS = 8

_SEND_SIDE = {
    Instruction.send,
    Instruction.recv_copy_send,
    Instruction.recv_reduce_send,
    Instruction.recv_reduce_copy_send,
}
_RECV_SIDE = {
    Instruction.recv,
    Instruction.recv_copy_send,
    Instruction.recv_reduce_send,
    Instruction.recv_reduce_copy,
    Instruction.recv_reduce_copy_send,
}
_LOCAL_SRC = {Instruction.send, Instruction.copy, Instruction.reduce}
_LOCAL_DST = {
    Instruction.recv, Instruction.recv_copy_send, Instruction.recv_reduce_send,
    Instruction.recv_reduce_copy, Instruction.copy, Instruction.reduce,
    Instruction.recv_reduce_copy_send,
}


def _nchunks_per_loop(program):
    sizes = defaultdict(int)
    for gpu in program.gpus:
        for tb in gpu.threadblocks:
            for op in tb.ops:
                if op.inst in _LOCAL_SRC and op.src is not None:
                    k = (gpu.rank, op.src.buffer)
                    sizes[k] = max(sizes[k], op.src.index + op.src.size)
                if op.inst in _LOCAL_DST and op.dst is not None:
                    k = (gpu.rank, op.dst.buffer)
                    sizes[k] = max(sizes[k], op.dst.index + op.dst.size)
    if not program.gpus:
        return 0
    return max(
        max(sizes[(g.rank, Buffer.input)], sizes[(g.rank, Buffer.output)])
        for g in program.gpus
    )


def check_fifo_pressure(program, msg_bytes, nccl_buffsize=4 * 1024 * 1024):
    """Return None on success, or a human-readable error string on failure.

    program: a lowered msccl.language.ir.Program (post-MSCCLProgram.lower()).
    msg_bytes: target message size in bytes.
    nccl_buffsize: NCCL net buffer size in bytes (default 4 MiB).
    """
    if msg_bytes is None or msg_bytes <= 0:
        return None
    if not program.gpus:
        return None

    protocol = (getattr(program, "protocol", "Simple") or "Simple")
    chunk_steps = 4 if protocol.lower() == "simple" else 1
    step_size = max(1, nccl_buffsize // NCCL_STEPS)
    network_chunk_bytes = max(1, step_size * chunk_steps)

    nchunks = _nchunks_per_loop(program)
    if nchunks <= 0:
        return None
    msccl_chunk_bytes = max(1, msg_bytes // nchunks)
    max_bundle = max(1, network_chunk_bytes // msccl_chunk_bytes)

    for gpu in program.gpus:
        for tb_idx, tb in enumerate(gpu.threadblocks):
            pending = defaultdict(int)         # (channel, sendPeer) -> slots
            run = defaultdict(list)            # (channel, sendPeer) -> [(op_idx, slots, inst)]

            for op_idx, op in enumerate(tb.ops):
                cnt = op.cnt()
                if cnt <= 0:
                    continue
                prims_calls = -(-cnt // max_bundle)
                fifo_slots = prims_calls * chunk_steps
                channel = op.channel if op.channel != -1 else tb.channel

                is_send = op.inst in _SEND_SIDE
                is_recv = op.inst in _RECV_SIDE

                # For fused ops, account for the send-side increment FIRST,
                # then let the recv-side drain reset all buckets. This is the
                # conservative ordering: a fused RECV_*_SEND that lands while
                # the bucket is already at >8 still fires.
                if is_send:
                    peer = op.send_peer()
                    if peer >= 0:
                        key = (channel, peer)
                        pending[key] += fifo_slots
                        run[key].append((op_idx, fifo_slots, str(op.inst)))
                        if pending[key] > NCCL_STEPS and len(run[key]) > 1:
                            return _format_violation(
                                gpu.rank, tb_idx, channel, peer,
                                run[key], pending[key],
                                protocol, msg_bytes, nchunks, chunk_steps,
                                max_bundle, msccl_chunk_bytes,
                            )

                if is_recv:
                    # Yield point: peer's matching send fires; assume drain.
                    pending.clear()
                    run.clear()

    return None


def _format_violation(rank, tb_idx, channel, peer, run, cum,
                      protocol, msg_bytes, nchunks, chunk_steps,
                      max_bundle, msccl_chunk_bytes):
    prims = ", ".join(f"#{i}({op}, slots={s})" for i, s, op in run)
    return (
        f"FIFO pressure check failed: rank={rank} tb={tb_idx}, "
        f"connection (channel={channel}, peer={peer}, dir=send) "
        f"accumulated {cum} slots > NCCL_STEPS={NCCL_STEPS} across "
        f"{len(run)} consecutive primitives without a draining recv: "
        f"[{prims}]. "
        f"Compile context: protocol={protocol}, msg_bytes={msg_bytes}, "
        f"nchunksperloop={nchunks}, msccl_chunk_bytes={msccl_chunk_bytes}, "
        f"chunk_steps={chunk_steps}, max_bundle={max_bundle}. "
        f"Hint: coalesce into a single primitive with chunk(..., size=N), "
        f"or split work across multiple TBs."
    )
