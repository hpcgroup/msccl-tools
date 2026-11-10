# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The two fast paths of the language compiler must give the results of the code they shorten:
# - InstructionDAG._complete_metadata (it skips repeated visits) against a walk of every path;
# - the text writer of ir_to_xml (_ir_text) against the lxml writer, on the same pre-processed program.

from msccl.topologies import fully_connected
from msccl.language import *
from msccl.language.collectives import *
import msccl.language.ir as ir_module


def _ring_allreduce(n, instances=1, protocol='Simple', name='ring', fusion=False):
    prgm = MSCCLProgram(name, fully_connected(n), AllReduce(n, n, True), instances, protocol=protocol,
                        threadblock_policy=ThreadblockPolicy.manual, instr_fusion=fusion)
    with prgm:
        for step in range(0, n - 1):
            for index in range(0, n):
                rank = (index + step) % n
                next_rank = (index + step + 1) % n
                chunk(next_rank, Buffer.input, index).reduce(
                    chunk(rank, Buffer.input, index), ch=index % 2, recvtb=index % 2, sendtb=index % 2)
        for step in range(-1, n - 2):
            for index in range(0, n):
                rank = (index + step) % n
                next_rank = (index + step + 1) % n
                chunk(rank, Buffer.input, index).copy(
                    next_rank, Buffer.input, index, ch=index % 2, recvtb=index % 2, sendtb=index % 2)
        assert Check()
    return prgm


def _doubling_allreduce(n, instances=1):
    # Recursive doubling / halving: many paths of equal length between two ops.
    prgm = MSCCLProgram('doubling', fully_connected(n), AllReduce(n, n, True), instances, protocol='LL',
                        threadblock_policy=ThreadblockPolicy.manual, interleaved_replication=False,
                        instr_fusion=False)
    with prgm:
        count = n // 2
        while count >= 1:
            for rank in range(n):
                peer = rank ^ count
                index = (peer // count) * count
                c1 = chunk(rank, Buffer.input, index, size=count)
                chunk(peer, Buffer.output, index, size=count).reduce(c1, sendtb=peer, recvtb=rank, ch=0)
            count //= 2
        count = 1
        while count < n:
            for rank in range(n):
                peer = rank ^ count
                index = (rank // count) * count
                chunk(rank, Buffer.output, index, size=count).copy(
                    peer, Buffer.output, index, sendtb=peer, recvtb=rank, ch=0)
            count *= 2
        assert Check()
    return prgm


def _one_sided(name='one_sided'):
    # Rank 2 has no threadblock: its <gpu> element has no children.
    prgm = MSCCLProgram(name, fully_connected(3), AllToAll(3, 1, False), 1)
    with prgm:
        chunk(0, Buffer.input, 0).copy(1, Buffer.output, 0)
    return prgm


def _scratch():
    prgm = MSCCLProgram('scratch', fully_connected(3), AllReduce(3, 3, inplace=False), 2)
    with prgm:
        chunk(0, Buffer.input, 0).copy(2, 'scratch', 2)
        chunk(1, Buffer.input, 0).copy(2, 'scratch')
    return prgm


def _programs():
    return [_ring_allreduce(4), _ring_allreduce(8, instances=4, protocol='LL'), _ring_allreduce(6, fusion=True),
            _doubling_allreduce(8), _doubling_allreduce(16, instances=2), _one_sided(), _scratch()]


def _reachable(dag):
    ops, seen, stack = [], set(), list(dag.operations.values())
    while stack:
        op = stack.pop()
        if id(op) in seen:
            continue
        seen.add(id(op))
        ops.append(op)
        stack.extend(op.next)
        if op.recv_match is not None:
            stack.append(op.recv_match)
    return ops


def _every_path_metadata(dag):
    # _complete_metadata as it was before it skipped repeated visits.
    def dfs(op, cs):
        op.chunk_step = max(op.chunk_step, cs+1)
        if len(op.next) == 0 and op.recv_match is None:
            op.priority = 0
        else:
            for o in op.next:
                dfs(o, op.chunk_step)
            if len(op.next) > 0:
                highest_next_priority = max([x.priority+1 for x in op.next])
                op.priority = max(highest_next_priority, op.priority)
            if op.is_send():
                dfs(op.recv_match, op.chunk_step)
                op.priority = max(op.priority, op.recv_match.priority+1)

    for op in dag.operations.values():
        if op.inst == Instruction.start:
            dfs(op, -2)


def test_complete_metadata_equals_walk_of_every_path():
    for prgm in _programs():
        dag = prgm.instr_dag
        dag.convert_set_list()
        if prgm.instr_fusion:
            dag.optimize()
        ops = _reachable(dag)
        assert all(op.chunk_step == -1 and op.priority == -1 for op in ops)
        _every_path_metadata(dag)
        expected = [(op.chunk_step, op.priority) for op in ops]
        for op in ops:
            op.chunk_step = -1
            op.priority = -1
        dag._complete_metadata()
        assert [(op.chunk_step, op.priority) for op in ops] == expected
        # A second call changes nothing.
        dag._complete_metadata()
        assert [(op.chunk_step, op.priority) for op in ops] == expected


def _both_writers(prgm, lowered=None, **kwargs):
    # The text writer's answer, and the lxml writer's output for the same pre-processed program.
    if lowered is None:
        lowered = prgm.lower()
    kept = []
    real = ir_module._ir_text

    def keep_and_decline(*args):
        kept.append(real(*args))
        return None

    ir_module._ir_text = keep_and_decline
    try:
        xml = ir_module.ir_to_xml(lowered, dependence_nop=prgm.dependence_nop, **kwargs)
    finally:
        ir_module._ir_text = real
    assert len(kept) == 1
    return kept[0], xml


def test_ir_text_equals_lxml():
    for prgm in _programs():
        text, xml = _both_writers(prgm)
        assert text is not None
        assert text == xml


def test_ir_text_writes_childless_elements_like_lxml():
    text, xml = _both_writers(_one_sided())
    assert text == xml
    assert '<gpu id="2" i_chunks="0" o_chunks="0" s_chunks="0"/>' in xml


def test_ir_text_declines_what_lxml_would_change():
    # A name with a character that lxml escapes: the lxml writer is used.
    text, xml = _both_writers(_one_sided(name='a&b'))
    assert text is None
    assert 'name="a&amp;b"' in xml
    assert ir_to_xml(_one_sided(name='a&b').lower()) == xml
    # Other call forms: the lxml writer is used.
    for kwargs in ({'pretty_print': False}, {'old_format': False}):
        text, xml = _both_writers(_ring_allreduce(4), **kwargs)
        assert text is None
        assert xml.startswith('<algo ')
    # A value that is not an int: lxml writes str(True).
    prgm = _one_sided()
    lowered = prgm.lower()
    lowered.gpus[0].threadblocks[0].channel = True
    text, xml = _both_writers(prgm, lowered)
    assert text is None
    assert 'chan="True"' in xml


def test_generate_xml_uses_the_text_writer():
    calls = []
    real = ir_module._ir_text

    def spy(*args):
        calls.append(real(*args))
        return calls[-1]

    ir_module._ir_text = spy
    try:
        xml = _ring_allreduce(4).generate_xml()
    finally:
        ir_module._ir_text = real
    assert len(calls) == 1 and calls[0] == xml
