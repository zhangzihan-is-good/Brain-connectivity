"""Version4 node-only strict binary64 summation compiled directly with LLVM.

No NumBa import, fast math, floating reassociation, FMA, magnitude threshold,
clipping, amplitude rescaling, or estimator-specific branch is used.  The
finite-input algorithm is the error-free partial expansion and final half-even
correction used by math.fsum.  The JIT engine and ctypes function remain alive
for the lifetime of this module.  CFUNCTYPE releases the GIL during each call.
"""
import ctypes
import math
import time
import numpy as np
try:
    import llvmlite
    from llvmlite import binding as llvm, ir
except ImportError:
    llvmlite = None

I, J = np.triu_indices(109, 1)
EDGE = np.array([np.flatnonzero((I == node) | (J == node)) for node in range(109)])
SOURCE = I[EDGE] == np.arange(109)[:, None]


def _compile():
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    target = llvm.Target.from_default_triple()
    machine = target.create_target_machine(opt=3)
    module = ir.Module(name='strict_binary64_fsum')
    module.triple = llvm.get_default_triple()
    module.data_layout = str(machine.target_data)
    d, i64, void = ir.DoubleType(), ir.IntType(64), ir.VoidType()
    zero, one = ir.Constant(i64, 0), ir.Constant(i64, 1)
    fzero = ir.Constant(d, 0.0)
    fn = ir.Function(module, ir.FunctionType(void, [d.as_pointer(), d.as_pointer(), i64, i64]), name='finite_fsum_rows')
    data, output, rows, cols = fn.args
    for arg, name in zip(fn.args, ['data', 'output', 'rows', 'cols']):
        arg.name = name
    names = ['entry', 'row_cond', 'row_start', 'col_cond', 'col_start', 'partial_start', 'partial_cond',
             'partial_body', 'keep_low', 'partial_next', 'partial_done', 'store_term',
             'col_next', 'col_done', 'final_start', 'empty_row', 'final_cond', 'final_body', 'correction_cond', 'correction_sign',
             'correction_add', 'correction_store', 'row_store', 'storage_error', 'exit']
    blocks = {name: fn.append_basic_block(name) for name in names}
    b = ir.IRBuilder(blocks['entry'])
    # This node-only API accepts at most 216 terms; each term adds at most one partial.
    partial = b.alloca(ir.ArrayType(d, 216), name='partial')
    iv = {name: b.alloca(i64, name=name) for name in ['row', 'col', 'n', 'j', 'write']}
    fv = {name: b.alloca(d, name=name) for name in ['x', 'hi', 'lo']}
    fabs = module.declare_intrinsic('llvm.fabs', [d])

    def load(name):
        return b.load(iv[name] if name in iv else fv[name])

    def save(name, value):
        b.store(value, iv[name] if name in iv else fv[name])

    def ptr(index):
        return b.gep(partial, [zero, index], inbounds=True)

    def goto(name):
        b.branch(blocks[name])

    def at(name):
        b.position_at_end(blocks[name])

    save('row', zero)
    goto('row_cond')
    at('row_cond')
    b.cbranch(b.icmp_signed('<', load('row'), rows), blocks['row_start'], blocks['exit'])
    at('row_start')
    save('n', zero)
    save('col', zero)
    goto('col_cond')
    at('col_cond')
    b.cbranch(b.icmp_signed('<', load('col'), cols), blocks['col_start'], blocks['col_done'])
    at('col_start')
    index = b.add(b.mul(load('row'), cols), load('col'))
    save('x', b.load(b.gep(data, [index], inbounds=True)))
    # An exactly zero term contributes no partial, uniformly for all methods.
    b.cbranch(b.fcmp_ordered('==', load('x'), fzero), blocks['col_next'], blocks['partial_start'])
    at('partial_start')
    save('j', zero)
    save('write', zero)
    goto('partial_cond')
    at('partial_cond')
    b.cbranch(b.icmp_signed('<', load('j'), load('n')), blocks['partial_body'], blocks['partial_done'])
    at('partial_body')
    x, y = load('x'), b.load(ptr(load('j')))
    swap = b.fcmp_ordered('<', b.call(fabs, [x]), b.call(fabs, [y]))
    xx, yy = b.select(swap, y, x), b.select(swap, x, y)
    hi = b.fadd(xx, yy)
    lo = b.fsub(yy, b.fsub(hi, xx))
    save('x', hi)
    save('lo', lo)
    b.cbranch(b.fcmp_ordered('!=', lo, fzero), blocks['keep_low'], blocks['partial_next'])
    at('keep_low')
    b.store(load('lo'), ptr(load('write')))
    save('write', b.add(load('write'), one))
    goto('partial_next')
    at('partial_next')
    save('j', b.add(load('j'), one))
    goto('partial_cond')
    at('partial_done')
    b.cbranch(b.icmp_signed('<', load('write'), ir.Constant(i64, 216)), blocks['store_term'], blocks['storage_error'])
    at('store_term')
    b.store(load('x'), ptr(load('write')))
    save('n', b.add(load('write'), one))
    goto('col_next')
    at('col_next')
    save('col', b.add(load('col'), one))
    goto('col_cond')
    at('col_done')
    b.cbranch(b.icmp_signed('>', load('n'), zero), blocks['final_start'], blocks['empty_row'])
    at('empty_row')
    save('hi', fzero)
    goto('row_store')
    at('final_start')
    save('n', b.sub(load('n'), one))
    save('hi', b.load(ptr(load('n'))))
    save('lo', fzero)
    goto('final_cond')
    at('final_cond')
    b.cbranch(b.icmp_signed('>', load('n'), zero), blocks['final_body'], blocks['correction_cond'])
    at('final_body')
    save('n', b.sub(load('n'), one))
    y, x = b.load(ptr(load('n'))), load('hi')
    hi = b.fadd(x, y)
    lo = b.fsub(y, b.fsub(hi, x))
    save('hi', hi)
    save('lo', lo)
    b.cbranch(b.fcmp_ordered('!=', lo, fzero), blocks['correction_cond'], blocks['final_cond'])
    at('correction_cond')
    b.cbranch(b.icmp_signed('>', load('n'), zero), blocks['correction_sign'], blocks['row_store'])
    at('correction_sign')
    lo, tail = load('lo'), b.load(ptr(b.sub(load('n'), one)))
    positive = b.and_(b.fcmp_ordered('>', lo, fzero), b.fcmp_ordered('>', tail, fzero))
    negative = b.and_(b.fcmp_ordered('<', lo, fzero), b.fcmp_ordered('<', tail, fzero))
    b.cbranch(b.or_(positive, negative), blocks['correction_add'], blocks['row_store'])
    at('correction_add')
    doubled = b.fmul(load('lo'), ir.Constant(d, 2.0))
    corrected = b.fadd(load('hi'), doubled)
    save('x', corrected)
    b.cbranch(b.fcmp_ordered('==', b.fsub(corrected, load('hi')), doubled), blocks['correction_store'], blocks['row_store'])
    at('correction_store')
    save('hi', load('x'))
    goto('row_store')
    at('storage_error')
    save('hi', ir.Constant(d, float('nan')))
    goto('row_store')
    at('row_store')
    hi = load('hi')
    # math.fsum returns positive zero for exact cancellation, including -0.
    hi = b.select(b.fcmp_ordered('==', hi, fzero), fzero, hi)
    b.store(hi, b.gep(output, [load('row')], inbounds=True))
    save('row', b.add(load('row'), one))
    goto('row_cond')
    at('exit')
    b.ret_void()

    parsed = llvm.parse_assembly(str(module))
    parsed.verify()
    pm = llvm.ModulePassManager()
    pmb = llvm.PassManagerBuilder()
    pmb.opt_level = 3
    pmb.loop_vectorize = False
    pmb.slp_vectorize = False
    pmb.populate(pm)
    pm.run(parsed)
    parsed.verify()
    optimized_ir = str(parsed)
    # Plain fadd/fsub/fmul have no reassociation or contraction permission.
    for flag in [' fast ', ' reassoc ', ' contract ', ' afn ', ' arcp ', ' nnan ', ' ninf ', ' nsz ']:
        if flag in optimized_ir:
            raise RuntimeError('Unexpected floating optimization flag: ' + flag)
    assembly = machine.emit_assembly(parsed)
    if any(instruction in assembly.lower() for instruction in ['vfmadd', 'vfmsub', 'vfnmadd', 'vfnmsub']):
        raise RuntimeError('Unexpected fused multiply-add instruction')
    engine = llvm.create_mcjit_compiler(llvm.parse_assembly(''), machine)
    engine.add_module(parsed)
    engine.finalize_object()
    engine.run_static_constructors()
    address = engine.get_function_address('finite_fsum_rows')
    signature = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int64, ctypes.c_int64)
    return engine, parsed, signature(address), optimized_ir, assembly


if llvmlite is not None:
    _ENGINE, _LLVM_MODULE, _NATIVE_FUNCTION, LLVM_IR, LLVM_ASSEMBLY = _compile()
else:
    _NATIVE_FUNCTION = None


def fsum_rows(a):
    a = np.asarray(a, np.float64)
    if a.ndim < 1:
        raise ValueError('fsum_rows requires a last summation axis')
    shape, cols = a.shape[:-1], a.shape[-1]
    if not np.isfinite(a).all():
        raise ValueError('fsum_rows supports finite inputs only')
    if cols > 216:
        raise ValueError('Node-only fsum_rows supports at most 216 last-axis terms')
    if cols == 0:
        return np.zeros(shape, dtype=np.float64)
    a = np.ascontiguousarray(a.reshape(-1, cols))
    if _NATIVE_FUNCTION is None:
        return np.array([math.fsum(row) for row in a],dtype=float).reshape(shape)
    output = np.empty(a.shape[0], dtype=np.float64)
    pointer = ctypes.POINTER(ctypes.c_double)
    _NATIVE_FUNCTION(a.ctypes.data_as(pointer), output.ctypes.data_as(pointer), len(a), cols)
    if not np.isfinite(output).all():
        raise OverflowError('fsum intermediate overflow or partial-expansion storage exhausted')
    return output.reshape(shape)

