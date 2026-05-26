import inspect
import re
import textwrap
import types
import triton


def cacheable(f):
    """
    A decorator that allow you to write something of the form:

    @cacheable
    def my_kernel(): return (expression dynamically defining a kernel)

    such that it interacts gracefully with triton cache and preload.
    """

    g = f()
    g.fn.__name__ = f.__name__
    g.fn.__module__ = f.__module__
    g.fn.__qualname__ = f.__qualname__
    g.__name__ = f.__name__
    g.__module__ = f.__module__
    g.__qualname__ = f.__qualname__
    g._fn_name = f"{f.__module__}.{f.__qualname__}"
    return g


def define_kernel(src, module, attrs=None, **extra_globals):
    """
    Dynamically create a Triton function or kernel from a src string,
    linking any symbols in the kernel to objects specified by extra_globals.
    """

    # create templace function
    def _empty_fn():
        pass

    gdict = dict(**(_empty_fn.__globals__))
    gdict.update(extra_globals)
    f = types.FunctionType(_empty_fn.__code__, gdict)
    f.__module__ = module.__name__

    src = textwrap.dedent(src)
    src = src[src.find("def "):]

    stored_functions = []
    function_name = src[4:].split("(")[0].strip()

    exec_globals = gdict
    exec_globals.update({"stored_functions": stored_functions})
    exec(src + "\n\nstored_functions.append(" + function_name + ")\n", exec_globals)

    f.__signature__ = inspect.signature(stored_functions[0])
    f.__name__ = function_name
    f.__doc__ = stored_functions[0].__doc__

    if attrs is None:
        attrs = dict()
    f = triton.JITFunction(f, **attrs)
    f._unsafe_update_src(src)
    return f


def specialize(fn, module, constants, tuples, name=None, do_not_specialize=tuple()):
    assert isinstance(fn, triton.runtime.jit.JITFunction)
    if name is None:
        name = f"{fn.__name__}"
    # Get original source code
    src = inspect.getsource(fn.fn)
    src = textwrap.dedent(src)
    lines = src.split("\n")
    # Skip decorator and def line
    def_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("def"))
    # separate header vs body LOC
    header_end = def_idx
    while not lines[header_end].rstrip().endswith(":"):
        header_end += 1
    body_lines = lines[header_end + 1:]
    header_lines = lines[def_idx:header_end + 1]
    # clean-up header
    header_clean = [
        l.split("#", 1)[0].strip()  # keep code, discard comment
        for l in header_lines
        if l.split("#", 1)[0].strip()  # skip blank‑after‑comment lines
    ]
    # decompose arguments
    header_src = " ".join(header_clean)  # turn it into a single line
    m = re.search(r"\((.*)\)\s*:", header_src)
    if not m:
        raise ValueError("Could not parse function header")
    args_str = m.group(1)
    args = [arg.strip() for arg in args_str.split(",") if arg.strip()]
    non_specialized_args = []
    for arg in args:
        arg_key = arg.split(":")[0].split("=")[0].strip()
        new_args = tuples.get(arg_key, [arg])
        if arg_key not in constants:
            non_specialized_args += new_args
    # add global symbols
    spec_fns = {v.__name__: v for k, v in constants.items() if isinstance(v, triton.runtime.jit.JITFunction)}
    globals = spec_fns | fn.get_capture_scope()
    # build new source code and define kernel dynamically
    new_signature = f"def {name}({', '.join(non_specialized_args)}):"
    constexpr_lines = [
        f"    {key}: tl.constexpr = {value.__name__ if callable(value) else value}" for key, value in constants.items()
    ]
    tuple_lines = [
        f"    {key} = {'(' + ','.join(value) + (',' if len(value)>=1 else '') + ')'}" for key, value in tuples.items()
    ]
    new_src = "\n".join(["@triton.jit", new_signature] + constexpr_lines + tuple_lines + body_lines)
    # find function parameters
    sig = inspect.signature(triton.runtime.jit.JITFunction.__init__)
    params = list(sig.parameters.values())[2:]
    attrs = {param.name: getattr(fn, param.name, param.default) for param in params}

    # make a new repr which appends the repr of the specialized functions.
    base_repr = attrs["repr"]

    def new_repr(specialization):
        ret = base_repr(specialization)
        for spec_fn in spec_fns.values():
            spec_repr = spec_fn.repr(None)
            if spec_repr:
                spec_repr = spec_repr.strip("_")
            if spec_repr:
                ret += f"_{spec_repr}"
        return ret

    attrs["repr"] = new_repr

    if do_not_specialize:
        attrs["do_not_specialize"] = do_not_specialize
    ret = define_kernel(new_src, module, attrs, **globals)
    return ret
