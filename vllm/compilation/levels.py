# constants for the levels of the compilation process


class CompilationLevel:
    NO_COMPILATION = 0
    DYNAMO_AS_IS = 1
    DYNAMO_ONCE = 2
    INDUCTOR = 3
    INDUCTOR_MAX_AUTOTUNE = 4
