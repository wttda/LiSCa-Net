
def format_runtime(runtime):
    if isinstance(runtime, (list, tuple)) and len(runtime) == 2:
        return f'{runtime[0]:.1f}/{runtime[1]:.2f}'
    else:
        return f'{float(runtime):.1f}'

