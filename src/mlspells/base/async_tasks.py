import asyncio

def run_async(callable):
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(callable)

def prepare_async():
    if _is_notebook():
        try:
            import nest_asyncio
        except ImportError:
            raise ImportError(name="nest_asyncio")

        nest_asyncio.apply()


def _is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
    
        