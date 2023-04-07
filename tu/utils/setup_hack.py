import importlib.util


def import_method_from_path(module_path, method_name=None):
    # module_path = '/path/to/module.py'
    module_name = 'my_module'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if method_name is None:
        return module
    return getattr(module, method_name)
