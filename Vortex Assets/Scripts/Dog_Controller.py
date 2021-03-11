import Vortex

def on_simulation_start(extension):
    extension.IO = []

    for i in range(8):
        extension.IO.append([create_input(extension, 'input ' + str(i), Vortex.Types.Type_Double), create_output(extension, 'output ' + str(i), Vortex.Types.Type_Double)])


def pre_step(extension):

    for [input, output] in extension.IO:
        output.value = input.value * 3
        pass

def create_output(extension, name, o_type, default_value=None):
    """Create output field with optional default value, reset on every simulation run."""
    if extension.getOutput(name) is None:
        extension.addOutput(name, o_type)
    if default_value is not None:
        extension.getOutput(name).value = default_value
    return extension.getOutput(name)


def create_parameter(extension, name, p_type, default_value=None):
    """Create parameter field with optional default value set only when the field is created."""
    if extension.getParameter(name) is None:
        field = extension.addParameter(name, p_type)
        if default_value is not None:
            field.value = default_value
    return extension.getParameter(name)


def create_input(extension, name, i_type, default_value=None):
    """Create input field with optional default value set only when the field is created."""
    if extension.getInput(name) is None:
        field = extension.addInput(name, i_type)
        if default_value is not None:
            field.value = default_value
    return extension.getInput(name)