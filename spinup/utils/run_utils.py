import os


def setup_logger_kwargs(exp_name, seed=None, data_dir=None):
    base = data_dir if data_dir is not None else os.path.join(os.getcwd(), "data")
    if seed is None:
        subdir = exp_name
    else:
        subdir = f"{exp_name}_s{seed}"
    output_dir = os.path.join(base, subdir)
    return dict(output_dir=output_dir, exp_name=exp_name)
