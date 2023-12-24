# libtorch-model-trainer
Still in progress, the idea is to train torch models without access to pytorch ( numpy + libtorch only )

This is the pipeline:

- Dataloading strictly with numpy ( and opencv/pil if needed )
- Training schedule and loop in python side
- Pass training hyperperameters ( optimizer info, variable lr, frozen layers ) to libtorch through shared object library (DLL/SO) using ctypes
- Pass X & Y Variables ( numpy tensors ) through ctypes
- Variables are placed on CUDA from C++ side
- Loss and Acc are returned to python, and appended to log


