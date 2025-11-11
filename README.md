# Wang-Landau algorithm

This repository contains an implementation of the Wang-Landau algorithm for estimating the density of states in statistical mechanics systems.

# Requirements

- C++20 compiler 
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) library.

# Building

To build the project, run the [build.py](build.py) script:

```bash
python3 build.py
```

Adjust the script as necessary to fit your environment.

# Usage

The user is required to implement a Model class that defines the system to be studied. The Model class should provide methods for initializing the system, proposing moves, and calculating energy states.

The Model class should implement the interface defined in [src/model.h](src/model.h). An example implementation for the Ising model can be found in [src/examples/ising_model.h](src/examples/ising_model.h).

The output of the Wang-Landau algorithm will be the logarithm density of states, which can be used to compute thermodynamic properties of the system (see [examples/measure.h](src/examples/measure.h) and [src/examples/example_measure.cpp](src/examples/example_measure.cpp) for reference).

By default, the result will output to the standard output:

```bash 
./wang_landau > output.txt
```

where the output.txt file will contain the logarithm density of states:

```csv
E1  log_g(E1)
E2  log_g(E2)
...
```

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.