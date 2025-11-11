# Wang-Landau algorithm

This repository contains an implementation of the Wang-Landau algorithm for estimating the density of states in statistical mechanics systems.

# Usage

The user is required to implement a Model class that defines the system to be studied. The Model class should provide methods for initializing the system, proposing moves, and calculating energy states.

The Model class should implement the following interface:

```c++
class Model {
public:
    // Initialize the model with given parameters
    Model(/* parameters */);

    // Propose a move in the system
    void proposeMove();

    // Calculate the energy of the current state
    double calculateEnergy() const;

    // Other necessary methods...
};
```

The output of the Wang-Landau algorithm will be the estimated logarithm density of states, which can be used to compute thermodynamic properties of the system.

By default, the result will output to the standard output:

```bash 
./wang_landau > output.txt
```

where the output.txt file will contain the estimated density of states:

```csv
Energy, log(Density of States)
E1, log_g(E1)
E2, log_g(E2)
...
```

# Requirements

This project requires Eigen3 library.

# Building

To build the project, run the [build.py](build.py) script:

```bash
python3 build.py
```

Adjust the script as necessary to fit your environment.

# License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.