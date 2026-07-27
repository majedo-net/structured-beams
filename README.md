# Structured Beams 
We can use this repository to store code for the generation of structured beams and optimization of metalens.

## Example Code
The example here contains the code to optimize an axisymmetric metalens to generate a bessel beam from incident gaussian beam.
The FDTD solver used is MEEP and runs in a Docker container.

### Dependencies
Docker (or podman) is the only required dependency. On windows it is probably easiest to install (Docker desktop)[https://docs.docker.com/desktop/setup/install/windows-install/].

I've provided a Makefile as a simple way to make sure the frequently used commands are entered the same each time. In this case they are pretty simple so you could just type the commands from the Makefile into your shell, but in some cases when the arguments get longer this can be useful.
To install make on windows, I have typically used the (Chocolatey package manager)[https://chocolatey.org/install]. 
Then GNU Make can be installed with `choco install make`. 
On MacOS, the system installed make might work fine but the current version can also be installed using homebrew, then invoke `gmake` instead of `make`. 

### Running example
The first time you run the example you have to first build the container, and any subsequent runs you must rebuild if you change the contents of the Dockerfile. 
For example, you might want to check the number of MPI threads used and change to match what is available on your system. 
To build using the Makefile, execute `make meep-docker-build`. 

If you only change python files, there is no need to rebuild the container. 
To run the optimization using the command in the Makefile, execute `make meep-docker-run`.

The optimization will take a few hours to run depending on your system. 
The total number of iterations is determined by two variables in `ms3d_opt.py`, starting in line 266 as of writing, the number of iterations is `num_betas * update_factor`. 
To run a shorter optimization, reduce `update_factor`. 
