FROM kovaleskilab/meep:v4_michael

WORKDIR /ms-opt

RUN pip install SciencePlots autograd nlopt

ENTRYPOINT ["mpirun","-np","8","python"]
CMD ["./ms3d_opt.py"]