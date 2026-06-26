"""
Multi-node training infrastructure: MPI.jl-based gradient all-reduce for data
parallelism across the HPC/cloud cluster, since no Julia DDP/FSDP equivalent
exists. Pulled forward in the roadmap (originally Phase 6) because multi-node
compute is available from the start of the project. Not yet implemented.
"""
module Distributed

end # module
