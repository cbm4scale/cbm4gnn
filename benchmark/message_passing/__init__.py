from benchmark.message_passing.message_passing_variants import (
    NativePytorchScatterAddMessagePassing,
    NativePytorchCOOSparseMatrixMessagePassing,
    NativePytorchCSRSparseMatrixMessagePassing,
    TorchScatterCOOScatterAddMessagePassing,
    TorchScatterGatherCOOSegmentCOO,
    TorchScatterGatherCSRSegmentCSR,
    MKLCSRSparseMatrixMessagePassing,
)

all = ["NativePytorchScatterAddMessagePassing",
       "NativePytorchCOOSparseMatrixMessagePassing",
       "NativePytorchCSRSparseMatrixMessagePassing",
       "TorchScatterCOOScatterAddMessagePassing",
       "TorchScatterGatherCOOSegmentCOO",
       "TorchScatterGatherCSRSegmentCSR",
       "MKLCSRSparseMatrixMessagePassing",
       ]
