#include <torch/extension.h>
#include "mkl_spblas.h"
#include "mkl.h"

void check_alloc(void* ptr) {
    if (ptr == NULL) {
            std::cerr << "Memory allocation failed." << std::endl;
            exit(-1);
            // Handle allocation failure
        } else {
            // Allocation was successful, use mkl_array as needed
            std::cout << "Memory allocation successful." << std::endl;
    }
}

void check_status(sparse_status_t s) {
    if (s == SPARSE_STATUS_SUCCESS) {
        return;
    }
    else if (s == SPARSE_STATUS_NOT_INITIALIZED) {
        printf("MKL: Not Initialized\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_ALLOC_FAILED) {
        printf("MKL: Not Alloc'ed\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_INVALID_VALUE) {
        printf("MKL: Invalid Value\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_EXECUTION_FAILED) {
        printf("MKL: Execution Failed\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_INTERNAL_ERROR) {
        printf("MKL: Internal Error\n");
        exit(-1);
    }
    else if (s == SPARSE_STATUS_NOT_SUPPORTED) {
        printf("MKL: Not Supported\n");
        exit(-1);
    }
}

void csr_s_spmm_(const at::Tensor& lhs_ptr, 
                 const at::Tensor& lhs_val, 
                 const at::Tensor& rhs, 
                 at::Tensor& dst) {
    MKL_INT *cbm_row = lhs_ptr[0].data_ptr<MKL_INT>();
    MKL_INT *cbm_col = lhs_ptr[1].data_ptr<MKL_INT>();
    float *cbm_val = lhs_val.data_ptr<float>();

    MKL_INT lda = dst.size(0);
    MKL_INT ldb = rhs.size(1);
    MKL_INT ldc = rhs.size(1);
    MKL_INT nnz = lhs_val.size(0);

    sparse_status_t s_coo;
    sparse_matrix_t coo;
    
    s_coo = mkl_sparse_s_create_coo(&coo,                                 // mkl matrix reference
                                    SPARSE_INDEX_BASE_ZERO,                 // index style (c-style)
                                    lda,                             // number of rows
                                    ldb,                             // number of columns
                                    nnz,                                // number of nonzeros
                                    cbm_row,  // pointer to row_coo
                                    cbm_col,  // pointer to col_coo
                                    cbm_val);             // pointer to val_coo
    
    check_status(s_coo);

    sparse_status_t s_csr;
    sparse_matrix_t csr;

    s_csr = mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr);
    
    check_status(s_csr);

    sparse_status_t s_spmm;

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    s_spmm = mkl_sparse_s_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                             1.0f, 
                             csr,
                             descr,
                             SPARSE_LAYOUT_ROW_MAJOR, 
                             rhs.data_ptr<float>(), 
                             ldb,
                             ldb, 
                             0.0f, 
                             dst.data_ptr<float>(),
                             ldc);

    check_status(s_spmm);

}

void csr_d_spmm_(const at::Tensor& lhs_ptr, 
                 const at::Tensor& lhs_val, 
                 const at::Tensor& rhs, 
                 at::Tensor& dst) {
    MKL_INT *cbm_row = lhs_ptr[0].data_ptr<MKL_INT>();
    MKL_INT *cbm_col = lhs_ptr[1].data_ptr<MKL_INT>();
    double *cbm_val = lhs_val.data_ptr<double>();

    MKL_INT lda = dst.size(0);
    MKL_INT ldb = rhs.size(1);
    MKL_INT ldc = rhs.size(1);
    MKL_INT nnz = lhs_val.size(0);

    sparse_status_t s_coo;
    sparse_matrix_t coo;
    
    s_coo = mkl_sparse_d_create_coo(&coo,                                 // mkl matrix reference
                                    SPARSE_INDEX_BASE_ZERO,                 // index style (c-style)
                                    lda,                             // number of rows
                                    ldb,                             // number of columns
                                    nnz,                                // number of nonzeros
                                    cbm_row,  // pointer to row_coo
                                    cbm_col,  // pointer to col_coo
                                    cbm_val);             // pointer to val_coo
    
    check_status(s_coo);

    sparse_status_t s_csr;
    sparse_matrix_t csr;

    s_csr = mkl_sparse_convert_csr(coo, SPARSE_OPERATION_NON_TRANSPOSE, &csr);
    
    check_status(s_csr);

    sparse_status_t s_spmm;

    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    s_spmm = mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE,
                             1.0f, 
                             csr,
                             descr,
                             SPARSE_LAYOUT_ROW_MAJOR, 
                             rhs.data_ptr<double>(), 
                             ldb,
                             ldb, 
                             0.0f, 
                             dst.data_ptr<double>(),
                             ldc);

    check_status(s_spmm);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("s_csr_spmm_", &csr_s_spmm_);
    m.def("d_csr_spmm_", &csr_d_spmm_);
}