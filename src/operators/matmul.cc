#include "operators/matmul.h"
#include "core/tensor.h"
#include <cstddef>

namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        Shape shape_a = inputs[0]->getDims();
        Shape shape_b = inputs[1]->getDims();
        if (transA) {
            std::swap(shape_a[shape_a.size() - 1], shape_a[shape_a.size() - 2]);
        }
        if (transB) {
            std::swap(shape_b[shape_b.size() - 1], shape_b[shape_b.size() - 2]);
        }
        Shape result;
        for(size_t i = 0; i < shape_a.size() - 2; i++) {
            result.push_back(shape_a[i]);
        }
        result.push_back(shape_a[shape_a.size() - 2]);
        result.push_back(shape_b[shape_b.size() - 1]);
        return {{result}};
    }

} // namespace infini