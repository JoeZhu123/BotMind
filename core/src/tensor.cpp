#include "backend.h"
#include <numeric>

namespace botmind {

size_t Tensor::GetByteSize() const {
    size_t element_size = 0;
    switch (dtype) {
        case DataType::FLOAT32: element_size = 4; break;
        case DataType::FLOAT16: element_size = 2; break;
        case DataType::INT8:    element_size = 1; break;
        case DataType::INT32:   element_size = 4; break;
    }
    
    size_t num_elements = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
    return num_elements * element_size;
}

} // namespace botmind

