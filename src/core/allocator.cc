#include "core/allocator.h"
#include <cstddef>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        // IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment
        size = this->getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================
        size_t addr = 0;
        if(free_blocks.empty()) {
            addr = used;
            used += size;
            peak = std::max(peak, used);
        } else {
            auto it = free_blocks.begin();
            size_t max_addr = 0;
            for(; it != free_blocks.end(); it++) {
                max_addr = std::max(max_addr, it->first);
                if(it->second >= size) {
                    addr = it->first;
                    if(it->second > size) {
                        free_blocks.insert(std::make_pair(addr + size, it->second - size));
                    }
                    free_blocks.erase(it);
                    break;
                }
            }
            if(it == free_blocks.end()) {
                addr = max_addr;
                used += size;
                peak = std::max(peak, used);
            }
        }
        return addr;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
        if(free_blocks.find(addr + size) != free_blocks.end()) {
            size += free_blocks[addr + size];
            free_blocks.erase(addr + size);
        }
        free_blocks.insert(std::make_pair(addr, size));
        used -= size;
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
