#ifndef UTIL_H__
#define UTIL_H__

#include <mutex>
#include <condition_variable>
#include <cassert>

#ifdef __clang__
#define SHRDR_PACKED __attribute__((packed))
#else
#define SHRDR_PACKED
#endif

#ifdef _MSC_VER
#define SHRDR_ALWAYS_INLINE __forceinline
#else
#define SHRDR_ALWAYS_INLINE __attribute__((always_inline))
#endif

#ifdef _MSC_VER
#define SHRDR_NO_INLINE __declspec(noinline)
#else
#define SHRDR_NO_INLINE __attribute__((noinline))
#endif

namespace shrdr {

class Barrier {
    // Bare bones thread barrier implemenation adapted from boost::barrier
public:
    Barrier(unsigned int count) :
        mutex(),
        cond(),
        original_count(count),
        count(count),
        generation(0)
    {
        assert(count > 0);
    }

    bool wait()
    {
        std::unique_lock<std::mutex> lock(mutex);
        unsigned int gen = generation;

        if (--count == 0) {
            generation++;
            count = original_count;
            lock.unlock();
            cond.notify_all();
            return true;
        }

        while (gen == generation) {
            cond.wait(lock);
        }
        return false;
    }

private:
    std::mutex mutex;
    std::condition_variable cond;
    unsigned int original_count;
    unsigned int count;
    unsigned int generation;
};

} // namespace shrdr

#endif // UTIL_H__