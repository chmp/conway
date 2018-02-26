#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>

#include <boost/align.hpp>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>
#include <boost/thread/latch.hpp>

#define ALIGNMENT 64
#define RESTRICT __restrict__

template<typename T>
using aligned_vector =  std::vector<T, boost::alignment::aligned_allocator<T, ALIGNMENT>>; 

template<typename T>
struct State {
    State(std::size_t width, std::size_t height) 
    : width(width),
    height(height),
    buffers{
        aligned_vector<T>((width + 2) * (height + 2), T()),
        aligned_vector<T>((width + 2) * (height + 2), T()),
    },
    front_(0)
    {}
    
    State(State&&) = default;

    void swap() {
        front_ = (front_ + 1) % 2;
    }

    aligned_vector<T>& front() {
        return buffers[front_];
    }

    aligned_vector<T>& back() {
        return buffers[(front_ + 1) % 2];
    }

    std::size_t width;
    std::size_t height;
    aligned_vector<T> buffers[2];
    int front_;
};

template<typename T>
inline auto _at(T* v, std::size_t full_width, size_t x, std::size_t y) -> T& {
    return v[full_width * y + x];
}

template<typename T>
inline auto at(T& v, std::size_t width, std::size_t height, std::size_t x, std::size_t y) 
    -> decltype(std::declval<T>()[x]) 
{
    return v[(width + 2) * y + x];
}

template<typename T>
State<T> create(std::size_t width, std::size_t height, double p) {
    auto result = State<T>{width, height};

    // TODO: add seed
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(p);
 
    for(std::size_t x = 0; x < width; ++x) {
        for(std::size_t y = 0; y < height; ++y) {
            at(result.front(), width, height, x + 1, y + 1) = dist(gen);
        }
    }

    return result;
}

template<typename T>
void step(
    std::size_t width, std::size_t height, const T* RESTRICT front, T* RESTRICT back,
    std::size_t y_offset = 0, std::size_t y_step = 1
) {
    const T* RESTRICT aligned_front = BOOST_ALIGN_ASSUME_ALIGNED(front, ALIGNMENT);
    T* RESTRICT aligned_back = BOOST_ALIGN_ASSUME_ALIGNED(back, ALIGNMENT);
    auto full_width = width + 2;

    for(std::size_t y = y_offset; y < height; y += y_step) {
        for(std::size_t x = 0; x < width; ++x) {
            auto self = _at(aligned_front, full_width, x + 1, y + 1);
            T neighbors = (
                _at(aligned_front, full_width, x + 0, y + 0) + 
                _at(aligned_front, full_width, x + 1, y + 0) +
                _at(aligned_front, full_width, x + 2, y + 0) + 
                _at(aligned_front, full_width, x + 0, y + 1) +
                _at(aligned_front, full_width, x + 2, y + 1) + 
                _at(aligned_front, full_width, x + 0, y + 2) +
                _at(aligned_front, full_width, x + 1, y + 2) + 
                _at(aligned_front, full_width, x + 2, y + 2)
            );

            _at(aligned_back, full_width, x + 1, y + 1) = 
                (self != 0) ? (neighbors == 2) || (neighbors == 3) : (neighbors == 3);
        }
    }
}

struct Coordinator {
    boost::latch setup_latch;
    boost::latch stop_latch;
    boost::barrier step_barrier;
    
    Coordinator(std::size_t num_threads) 
    : setup_latch(1), stop_latch(num_threads), step_barrier(num_threads) 
    {}

    void wait_for_start() { setup_latch.wait(); }
    void signal_end() { stop_latch.count_down(); }
    void wait_for_step() { step_barrier.wait(); }

    void run() {
        setup_latch.count_down();
        stop_latch.wait();
    }
};

using variables_map = boost::program_options::variables_map;

variables_map parse_args(int argc, char** argv) {
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("size", po::value<std::size_t>(), "size")
        ("steps", po::value<int>(), "number of steps to run")
        ("threads", po::value<int>(), "number of threads");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    

    if(vm.count("size") == 0) {
        throw std::runtime_error("required argument size not passed");
    }
    if(vm.count("steps") == 0) {
        throw std::runtime_error("required argument steps not passed");
    }
    if(vm.count("threads") == 0) {
        throw std::runtime_error("required argument threads not passed");
    }
    return vm;
}

template<typename F>
double execution_time(F f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

int main(int argc, char** argv) {
    variables_map vm = parse_args(argc, argv);

    std::size_t width = vm["size"].as<std::size_t>();
    std::size_t height = vm["size"].as<std::size_t>();
    std::size_t num_threads = vm["threads"].as<int>();;
    int num_steps = vm["steps"].as<int>();

    auto state = create<char>(width, height, 0.5);

    Coordinator coordinator(num_threads);
    std::vector<boost::thread> threads;

    for(int i = 0; i < num_threads; ++i) {
        threads.emplace_back([&state, &coordinator, i, num_threads, num_steps](){
            coordinator.wait_for_start();
            for(int iter = 0; iter < num_steps / 2; ++iter) {
                step(
                    state.width, state.height, state.buffers[0].data(), state.buffers[1].data(), 
                    i, num_threads
                );
                coordinator.wait_for_step();
                step(
                    state.width, state.height, state.buffers[1].data(), state.buffers[0].data(), 
                    i, num_threads
                );
                coordinator.wait_for_step();
            }
            coordinator.signal_end();
        });
    }

    auto runtime = execution_time([&coordinator]() {
        coordinator.run();
    });
    
    std::cout << 
        "{" << 
        "\"size\": " << state.width << ", " << 
        "\"height\": " << state.height << ", " <<
        "\"steps\": " << num_steps << ", " <<
        "\"runtime\": " << runtime <<
        "}" <<
        std::endl;
        
    std::for_each(std::begin(threads), std::end(threads), [](auto& t){ t.join(); });
    return 0;
}