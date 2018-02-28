#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>

#include <boost/align.hpp>
#include <boost/program_options.hpp>
#include <boost/thread.hpp>
#include <boost/thread/latch.hpp>

#define ALIGNMENT 64
#define RESTRICT __restrict__

template<typename F>
double execution_time(F f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}

template<typename F>
struct ScopeGuard {
    F f;
    ScopeGuard(F f) : f(std::move(f)) {}
    ~ScopeGuard() { f(); }
};

template<typename F>
ScopeGuard<F> make_scope_guard(F f) {
    return ScopeGuard<F>(std::move(f));
}

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
    row_stats{
        aligned_vector<T>(height + 2, 0),
        aligned_vector<T>(height + 2, 0),
    }
    {}

    State(State&&) = default;

    aligned_vector<T>& front(int cycle) { return buffers[cycle]; }
    aligned_vector<T>& back(int cycle) { return buffers[(cycle + 1) % 2]; }
    aligned_vector<T>& front_stats(int cycle) { return row_stats[cycle]; }
    aligned_vector<T>& back_stats(int cycle) { return row_stats[(cycle + 1) % 2]; }

    const aligned_vector<T>& front(int cycle) const { return buffers[cycle]; }
    const aligned_vector<T>& back(int cycle) const { return buffers[(cycle + 1) % 2]; }
    const aligned_vector<T>& front_stats(int cycle) const { return row_stats[cycle]; }
    const aligned_vector<T>& back_stats(int cycle) const { return row_stats[(cycle + 1) % 2]; }

    const std::size_t width;
    const std::size_t height;
    aligned_vector<T> buffers[2];

    aligned_vector<T> row_stats[2];
};

template<typename T>
inline T& _at(T* v, std::size_t full_width, size_t x, std::size_t y) {
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

    for(std::size_t y = 0; y < height; ++y) {
        T row_stats = 0;
        for(std::size_t x = 0; x < width; ++x) {
            T val = dist(gen);
            at(result.front(0), width, height, x + 1, y + 1) = val;
            row_stats |= val;
        }
        result.front_stats(0)[y + 1] = row_stats;
    }

    return result;
}

template<typename T>
State<T> load(std::size_t width, std::size_t height, std::string file) {
    auto result = State<T>{width, height};

    std::ifstream s(file);
    char c;

    for(std::size_t y = 0; y < height; ++y) {
        T row_stats = 0;
        for(std::size_t x = 0; x < width; ++x) {
            s.get(c);
            T val = c != '.';
            at(result.front(0), width, height, x + 1, y + 1) = val;
            row_stats |= val;
        }
        s.get(c);
        if(c != '\n') {
            throw std::logic_error("invalid file format");
        }
        result.front_stats(0)[y + 1] = row_stats;
    }

    return result;
}

template<typename T>
void save(const State<T>& state, std::string file, int cycle=0) {
    std::ofstream s(file);

    for(std::size_t y = 0; y < state.height; ++y) {
        for(std::size_t x = 0; x < state.width; ++x) {
            s.put((at(state.front(cycle), state.width, state.height, x + 1, y + 1) != 0) ? '*' : '.');
        }
        s.put('\n');
    }
}

template<typename T>
inline void step(
    std::size_t width, std::size_t height,
    const T* RESTRICT front, T* RESTRICT back, const T* RESTRICT front_stats, T* RESTRICT back_stats,
    std::size_t y_offset, std::size_t y_step
) {
    const T* RESTRICT aligned_front = BOOST_ALIGN_ASSUME_ALIGNED(front, ALIGNMENT);
    T* RESTRICT aligned_back = BOOST_ALIGN_ASSUME_ALIGNED(back, ALIGNMENT);
    const T* RESTRICT aligned_front_stats = BOOST_ALIGN_ASSUME_ALIGNED(front_stats, ALIGNMENT);
    T* RESTRICT aligned_back_stats = BOOST_ALIGN_ASSUME_ALIGNED(back_stats, ALIGNMENT);

    auto full_width = width + 2;

    for(std::size_t y = y_offset; y < height; y += y_step) {
        T active_row = aligned_front_stats[y] | aligned_front_stats[y + 1] | aligned_front_stats[y + 2];
        if(active_row == 0) {
            continue;
        }

        T row_stats = 0;
        for(std::size_t x = 0; x < width; ++x) {
            auto self = _at(aligned_front, full_width, x + 1, y + 1);
            T neighbors = (
                _at(aligned_front, full_width, x + 0, y + 0) +
                _at(aligned_front, full_width, x + 1, y + 0) +
                _at(aligned_front, full_width, x + 2, y + 0) +
                _at(aligned_front, full_width, x + 0, y + 1) +
                // skip self
                // _at(aligned_front, full_width, x + 1, y + 1) +
                _at(aligned_front, full_width, x + 2, y + 1) +
                _at(aligned_front, full_width, x + 0, y + 2) +
                _at(aligned_front, full_width, x + 1, y + 2) +
                _at(aligned_front, full_width, x + 2, y + 2)
            );

            T new_value = (self != 0) ? ((neighbors == 2) || (neighbors == 3)) : (neighbors == 3);
            _at(aligned_back, full_width, x + 1, y + 1) = new_value;
            row_stats |= new_value;
        }
        aligned_back_stats[y + 1] = row_stats;
    }
}

template<typename T>
inline void step(State<T>& state, int cycle, std::size_t y_offset, std::size_t y_step) {
    step(
        state.width, state.height,
        state.front(cycle).data(), state.back(cycle).data(),
        state.front_stats(cycle).data(), state.back_stats(cycle).data(),
        y_offset, y_step
    );
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

template<typename T>
double run_threaded(std::size_t num_threads, State<T>& state, std::size_t num_steps) {
    auto coordinator = Coordinator{num_threads};

    std::vector<boost::thread> threads;
    auto scope_guard = make_scope_guard([&threads]() {
        std::for_each(std::begin(threads), std::end(threads), [](auto& t){ t.join(); });
    });

    for(std::size_t i = 0; i < num_threads; ++i) {
        threads.emplace_back([&state, &coordinator, i, num_threads, num_steps](){
            coordinator.wait_for_start();
            for(std::size_t iter = 0; iter < num_steps / 2; ++iter) {
                step(state, 0, i, num_threads);
                coordinator.wait_for_step();
                step(state, 1, i, num_threads);
                coordinator.wait_for_step();
            }
            coordinator.signal_end();
        });
    }

    return execution_time([&coordinator]() { coordinator.run(); });
}

template<typename T>
double run_unthreaded(State<T>& state, std::size_t num_steps) {
    return execution_time([&state, num_steps]() {
        save(state, "/dev/stdout", 0);
        std::cout << std::endl;

        for(std::size_t iter = 0; iter < num_steps / 2; ++iter) {
            step(state, 0, 0, 1);

            save(state, "/dev/stdout", 1);
            std::cout << std::endl;

            step(state, 1, 0, 1);

            save(state, "/dev/stdout", 0);
            std::cout << std::endl;
        }
     });
}

using variables_map = boost::program_options::variables_map;

variables_map parse_args(int argc, char** argv) {
    namespace po = boost::program_options;

    po::options_description desc("Allowed options");
    desc.add_options()
        ("size", po::value<std::size_t>(), "size")
        ("steps", po::value<std::size_t>(), "number of steps to run")
        ("threads", po::value<std::size_t>(), "number of threads")
        ("load", po::value<std::string>(), "file to load")
        ("save", po::value<std::string>(), "file to save");

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

int main(int argc, char** argv) {
    // TODO: allow to simulate single step to test correctness
    variables_map vm = parse_args(argc, argv);

    auto num_threads = vm["threads"].as<std::size_t>();
    auto num_steps = vm["steps"].as<std::size_t>();
    auto state = (vm.count("load") == 0) ?
        create<char>(vm["size"].as<std::size_t>(), vm["size"].as<std::size_t>(), 0.25) :
        load<char>(vm["size"].as<std::size_t>(), vm["size"].as<std::size_t>(), vm["load"].as<std::string>());

    auto runtime = (num_threads == 0) ?
        run_unthreaded(state, num_steps) :
        run_threaded(num_threads, state, num_steps);

    std::cout
        << "{" <<
        "\"size\": " << state.width << ", " <<  "\"height\": " << state.height << ", " <<
        "\"steps\": " << num_steps << ", " << "\"runtime\": " << runtime <<
        "}" << std::endl;

    if(vm.count("save")) {
        save(state, vm["save"].as<std::string>());
    }

    return 0;
}
