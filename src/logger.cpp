#include "logger.h"

#include <iostream>
#include <memory>
#include <vector>

#include "spdlog/async_logger.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

using spdlog::logger;
using spdlog::sinks::rotating_file_sink_mt;
using spdlog::sinks::sink;
using spdlog::sinks::stdout_color_sink_mt;
using std::make_shared;

namespace nn {
std::shared_ptr<logger> Logger::getLogger() {
    static Logger instance;
    return instance.logger_;
}

Logger::Logger() try {
#ifndef NDEBUG
    auto console_sink = make_shared<stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::trace);
#endif

    auto file_sink = make_shared<rotating_file_sink_mt>(
        "neuralnetwork", 1024 * 1024 * 10, 3);
#ifdef NDEBUG
    file_sink->set_level(spdlog::level::warn);
#else
    file_sink->set_level(spdlog::level::info);
#endif

    std::vector<spdlog::sink_ptr> sinks;
#ifndef NDEBUG
    sinks.push_back(console_sink);
#endif
    sinks.push_back(file_sink);

    logger_ = make_shared<logger>("", sinks.begin(), sinks.end());
} catch (const spdlog::spdlog_ex ex) {
    std::cout << "Log initialization failed: " << ex.what() << std::endl;
    throw ex;
}

Logger::~Logger() {
    spdlog::drop_all();
}
}  // namespace nn
