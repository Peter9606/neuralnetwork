#pragma once

#include <memory>
#include "spdlog/spdlog.h"

using spdlog::logger;

namespace nn {
/** @class Logger
 * get a singleton spdlog::logger instance
 */
class Logger {
 public:
    /*
     * Get a singleton spdlog::logger instance
     *
     * @return spdlog::logger shared pointer
     */
    static std::shared_ptr<logger> getLogger();

 private:
    Logger();
    ~Logger();

    Logger(const Logger&) = delete;
    Logger(Logger&&)      = delete;
    Logger& operator=(const Logger&) = delete;
    Logger& operator=(Logger&&) = delete;

 private:
    std::shared_ptr<logger> logger_ = nullptr;
};
}  // namespace nn
