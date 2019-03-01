/*
 * Copyright 2019, Peter Han, All rights reserved.
 * This code is released into the public domain.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
// Copyright 2019, Peter Han, All rights reserved.
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
