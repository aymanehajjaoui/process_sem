/* SystemUtils.hpp */

#pragma once

#include <iostream>
#include <sys/statvfs.h>
#include <chrono>
#include <csignal>
#include <thread>
#include <sys/stat.h>
#include <dirent.h>

#include "Common.hpp"

// Checks if available disk space on the given path is below a threshold (in bytes)
bool is_disk_space_below_threshold(const char *path, double threshold);

// Sets process affinity to a specific CPU core
void set_process_affinity(int core_id);

// Sets thread scheduling priority using SCHED_FIFO
bool set_thread_priority(std::thread &th, int priority);

// Sets thread affinity to a specific CPU core
bool set_thread_affinity(std::thread &th, int core_id);

// Handles SIGINT (Ctrl+C) for graceful shutdown
void signal_handler(int sig);

// Prints duration in human-readable form using nanosecond timestamps
void print_duration(const std::string &label, uint64_t start_ns, uint64_t end_ns);

// Displays statistics for a given channel's shared counters
void print_channel_stats(const shared_counters_t *counters);

// Ensures the given folder exists and clears its content if it already exists
void folder_manager(const std::string &folder_path);

// Interactive prompt for user to configure saving behavior for data and model output
bool ask_user_preferences(bool &save_data_csv, bool &save_data_dac, bool &save_output_csv, bool &save_output_dac);

// Simple software barrier to synchronize multiple processes/threads
void wait_for_barrier(std::atomic<int>& barrier, int total_participants);
