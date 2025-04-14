/* SystemUtils.cpp */

#include "SystemUtils.hpp"
#include "Common.hpp"

#include <iostream>
#include <filesystem>
#include <csignal>
#include <iomanip>
#include <thread>
#include <sys/statvfs.h>

// Check if available disk space is below threshold
bool is_disk_space_below_threshold(const char *path, double threshold)
{
    struct statvfs stat;
    if (statvfs(path, &stat) != 0)
    {
        std::cerr << "Error getting filesystem statistics.\n";
        return false;
    }
    double available_space = stat.f_bsize * stat.f_bavail;
    return available_space < threshold;
}

// Set CPU affinity for current process
void set_process_affinity(int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &cpuset) != 0)
    {
        std::cerr << "Failed to set process affinity to Core " << core_id << "\n";
    }
}

// Set scheduling priority for a thread
bool set_thread_priority(std::thread &th, int priority)
{
    struct sched_param param{.sched_priority = priority};
    if (pthread_setschedparam(th.native_handle(), SCHED_FIFO, &param) != 0)
    {
        std::cerr << "Failed to set thread priority to " << priority << "\n";
        return false;
    }
    return true;
}

// Set CPU affinity for a thread
bool set_thread_affinity(std::thread &th, int core_id)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    return pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset) == 0;
}

// SIGINT handler for graceful shutdown
void signal_handler(int sig)
{
    if (sig == SIGINT)
    {
        std::cout << "^C SIGINT received, initiating shutdown...\n";
        stop_program.store(true);
        stop_acquisition.store(true);

        if (pid1 > 0) kill(pid1, SIGINT);
        if (pid2 > 0) kill(pid2, SIGINT);

        std::cin.setstate(std::ios::failbit);

        for (Channel* ch : {&channel1, &channel2})
        {
            sem_post(&ch->data_sem_csv);
            sem_post(&ch->data_sem_dac);
            sem_post(&ch->model_sem);
            sem_post(&ch->result_sem_csv);
            sem_post(&ch->result_sem_dac);
        }
    }
}

// Print duration in human-readable format
void print_duration(const std::string &label, uint64_t start_ns, uint64_t end_ns)
{
    auto duration_ms = (end_ns > start_ns) ? (end_ns - start_ns) / 1'000'000 : 0;
    std::cout << std::left << std::setw(40) << label + " acquisition time:"
              << duration_ms / 60000 << " min "
              << (duration_ms % 60000) / 1000 << " sec "
              << duration_ms % 1000 << " ms\n";
}

// Display statistics for each channel
void print_channel_stats(const shared_counters_t *counters)
{
    std::cout << "\n====================================\n\n";

    print_duration("Channel 1", counters[0].trigger_time_ns.load(), counters[0].end_time_ns.load());
    print_duration("Channel 2", counters[1].trigger_time_ns.load(), counters[1].end_time_ns.load());

    for (int i = 0; i < 2; ++i)
    {
        const auto &c = counters[i];
        std::cout << std::left << std::setw(60)
                  << "Total data acquired CH" << i + 1 << ": " << c.acquire_count.load() << '\n';

        if (save_data_csv)
            std::cout << std::setw(60) << "Total lines written CH" << i + 1 << " to CSV: " << c.write_count_csv.load() << '\n';
        if (save_data_dac)
            std::cout << std::setw(60) << "Total lines written CH" << i + 1 << " to DAC: " << c.write_count_dac.load() << '\n';

        std::cout << std::setw(60) << "Total model calculated CH" << i + 1 << ": " << c.model_count.load() << '\n';

        if (save_output_csv)
            std::cout << std::setw(60) << "Total results logged CH" << i + 1 << " to CSV: " << c.log_count_csv.load() << '\n';
        if (save_output_dac)
            std::cout << std::setw(60) << "Total results written CH" << i + 1 << " to DAC: " << c.log_count_dac.load() << '\n';
    }

    std::cout << "\n====================================\n";
}

// Ensure folder exists and is empty
void folder_manager(const std::string &folder_path)
{
    namespace fs = std::filesystem;
    try
    {
        fs::path dir_path(folder_path);
        if (fs::exists(dir_path))
        {
            for (const auto &entry : fs::directory_iterator(dir_path))
            {
                try { fs::remove_all(entry); }
                catch (const fs::filesystem_error &e)
                {
                    std::cerr << "Failed to delete: " << entry.path() << " - " << e.what() << '\n';
                }
            }
        }
        else if (!fs::create_directories(dir_path))
        {
            std::cerr << "Failed to create directory: " << folder_path << '\n';
        }
    }
    catch (const fs::filesystem_error &e)
    {
        std::cerr << "Filesystem error: " << e.what() << '\n';
    }
}

// Ask user what data and output they want to save
bool ask_user_preferences(bool &save_data_csv, bool &save_data_dac,
                          bool &save_output_csv, bool &save_output_dac)
{
    int attempts = 3;

    for (int i = 0; i < attempts; ++i)
    {
        int choice;
        std::cout << "Save acquired data?\n"
                  << " 1. CSV only\n"
                  << " 2. DAC only\n"
                  << " 3. Both\n"
                  << " 4. None\n"
                  << "Enter choice (1-4): ";
        std::cin >> choice;

        if (choice >= 1 && choice <= 4)
        {
            save_data_csv = (choice == 1 || choice == 3);
            save_data_dac = (choice == 2 || choice == 3);
            break;
        }

        std::cerr << "Invalid input. Try again.\n";
        if (i == attempts - 1) return false;
    }

    for (int i = 0; i < attempts; ++i)
    {
        int choice;
        std::cout << "\nModel output?\n"
                  << " 1. CSV only\n"
                  << " 2. DAC only\n"
                  << " 3. Both\n"
                  << " 4. None\n"
                  << "Enter choice (1-4): ";
        std::cin >> choice;

        if (choice >= 1 && choice <= 4)
        {
            save_output_csv = (choice == 1 || choice == 3);
            if (save_data_dac && (choice == 2 || choice == 3))
            {
                save_output_dac = false;
                std::cerr << "[Warning] DAC already used for raw data. Model output will not use DAC.\n";
            }
            else
            {
                save_output_dac = (choice == 2 || choice == 3);
            }
            return true;
        }

        std::cerr << "Invalid input. Try again.\n";
        if (i == attempts - 1) return false;
    }

    return true;
}

// Barrier synchronization using atomic counter
void wait_for_barrier(std::atomic<int> &barrier, int total_participants)
{
    barrier.fetch_add(1);
    while (barrier.load() < total_participants)
        std::this_thread::yield();
}
