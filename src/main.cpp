/* main.cpp */

#include <iostream>
#include <thread>
#include <atomic>
#include <csignal>
#include <chrono>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>
#include <iomanip>

#include "rp.h"
#include "Common.hpp"
#include "SystemUtils.hpp"
#include "DataAcquisition.hpp"
#include "DataWriterCSV.hpp"
#include "DataWriterDAC.hpp"
#include "ModelProcessing.hpp"
#include "ModelWriterCSV.hpp"
#include "ModelWriterDAC.hpp"
#include "DAC.hpp"

// Global process IDs and user preferences
pid_t pid1 = -1;
pid_t pid2 = -1;
bool save_data_csv = false;
bool save_data_dac = false;
bool save_output_csv = false;
bool save_output_dac = false;

int main()
{
    if (rp_Init() != RP_OK)
    {
        std::cerr << "Red Pitaya API initialization failed!" << std::endl;
        return -1;
    }

    // Initialize semaphores for both channels
    sem_init(&channel1.data_sem_csv, 0, 0);
    sem_init(&channel1.data_sem_dac, 0, 0);
    sem_init(&channel1.model_sem, 0, 0);
    sem_init(&channel1.result_sem_csv, 0, 0);
    sem_init(&channel1.result_sem_dac, 0, 0);

    sem_init(&channel2.data_sem_csv, 0, 0);
    sem_init(&channel2.data_sem_dac, 0, 0);
    sem_init(&channel2.model_sem, 0, 0);
    sem_init(&channel2.result_sem_csv, 0, 0);
    sem_init(&channel2.result_sem_dac, 0, 0);

    std::signal(SIGINT, signal_handler);

    // Prepare output folders
    folder_manager("DataOutput");
    folder_manager("ModelOutput");

    // Setup shared memory for counters
    int shm_fd_counters = shm_open(SHM_COUNTERS, O_CREAT | O_RDWR, 0666);
    if (shm_fd_counters == -1)
    {
        std::cerr << "Failed to create shared memory for counters!" << std::endl;
        return -1;
    }

    if (ftruncate(shm_fd_counters, sizeof(shared_counters_t) * 2) == -1)
    {
        std::cerr << "Failed to set size for shared memory!" << std::endl;
        return -1;
    }

    shared_counters_t *shared_counters = (shared_counters_t *)mmap(
        0, sizeof(shared_counters_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd_counters, 0);
    if (shared_counters == MAP_FAILED)
    {
        std::cerr << "Failed to map shared memory!" << std::endl;
        return -1;
    }

    // Initialize atomic counters
    for (int i = 0; i < 2; ++i)
    {
        new (&shared_counters[i].acquire_count) std::atomic<int>(0);
        new (&shared_counters[i].model_count) std::atomic<int>(0);
        new (&shared_counters[i].write_count_csv) std::atomic<int>(0);
        new (&shared_counters[i].write_count_dac) std::atomic<int>(0);
        new (&shared_counters[i].log_count_csv) std::atomic<int>(0);
        new (&shared_counters[i].log_count_dac) std::atomic<int>(0);
        new (&shared_counters[i].ready_barrier) std::atomic<int>(0);
    }

    std::cout << "Starting program..." << std::endl;

    // Ask user which outputs to save
    if (!ask_user_preferences(save_data_csv, save_data_dac, save_output_csv, save_output_dac))
    {
        std::cerr << "Invalid input. Exiting." << std::endl;
        return -1;
    }

    ::save_data_csv = save_data_csv;
    ::save_data_dac = save_data_dac;
    ::save_output_csv = save_output_csv;
    ::save_output_dac = save_output_dac;

    initialize_acq();
    initialize_DAC();

    // === Fork Child Process for Channel 1 ===
    if ((pid1 = fork()) == 0)
    {
        std::cout << "Child Process 1 (CH1) started. PID: " << getpid() << std::endl;

        int shm_fd = shm_open(SHM_COUNTERS, O_RDWR, 0666);
        shared_counters_t *counters = (shared_counters_t *)mmap(
            0, sizeof(shared_counters_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (counters == MAP_FAILED)
        {
            std::cerr << "CH1: Shared memory mapping failed!" << std::endl;
            exit(-1);
        }

        channel1.counters = &counters[0];
        set_process_affinity(0);
        wait_for_barrier(counters[0].ready_barrier, 2);

        std::thread acq_thread(acquire_data, std::ref(channel1), RP_CH_1);
        std::thread model_thread(model_inference, std::ref(channel1));

        std::thread write_thread_csv, write_thread_dac, log_thread_csv, log_thread_dac;
        if (save_data_csv)
            write_thread_csv = std::thread(write_data_csv, std::ref(channel1), "DataOutput/data_ch1.csv");
        if (save_data_dac)
            write_thread_dac = std::thread(write_data_dac, std::ref(channel1), RP_CH_1);
        if (save_output_csv)
            log_thread_csv = std::thread(log_results_csv, std::ref(channel1), "ModelOutput/output_ch1.csv");
        if (save_output_dac)
            log_thread_dac = std::thread(log_results_dac, std::ref(channel1), RP_CH_1);

        set_thread_priority(model_thread, model_priority);

        acq_thread.join();
        model_thread.join();
        if (write_thread_csv.joinable())
            write_thread_csv.join();
        if (write_thread_dac.joinable())
            write_thread_dac.join();
        if (log_thread_csv.joinable())
            log_thread_csv.join();
        if (log_thread_dac.joinable())
            log_thread_dac.join();

        std::cout << "Child Process 1 (CH1) finished." << std::endl;
        exit(0);
    }

    // === Fork Child Process for Channel 2 ===
    if ((pid2 = fork()) == 0)
    {
        std::cout << "Child Process 2 (CH2) started. PID: " << getpid() << std::endl;

        int shm_fd = shm_open(SHM_COUNTERS, O_RDWR, 0666);
        shared_counters_t *counters = (shared_counters_t *)mmap(
            0, sizeof(shared_counters_t) * 2, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
        if (counters == MAP_FAILED)
        {
            std::cerr << "CH2: Shared memory mapping failed!" << std::endl;
            exit(-1);
        }

        channel2.counters = &counters[1];
        set_process_affinity(1);
        wait_for_barrier(counters[0].ready_barrier, 2);

        std::thread acq_thread(acquire_data, std::ref(channel2), RP_CH_2);
        std::thread model_thread(model_inference, std::ref(channel2));

        std::thread write_thread_csv, write_thread_dac, log_thread_csv, log_thread_dac;
        if (save_data_csv)
            write_thread_csv = std::thread(write_data_csv, std::ref(channel2), "DataOutput/data_ch2.csv");
        if (save_data_dac)
            write_thread_dac = std::thread(write_data_dac, std::ref(channel2), RP_CH_2);
        if (save_output_csv)
            log_thread_csv = std::thread(log_results_csv, std::ref(channel2), "ModelOutput/output_ch2.csv");
        if (save_output_dac)
            log_thread_dac = std::thread(log_results_dac, std::ref(channel2), RP_CH_2);

        set_thread_priority(model_thread, model_priority);

        acq_thread.join();
        model_thread.join();
        if (write_thread_csv.joinable())
            write_thread_csv.join();
        if (write_thread_dac.joinable())
            write_thread_dac.join();
        if (log_thread_csv.joinable())
            log_thread_csv.join();
        if (log_thread_dac.joinable())
            log_thread_dac.join();

        std::cout << "Child Process 2 (CH2) finished." << std::endl;
        exit(0);
    }

    // === Parent Process Waits for Children ===
    int status;
    waitpid(pid1, &status, 0);
    waitpid(pid2, &status, 0);

    std::cout << "Both child processes finished.\n";

    cleanup();
    print_channel_stats(shared_counters);
    shm_unlink(SHM_COUNTERS);

    // Destroy semaphores
    for (auto &sem : {&channel1.data_sem_csv, &channel1.data_sem_dac, &channel1.model_sem, &channel1.result_sem_csv, &channel1.result_sem_dac,
                      &channel2.data_sem_csv, &channel2.data_sem_dac, &channel2.model_sem, &channel2.result_sem_csv, &channel2.result_sem_dac})
    {
        sem_destroy(sem);
    }

    return 0;
}
