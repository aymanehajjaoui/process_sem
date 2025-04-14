/* DataWriterCSV.hpp */

#pragma once

#include "Common.hpp"

// Writes acquired data from the CSV queue to a file.
// This function blocks until all data has been written or the program is stopped.
void write_data_csv(Channel &channel, const std::string &filename);
