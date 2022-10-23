#ifndef HPC_HELPERS_H
#define HPC_HELPERS_H

#include <iostream>
#include <chrono>

//need to provide a SPEEDUP macro to this file
#define SPEEDUP(block_label, naive_label)                 \
    std::cout << "SPEEDUP ("<< #naive_label <<"/"<< #block_label <<") = " << delta##naive_label.count() / delta##block_label.count() << std::endl;
                 

#define TIMERSTART(label)                                                  \
    std::chrono::time_point<std::chrono::system_clock> a##label, b##label; \
    a##label = std::chrono::system_clock::now();

#define TIMERSTOP(label)                                                   \
    b##label = std::chrono::system_clock::now();                           \
    std::chrono::duration<double> delta##label = b##label-a##label;        \
    std::cout << "# elapsed time ("<< #label <<"): "                       \
              << delta##label.count()  << "s" << std::endl;


#endif
