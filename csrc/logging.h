#include <fstream>
#include <cstdarg>
#include <cstdio>

void logg_511(const char* format, ...) {
    std::ofstream logFile("debug.log", std::ios_base::app);
    if (!logFile.is_open()) {
        // Give a warning if we can't open the file
        printf("Warning: could not open debug.log for writing\n");
        return;
    }

    va_list args;
    va_start(args, format);
    
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);

    logFile << buffer << std::endl;
    logFile.close();
}