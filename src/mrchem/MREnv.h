#ifndef MRENV_H
#define MRENV_H

#include <Eigen/Core>

#include "parallel.h"
#include "constants.h"
#include "Timer.h"

class MREnv {
public:
    static void initializeMRCPP();
    static void finalizeMRCPP(Timer t);
};

#endif // MRENV_H