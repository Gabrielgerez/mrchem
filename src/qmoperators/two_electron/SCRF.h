#pragma once

#include "chemistry/Nucleus.h"
#include "chemistry/Permittivity.h"
#include "qmfunctions/Density.h"
#include "qmfunctions/Orbital.h"
#include "qmfunctions/QMFunction.h"
#include "qmfunctions/qmfunction_fwd.h"

using PoissonOperator_p = std::shared_ptr<mrcpp::PoissonOperator>;
using DerivativeOperator_p = std::shared_ptr<mrcpp::DerivativeOperator<3>>;

namespace mrchem {
class ReactionPotential;
class SCRF final {
public:
    SCRF(Nuclei N, Permittivity e, PoissonOperator_p P, DerivativeOperator_p D);
    friend class ReactionPotential;
    void updateTotalDensity(OrbitalVector Phi,
                            double prec); // pass the electron orbitals and computes the total density
    void updatePotential(QMFunction new_potential);
    void updateDifferencePotential(QMFunction diff_potential);
    QMFunction getPotential() const { return this->potential; }
    QMFunction getDifferencePotential() const { return this->difference_potential; }
    void setReactionPotential(std::shared_ptr<ReactionPotential> new_Rp) { this->Rp = new_Rp; }

private:
    double apply_prec = -1.0;
    Nuclei nuclei;
    Permittivity epsilon;
    PoissonOperator_p poisson;
    DerivativeOperator_p derivative;
    Density rho_nuc;
    Density rho_tot;
    QMFunction difference_potential;
    QMFunction potential;
    std::shared_ptr<ReactionPotential> Rp;
    mrcpp::FunctionTreeVector<3> d_cavity; // Vector containing the 3 partial derivatives of the cavity function

    QMFunctionVector makeTerms(double prec);
    QMFunction updateGamma(QMFunction potential_nm1, double prec);
    double getNuclearEnergy();
    double getElectronicEnergy();
    double getTotalEnergy();
    void resetQMFunction(QMFunction &function);
};
} // namespace mrchem
