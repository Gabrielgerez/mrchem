#include "SCRF.h"
#include "MRCPP/MWOperators"
#include "ReactionPotential.h"
#include "chemistry/Nucleus.h"
#include "chemistry/Permittivity.h"
#include "chemistry/chemistry_utils.h"
#include "qmfunctions/Density.h"
#include "qmfunctions/Orbital.h"
#include "qmfunctions/QMFunction.h"
#include "qmfunctions/density_utils.h"
#include "qmfunctions/qmfunction_fwd.h"
#include "qmfunctions/qmfunction_utils.h"

using PoissonOperator_p = std::shared_ptr<mrcpp::PoissonOperator>;
using DerivativeOperator_p = std::shared_ptr<mrcpp::DerivativeOperator<3>>;
using OrbitalVector_p = std::shared_ptr<mrchem::OrbitalVector>;

namespace mrchem {
  SCRF::SCRF(Nuclei N, Permittivity e, OrbitalVector_p phi, PoissonOperator_p P, DerivativeOperator_p D)
        : nuclei(N)
        , epsilon(e)
        , Phi_p(phi)
        , poisson(P)
        , derivative(D)
        , rho_nuc(false)
        , rho_tot(false)
        , difference_potential(false)
        , potential(false) {}

void SCRF::updateTotalDensity(OrbitalVector Phi,
                              double prec) { // pass the electron orbitals and computes the total density
    if (not rho_nuc.hasReal()) { rho_nuc = chemistry::compute_nuclear_density(prec, this->nuclei, 1000); }
    std::cout << "integral of rho_nuc at SCRF::updateDifferencePotential:\t" << rho_nuc.integrate().real() << "\n";
    resetQMFunction(this->rho_tot);
    Density rho_el(false);
    density::compute(prec, rho_el, Phi, DensityType::Total);
    std::cout << "precision at SCRF::updateTotalDensity:\t" << prec << "\n";
    std::cout << "orbital vector at SCRT::updateDifferencePotential" << Phi.size() << "\n";
    rho_el.rescale(-1.0);
    std::cout << "integral of rho_el at SCRF::updateTotalDensity line 37:\t" << rho_el.integrate().real() << "\n";
    qmfunction::add(rho_tot, 1.0, rho_el, 1.0, rho_nuc, -1.0); // probably change this into a vector
}

QMFunction SCRF::updateGamma(QMFunction potential_nm1, double prec) {
    QMFunction gamma;
    gamma.alloc(NUMBER::Real);
    if (this->d_cavity.size() == 0) {
        mrcpp::FunctionTree<3> *dx_cavity = new mrcpp::FunctionTree<3>(*MRA);
        mrcpp::FunctionTree<3> *dy_cavity = new mrcpp::FunctionTree<3>(*MRA);
        mrcpp::FunctionTree<3> *dz_cavity = new mrcpp::FunctionTree<3>(*MRA);
        d_cavity.push_back(std::make_tuple(1.0, dx_cavity));
        d_cavity.push_back(std::make_tuple(1.0, dy_cavity));
        d_cavity.push_back(std::make_tuple(1.0, dz_cavity));
        mrcpp::project<3>(prec / 100, this->d_cavity, this->epsilon.getGradVector());
    }

    auto d_V = mrcpp::gradient(*derivative, potential_nm1.real());
    mrcpp::dot(prec, gamma.real(), d_V, d_cavity);
    gamma.rescale(std::log((epsilon.eps_in / epsilon.eps_out)) * (1.0 / (4.0 * MATHCONST::pi)));
    return gamma;
}

QMFunctionVector SCRF::makeTerms(double prec) {
    this->apply_prec = prec;
    QMFunction vacuum_potential;
    QMFunction rho_eff;
    QMFunction eps;
    QMFunction eps_inv;
    QMFunction numerator;
    QMFunction gamma;
    QMFunction total_potential;

    vacuum_potential.alloc(NUMBER::Real);
    rho_eff.alloc(NUMBER::Real);
    eps.alloc(NUMBER::Real);
    eps_inv.alloc(NUMBER::Real);
    numerator.alloc(NUMBER::Real);
    total_potential.alloc(NUMBER::Real);

    updateTotalDensity(*(this->Phi_p), prec);

    std::cout << "integral of rho_tot at SCRF::makeTerms:\t" << rho_tot.integrate().real() << "\n";
    mrcpp::apply(prec, vacuum_potential.real(), *poisson, rho_tot.real());
    std::cout << "integral of vacuum_potential at SCRF::makeTerms:\t" << vacuum_potential.integrate() << "\n";

    epsilon.flipFunction(false);
    qmfunction::project(eps, epsilon, NUMBER::Real, prec / 100);
    epsilon.flipFunction(true);
    qmfunction::project(eps_inv, epsilon, NUMBER::Real, prec / 100);
    qmfunction::add(numerator, 1.0, rho_tot, -1.0, eps, -1.0);
    std::cout << "integral of rho_tot at SCRF::makeTerms(), should be -1 because of charge:\t" << rho_tot.integrate().real() << "\n";
    std::cout << "integral of numerator at SCRF::makeTerms line 92:\t" << numerator.integrate() << "\n";
    qmfunction::multiply(rho_eff, numerator, eps_inv, prec);
    std::cout << "integral of rho_eff at SCRF::makeTerms line 93:\t" << rho_eff.integrate() << "\n";

    if (not this->potential.hasReal()) {
        QMFunction poisson_func;
        QMFunction V_n;
        poisson_func.alloc(NUMBER::Real);
        V_n.alloc(NUMBER::Real);

        gamma = updateGamma(vacuum_potential, prec);
        std::cout << "integral of gamma at SCRF::makeTerms line 102:\t" << gamma.integrate() << "\n";
        qmfunction::add(poisson_func, 1.0, rho_eff, 1.0, gamma, -1.0);
        std::cout << "integral of poisson_func at SCRF::makeTerms line 104:\t" << poisson_func.integrate() << "\n";
        mrcpp::apply(prec, V_n.real(), *poisson, poisson_func.real());
        this->difference_potential = V_n;
        std::cout << "integral of difference_potential at SCRF::makeTerms line 104:\t" << difference_potential.integrate() << "\n";
    }
    QMFunction Vr_np1;
    Vr_np1.alloc(NUMBER::Real);
    std::cout << "integral of Vr_np1 at SCRF::makeTerms line 107:\t" << Vr_np1.integrate() << "\n";
    qmfunction::add(Vr_np1, 1.0, potential, 1.0, difference_potential, -1.0);
    std::cout << "integral of Vr_np1 at SCRF::makeTerms line 109:\t" << Vr_np1.integrate() << "\n";

    qmfunction::add(total_potential, 1.0, Vr_np1, 1.0, vacuum_potential, -1.0);
    resetQMFunction(gamma);
    gamma = updateGamma(total_potential, prec);

    QMFunctionVector terms_vector;
    terms_vector.push_back(Vr_np1);
    terms_vector.push_back(gamma);
    terms_vector.push_back(rho_eff);
    terms_vector.push_back(vacuum_potential);

    return terms_vector;
}

void SCRF::updatePotential(QMFunction new_potential) {
    resetQMFunction(potential);
    this->potential = new_potential;
}

void SCRF::updateDifferencePotential(QMFunction diff_potential) {
    resetQMFunction(difference_potential);
    this->difference_potential = diff_potential;
    QMFunction new_potential;
    new_potential.alloc(NUMBER::Real);
    qmfunction::add(new_potential, 1.0, this->potential, 1.0, this->difference_potential, -1.0);
    this->Rp->updatePotential(new_potential);
}

double SCRF::getNuclearEnergy() {
    QMFunction V_n;
    QMFunction integral_product;
    V_n.alloc(NUMBER::Real);
    integral_product.alloc(NUMBER::Real);

    qmfunction::add(V_n, 1.0, this->potential, 1.0, this->difference_potential, -1.0);
    qmfunction::multiply(integral_product, this->rho_nuc, V_n, this->apply_prec);
    return integral_product.integrate().real();
}

double SCRF::getElectronicEnergy() {
    QMFunction V_n;
    QMFunction integral_product;
    QMFunction rho_el;
    V_n.alloc(NUMBER::Real);
    integral_product.alloc(NUMBER::Real);
    rho_el.alloc(NUMBER::Real);
    qmfunction::add(rho_el, 1.0, rho_tot, -1.0, rho_nuc, -1.0);
    qmfunction::add(V_n, 1.0, this->potential, 1.0, this->difference_potential, -1.0);
    qmfunction::multiply(integral_product, rho_el, V_n, this->apply_prec);
    return integral_product.integrate().real();
}

double SCRF::getTotalEnergy() {
    QMFunction V_n;
    QMFunction integral_product;
    V_n.alloc(NUMBER::Real);
    integral_product.alloc(NUMBER::Real);

    qmfunction::add(V_n, 1.0, this->potential, 1.0, this->difference_potential, -1.0);
    qmfunction::multiply(integral_product, this->rho_tot, V_n, this->apply_prec);
    return integral_product.integrate().real();
}

void SCRF::resetQMFunction(QMFunction &function) {
    if (function.hasReal()) function.free(NUMBER::Real);
    if (function.hasImag()) function.free(NUMBER::Imag);
    function.alloc(NUMBER::Real);
}
} // namespace mrchem
