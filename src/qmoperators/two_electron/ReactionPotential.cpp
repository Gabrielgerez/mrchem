#include "ReactionPotential.h"
#include "MRCPP/MWOperators"
#include "MRCPP/Printer"
#include "MRCPP/Timer"
#include "chemistry/chemistry_utils.h"
#include "qmfunctions/density_utils.h"
#include "qmfunctions/qmfunction_utils.h"

using mrcpp::Printer;
using mrcpp::Timer;

using PoissonOperator_p = std::shared_ptr<mrcpp::PoissonOperator>;
using DerivativeOperator_p = std::shared_ptr<mrcpp::DerivativeOperator<3>>;
using Cavity_p = std::shared_ptr<mrchem::Cavity>;
using OrbitalVector_p = std::shared_ptr<mrchem::OrbitalVector>;

namespace mrchem {

ReactionPotential::ReactionPotential(PoissonOperator_p P,
                                     DerivativeOperator_p D,
                                     Cavity_p C,
                                     const Nuclei &nucs,
                                     OrbitalVector_p Phi,
                                     int hist,
                                     double eps_i,
                                     double eps_o,
                                     bool islin)
        : QMPotential(1, false)
        , cavity(C)
        , nuclei(nucs)
        , orbitals(Phi)
        , poisson(P)
        , derivative(D)
        , rho_tot(false)
        , rho_el(false)
        , rho_nuc(false)
        , cavity_func(false)
        , gamma(false)
        , gammanp1(false)
        , history(hist)
        , e_i(eps_i)
        , e_o(eps_o)
        , is_lin(islin) {}

void ReactionPotential::setRhoEff(QMFunction &rho_eff_func, std::function<double(const mrcpp::Coord<3> &r)> eps) {
    rho_nuc = chemistry::compute_nuclear_density(this->apply_prec, this->nuclei, 1000);
    density::compute(this->apply_prec, rho_el, *orbitals, DENSITY::Total);
    rho_el.rescale(-1.0);
    qmfunction::add(rho_tot, 1.0, rho_el, 1.0, rho_nuc, -1.0);
    auto onesf = [eps](const mrcpp::Coord<3> &r) { return (1.0 / eps(r)) - 1.0; };
    QMFunction ones;
    ones.alloc(NUMBER::Real);
    mrcpp::build_grid(ones.real(), *cavity);
    qmfunction::project(ones, onesf, NUMBER::Real, this->apply_prec / 100);
    qmfunction::multiply(rho_eff_func, rho_tot, ones, this->apply_prec);
}

void ReactionPotential::setGamma(QMFunction const &inv_eps_func, QMFunction &gamma_func, QMFunction &V_func) {
    auto d_V = mrcpp::gradient(*derivative, V_func.real());
    if (is_lin) {
        QMFunction temp_func2;
        temp_func2.alloc(NUMBER::Real);
        mrcpp::dot(this->apply_prec, temp_func2.real(), d_V, this->d_cavity);
        qmfunction::multiply(gamma_func, temp_func2, inv_eps_func, this->apply_prec);
    } else if (not is_lin) {
        mrcpp::dot(this->apply_prec, gamma_func.real(), d_V, this->d_cavity);
    }
    gamma_func.rescale(this->d_coefficient / (4.0 * MATHCONST::pi));
    mrcpp::clear(d_V, true);
}

void ReactionPotential::accelerateConvergence(QMFunction &diff_func, QMFunction &temp, KAIN &kain) {
    OrbitalVector phi_n(0);
    OrbitalVector dPhi_n(0);
    phi_n.push_back(Orbital(SPIN::Paired));
    dPhi_n.push_back(Orbital(SPIN::Paired));

    phi_n[0].QMFunction::operator=(temp);
    dPhi_n[0].QMFunction::operator=(diff_func);

    kain.accelerate(this->apply_prec, phi_n, dPhi_n);

    temp.QMFunction::operator=(phi_n[0]);
    diff_func.QMFunction::operator=(dPhi_n[0]);

    phi_n.clear();
    dPhi_n.clear();
}

void ReactionPotential::poissonSolver(QMFunction rho_eff_func,
                                      QMFunction *diff_func,
                                      QMFunction *V_np1_func,
                                      double *error) {
    QMFunction poisson_func;
    if (V_np1_func->hasReal()) V_np1_func->free(NUMBER::Real);
    if (diff_func->hasReal()) diff_func->free(NUMBER::Real);
    V_np1_func->alloc(NUMBER::Real);

    qmfunction::add(poisson_func, 1.0, rho_eff_func, 1.0, this->gamma, -1.0);
    mrcpp::apply(this->apply_prec, V_np1_func->real(), *poisson, poisson_func.real());
    qmfunction::add(*diff_func, -1.0, *this, 1.0, *V_np1_func, -1.0);
    *error = diff_func->norm();
}

void ReactionPotential::SCRF(QMFunction *V_tot_func,
                             QMFunction *V_vac_func,
                             QMFunction *rho_eff_func,
                             QMFunction &temp,
                             const QMFunction &inv_eps_func) {
    mrcpp::print::header(0, "Running SCRF");
    KAIN kain(this->history);
    double error = 1.00;
    int iter = 1;
    for (int iter = 1; error >= this->apply_prec; iter++) {
        if (this->gamma.hasReal()) this->gamma.free(NUMBER::Real);
        if (V_tot_func->hasReal()) V_tot_func->free(NUMBER::Real);
        QMFunction V_np1_func;
        QMFunction diff_func;
        V_tot_func->alloc(NUMBER::Real);
        this->gamma.alloc(NUMBER::Real);

        qmfunction::add(*V_tot_func, 1.0, temp, 1.0, *V_vac_func, -1.0);
        setGamma(inv_eps_func, this->gamma, *V_tot_func);

        poissonSolver(*rho_eff_func, &diff_func, &V_np1_func, &error);

        if (iter > 1 and this->history > 0) { accelerateConvergence(diff_func, temp, kain); }

        V_np1_func.free(NUMBER::Real);
        qmfunction::add(V_np1_func, 1.0, temp, 1.0, diff_func, -1.0);
        temp = V_np1_func;
        println(0, "Iter:\t");
        println(0, iter);
        println(0, "error:\t");
        println(0, error);
    }
}

void ReactionPotential::setup(double prec) {

    setApplyPrec(prec);
    QMFunction &temp = *this;
    QMFunction V_vac_func;
    QMFunction inv_eps_func;
    QMFunction rho_eff_func;
    QMFunction V_tot_func;

    Cavity &C_tmp = *this->cavity;
    double eps_i = this->e_i, eps_o = this->e_o;

    if (is_lin) {
        inv_eps_func.alloc(NUMBER::Real);
        mrcpp::build_grid(inv_eps_func.real(), *cavity);

        auto eps = [C_tmp, eps_i, eps_o](const mrcpp::Coord<3> &r) { return eps_o + C_tmp.evalf(r) * (eps_i - eps_o); };

        auto inv_eps = [eps](const mrcpp::Coord<3> &r) { return 1.0 / eps(r); };
        qmfunction::project(inv_eps_func, inv_eps, NUMBER::Real, prec / 100);

        this->d_coefficient = e_i - e_o;
        setRhoEff(rho_eff_func, eps);

    } else {
        auto eps = [C_tmp, eps_i, eps_o](const mrcpp::Coord<3> &r) {
            return eps_i * std::exp(std::log(eps_o / eps_i) * (1 - C_tmp.evalf(r)));
        };
        this->d_coefficient = std::log(e_i / e_o);
        setRhoEff(rho_eff_func, eps);
    }

    V_vac_func.alloc(NUMBER::Real);
    mrcpp::apply(prec, V_vac_func.real(), *poisson, rho_tot.real());

    if (not temp.hasReal()) {

        mrcpp::FunctionTree<3> *dx_cavity = new mrcpp::FunctionTree<3>(*MRA);
        mrcpp::FunctionTree<3> *dy_cavity = new mrcpp::FunctionTree<3>(*MRA);
        mrcpp::FunctionTree<3> *dz_cavity = new mrcpp::FunctionTree<3>(*MRA);
        this->d_cavity.push_back(std::make_tuple(1.0, dx_cavity));
        this->d_cavity.push_back(std::make_tuple(1.0, dy_cavity));
        this->d_cavity.push_back(std::make_tuple(1.0, dz_cavity));
        mrcpp::project<3>(prec / 100, this->d_cavity, cavity->getGradVector());

        QMFunction zeroth_poisson;
        zeroth_poisson.alloc(NUMBER::Real);
        gamma.alloc(NUMBER::Real);
        setGamma(inv_eps_func, gamma, V_vac_func);
        qmfunction::add(zeroth_poisson, 1.0, gamma, 1.0, rho_eff_func, -1.0);
        QMFunction tmp_Vr_func;
        tmp_Vr_func.alloc(NUMBER::Real);
        mrcpp::apply(prec, tmp_Vr_func.real(), *poisson, zeroth_poisson.real());
        temp = tmp_Vr_func;
        SCRF(&V_tot_func, &V_vac_func, &rho_eff_func, temp, inv_eps_func);
    } else {
        if (this->variational) {
            // Variational implementation of generalized poisson equation.
            if (gamma.hasReal()) gamma.free(NUMBER::Real);
            qmfunction::deep_copy(gamma, gammanp1);
            QMFunction V_np1_func;
            QMFunction diff_func;
            double error;
            V_np1_func.alloc(NUMBER::Real);

            poissonSolver(rho_eff_func, &diff_func, &V_np1_func, &error);

            temp = V_np1_func;

            println(0, "error:");
            println(0, error);
        } else {
            SCRF(&V_tot_func, &V_vac_func, &rho_eff_func, temp, inv_eps_func);
        }
    }
    if (gammanp1.hasReal()) gammanp1.free(NUMBER::Real);
    if (V_tot_func.hasReal()) V_tot_func.free(NUMBER::Real);
    gammanp1.alloc(NUMBER::Real);
    V_tot_func.alloc(NUMBER::Real);

    qmfunction::add(V_tot_func, 1.0, temp, 1.0, V_vac_func, -1.0);
    setGamma(inv_eps_func, gammanp1, V_tot_func);
}

double &ReactionPotential::getTotalEnergy() {
    QMFunction temp_prod_func;
    qmfunction::multiply(temp_prod_func, rho_tot, *this, this->apply_prec);
    totalEnergy = temp_prod_func.integrate().real();
    return totalEnergy;
}

double &ReactionPotential::getElectronicEnergy() {
    QMFunction temp_prod_func;
    qmfunction::multiply(temp_prod_func, rho_el, *this, this->apply_prec);
    electronicEnergy = temp_prod_func.integrate().real();
    return electronicEnergy;
}

double &ReactionPotential::getNuclearEnergy() {
    QMFunction temp_prod_func;
    qmfunction::multiply(temp_prod_func, rho_nuc, *this, this->apply_prec);
    nuclearEnergy = temp_prod_func.integrate().real();
    return nuclearEnergy;
}

double &ReactionPotential::getElectronIn() {
    QMFunction temp_prod_func;
    qmfunction::multiply(temp_prod_func, rho_el, cavity_func, this->apply_prec);
    electronsIn = temp_prod_func.integrate().real();
    return electronsIn;
}

void ReactionPotential::clear() {
    clearApplyPrec();
    rho_tot.free(NUMBER::Real);
    rho_el.free(NUMBER::Real);
    rho_nuc.free(NUMBER::Real);
}

} // namespace mrchem
