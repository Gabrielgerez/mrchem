/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2021 Stig Rune Jensen, Luca Frediani, Peter Wind and contributors.
 *
 * This file is part of MRChem.
 *
 * MRChem is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MRChem is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with MRChem.  If not, see <https://www.gnu.org/licenses/>.
 *
 * For information on the complete list of contributors to MRChem, see:
 * <https://mrchem.readthedocs.io/>
 */
#include "Cavity.h"
#include "utils/math_utils.h"
namespace mrchem {

/** @brief Initializes the members of the class and constructs the analytical gradient vector of the Cavity. */
Cavity::Cavity(std::vector<mrcpp::Coord<3>> &centers, std::vector<double> &radii, double width)
        : width(width)
        , radii(radii)
        , centers(centers) {
    setGradVector();
}

/**  @relates mrchem::Cavity
 *   @brief Constructs a single element of the gradient of the Cavity.
 *
 *   This constructs the analytical partial derivative of the Cavity \f$C\f$ with respect to \f$x\f$, \f$y\f$ or \f$z\f$
 * coordinates and evaluates it at a point \f$\mathbf{r}\f$. This is given for \f$x\f$ by
 * \f[
 *    \frac{\partial C\left(\mathbf{r}\right)}{\partial x} = \left(1 - C{\left(\mathbf{r} \right)}\right)
 *                                                           \sum_{i=1}^{N} - \frac{\left(x-{x}_{i}\right)e^{-
 * \frac{\operatorname{s_{i}}^{2}{\left(\mathbf{r} \right)}}{\sigma^{2}}}}
 *                                                           {\sqrt{\pi}\sigma\left(0.5
 * \operatorname{erf}{\left(\frac{\operatorname{s_{i}}{\left(\mathbf{r}  \right)}}{\sigma} \right)}
 *                                                           + 0.5\right) \left| \mathbf{r} - \mathbf{r}_{i} \right|}
 * \f]
 * where the subscript \f$i\f$ is the index related to each
 * sphere in the cavity, and \f$\operatorname{s}\f$ is the signed normal distance from the surface of each sphere.
 *   @param r The coordinates of a test point in 3D space.
 *   @param index An integer that defines the variable of differentiation (0->x, 1->z and 2->z).
 *   @param centers A vector containing the coordinates of the centers of the spheres in the cavity.
 *   @param radii A vector containing the radii of the spheres.
 *   @param width A double value describing the width of the transition at the boundary of the spheres.
 *   @return A double number which represents the value of the differential (w.r.t. x, y or z) at point r.
 */

auto ddCavity(const mrcpp::Coord<3> &r,
              int index,
              const std::vector<mrcpp::Coord<3>> &centers,
              std::vector<double> &radii,
              double width) -> double {

    auto s_i = [centers, radii](const mrcpp::Coord<3> r, int i) -> double {
        return math_utils::calc_distance(centers[i], r) - radii[i];
    };
    auto C_i = [s_i, width](const mrcpp::Coord<3> r, int i) -> double {
        return 1.0 - 0.5 * (1 + std::erf(s_i(r, i) / width));
    };
    auto C = [C_i, centers](const mrcpp::Coord<3> r) -> double {
        double c = 1.0;
        for (int i = 0; i < centers.size(); i++) { c *= 1.0 - C_i(r, i); }
        return 1.0 - c;
    };
    auto ds_i = [s_i, index, radii, centers](const mrcpp::Coord<3> r, int i) -> double {
        return (r[index] - centers[i][index]) / (s_i(r, i) + radii[i]);
    };
    auto dC_i = [s_i, ds_i, width](const mrcpp::Coord<3> r, int i) -> double {
        return -std::exp(-std::pow(s_i(r, i) / width, 2)) * (ds_i(r, i)) / (std::sqrt(MATHCONST::pi) * width);
    };
    auto dC = [C, C_i, dC_i, centers](const mrcpp::Coord<3> r) -> double {
        double S = 0.0;
        for (int i = 0; i < centers.size(); i++) {
            double numerator = ((dC_i(r, i) < 1.0e-12) and (dC_i(r, i) >= 0.0))
                                   ? 1.0e-12
                                   : ((dC_i(r, i) > 1.0e-12) and (dC_i(r, i) <= 0.0)) ? -1.0e-12 : dC_i(r, i);
            double denominator =
                (((1.0 - C_i(r, i)) < 1.0e-12) and ((1.0 - C_i(r, i)) >= 0.0))
                    ? 1.0e-12
                    : (((1.0 - C_i(r, i)) > 1.0e-12) and ((1.0 - C_i(r, i)) <= 0.0)) ? -1.0e-12 : (1.0 - C_i(r, i));
            // please make this into and abs(denominator - 1.0e-12) thing
            S += numerator / denominator;
        }
        return (1 - C(r)) * S;
    };
    auto dds_i = [index, centers, s_i, radii](const mrcpp::Coord<3> r, int i) -> double {
        return -std::pow((r[index] - centers[i][index]), 2) / std::pow((s_i(r, i) + radii[i]), 3.0) +
               1 / (s_i(r, i) + radii[i]);
    };
    auto ddC_i = [index, width, s_i, ds_i, dds_i](const mrcpp::Coord<3> r, int i) -> double {
        return -std::exp(-std::pow(s_i(r, i) / width, 2)) * dds_i(r, i) / (std::sqrt(MATHCONST::pi) * width) +
               2.0 * s_i(r, i) * std::exp(-std::pow(s_i(r, i) / width, 2)) * std::pow(ds_i(r, i), 2) /
                   (std::sqrt(MATHCONST::pi) * std::pow(width, 3));
    };
    auto ddC = [centers, C, C_i, dC_i, ddC_i](const mrcpp::Coord<3> r) -> double {
        double S1 = 0.0;
        double S2 = 0.0;
        for (int i = 0; i < centers.size(); i++) {
            double DCi = ((dC_i(r, i) < 1.0e-12) and (dC_i(r, i) >= 0.0))
                             ? 1.0e-12
                             : ((dC_i(r, i) > 1.0e-12) and (dC_i(r, i) <= 0.0)) ? -1.0e-12 : dC_i(r, i);
            double DDCi = ((ddC_i(r, i) < 1.0e-12) and (ddC_i(r, i) >= 0.0))
                              ? 1.0e-12
                              : ((ddC_i(r, i) > 1.0e-12) and (ddC_i(r, i) <= 0.0)) ? -1.0e-12 : ddC_i(r, i);
            double denominator =
                (((1.0 - C_i(r, i)) < 1.0e-12) and ((1.0 - C_i(r, i)) >= 0.0))
                    ? 1.0e-12
                    : (((1.0 - C_i(r, i)) > 1.0e-12) and ((1.0 - C_i(r, i)) <= 0.0)) ? -1.0e-12 : (1.0 - C_i(r, i));

            S1 += DCi / denominator; // need to square this one after wards
            S2 += (DDCi / denominator) + std::pow(DCi / denominator, 2);
        }
        return -(1 - C(r)) * std::pow(S1, 2) + (1 - C(r)) * S2;
    };
    return ddC(r);
}

auto dCavity(const mrcpp::Coord<3> &r,
             int index,
             const std::vector<mrcpp::Coord<3>> &centers,
             std::vector<double> &radii,
             double width) -> double {
    auto s_i = [centers, radii](const mrcpp::Coord<3> r, int i) -> double {
        return math_utils::calc_distance(centers[i], r) - radii[i];
    };
    auto C_i = [s_i, width](const mrcpp::Coord<3> r, int i) -> double {
        return 1.0 - 0.5 * (1 + std::erf(s_i(r, i) / width));
    };
    auto C = [C_i, centers](const mrcpp::Coord<3> r) -> double {
        double c = 1.0;
        for (int i = 0; i < centers.size(); i++) { c *= 1.0 - C_i(r, i); }
        return 1.0 - c;
    };
    auto ds_i = [s_i, index, radii, centers](const mrcpp::Coord<3> r, int i) -> double {
        return (r[index] - centers[i][index]) / (s_i(r, i) + radii[i]);
    };
    auto dC_i = [s_i, ds_i, width](const mrcpp::Coord<3> r, int i) -> double {
        return -1.0 * std::exp(-std::pow(s_i(r, i) / width, 2)) * (ds_i(r, i)) / (sqrt(MATHCONST::pi) * width);
    };
    auto dC = [C, C_i, dC_i, centers](const mrcpp::Coord<3> r) -> double {
        double S = 0.0;
        for (int i = 0; i < centers.size(); i++) {
            double numerator = ((dC_i(r, i) < 1.0e-12) and (dC_i(r, i) >= 0.0))
                                   ? 1.0e-12
                                   : ((dC_i(r, i) > 1.0e-12) and (dC_i(r, i) <= 0.0)) ? -1.0e-12 : dC_i(r, i);
            double denominator =
                (((1.0 - C_i(r, i)) < 1.0e-12) and ((1.0 - C_i(r, i)) >= 0.0))
                    ? 1.0e-12
                    : (((1.0 - C_i(r, i)) > 1.0e-12) and ((1.0 - C_i(r, i)) <= 0.0)) ? -1.0e-12 : (1.0 - C_i(r, i));
            S += numerator / denominator;
        }
        return (1 - C(r)) * S;
    };
    return dC(r);
    /*    double C = 1.0;
double DC = 0.0;
for (int i = 0; i < centers.size(); i++) {
double s = math_utils::calc_distance(centers[i], r) - radii[i];
double ds = (r[index] - centers[i][index]) / (math_utils::calc_distance(centers[i], r));
double Theta = 0.5 * (1 + std::erf(s / width));
double Ci = 1.0 - Theta;
C *= 1.0 - Ci;

double DCi = -(1.0 / (width * std::sqrt(MATHCONST::pi))) * std::exp(-std::pow(s / width, 2.0)) * ds;

double numerator = ((DCi < 1.0e-12) and (DCi >= 0.0)) ? 1.0e-12 : ((DCi > 1.0e-12) and (DCi <= 0.0)) ? -1.0e-12  : DCi;
double denominator = (((1.0 - Ci) < 1.0e-12) and ((1.0 - Ci) >= 0.0)) ? 1.0e-12 : (((1.0 - Ci) > 1.0e-12) and ((1.0 -
Ci) <= 0.0)) ? -1.0e-12 : (1.0 - Ci); DC += numerator / denominator;
}
DC = C * DC;
return DC;*/
}
/** @brief Sets the different partial derivatives in the \link #gradvector gradient \endlink of the Cavity. */
void Cavity::setGradVector() {
    auto p_gradcavity = [this](const mrcpp::Coord<3> &r, int index) {
        return dCavity(r, index, centers, radii, width);
    };
    for (auto i = 0; i < 3; i++) {
        this->gradvector.push_back(
            [i, p_gradcavity](const mrcpp::Coord<3> &r) -> double { return p_gradcavity(r, i); });
    }
}
/** @brief Evaluates the value of the cavity at a 3D point \f$\mathbf{r}\f$
 *  @param r coordinate of 3D point at which the Cavity is to be evaluated at.
 *  @return double value of the Cavity at point \f$\mathbf{r}\f$
 */
double Cavity::evalf(const mrcpp::Coord<3> &r) const {
    double C = 1.0;
    for (int i = 0; i < centers.size(); i++) {
        double s = math_utils::calc_distance(centers[i], r) - radii[i];
        double Theta = 0.5 * (1 + std::erf(s / width));
        double Ci = 1 - Theta;
        C *= 1 - Ci;
    }
    C = 1 - C;
    return C;
}

bool Cavity::isVisibleAtScale(int scale, int nQuadPts) const {

    auto visibleScale = static_cast<int>(-std::floor(std::log2(nQuadPts * 2.0 * this->width)));

    if (scale < visibleScale) { return false; }

    return true;
}

bool Cavity::isZeroOnInterval(const double *a, const double *b) const {
    for (int k = 0; k < centers.size(); k++) {
        for (int i = 0; i < 3; i++) {
            double cavityMinOut = (this->centers[k][i] - radii[i]) - 3.0 * this->width;
            double cavityMinIn = (this->centers[k][i] - radii[i]) + 3.0 * this->width;
            double cavityMaxIn = (this->centers[k][i] + radii[i]) - 3.0 * this->width;
            double cavityMaxOut = (this->centers[k][i] + radii[i]) + 3.0 * this->width;
            if (a[i] > cavityMaxOut or (a[i] > cavityMinIn and b[i] < cavityMaxIn) or b[i] < cavityMinOut) {
                return true;
            }
        }
    }
    return false;
}

} // namespace mrchem
