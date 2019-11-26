/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2020 Stig Rune Jensen, Luca Frediani, Peter Wind and contributors.
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

#pragma once

#include "MRCPP/MWFunctions"

namespace mrchem {

class Cavity final : public mrcpp::RepresentableFunction<3> {
public:
    Cavity(std::vector<mrcpp::Coord<3>> &coords, std::vector<double> &R, double slope);
    double evalf(const mrcpp::Coord<3> &r) const override;
    auto getGradVector() { return this->gradvector; }
    std::vector<mrcpp::Coord<3>> &getCoordinates() { return pos; }
    std::vector<double> &getRadius() { return R; }

protected:
    std::vector<mrcpp::Coord<3>> pos;
    std::vector<double> R;
    double d;
    std::vector<std::function<double(const mrcpp::Coord<3> &r)>> gradvector;

    void setGradVector();
    bool isVisibleAtScale(int scale, int nQuadPts) const override;
    bool isZeroOnInterval(const double *a, const double *b) const override;
};
} // namespace mrchem
