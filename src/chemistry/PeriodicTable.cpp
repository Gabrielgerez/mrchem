/*
 * MRChem, a numerical real-space code for molecular electronic structure
 * calculations within the self-consistent field (SCF) approximations of quantum
 * chemistry (Hartree-Fock and Density Functional Theory).
 * Copyright (C) 2019 Stig Rune Jensen, Luca Frediani, Peter Wind and contributors.
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

/**
 *
 * \date Jun 7, 2009
 * \author Jonas Juselius <jonas.juselius@uit.no> \n
 *         CTCC, University of Tromsø
 *
 *
 */

/*
 * \date Mars 18, 2019
 * edit vdW radius from  Allinger's MM3 set (doi:10.1016/S0166-1280(09)80008-0)
 */

#include "MRCPP/Printer"

#include "Element.h"
#include "PeriodicTable.h"

using std::string;

namespace mrchem {

const Element &PeriodicTable::getElement(const char *sym) const {
    string id(sym);
    id[0] = toupper(id[0]);
    for (unsigned int i = 1; i < id.size(); i++) { id[i] = tolower(id[i]); }
    if (id.size() > 2) {
        if (this->byName.find(id) != this->byName.end()) { return *this->byName[id]; }
    } else {
        if (this->bySymbol.find(id) != this->bySymbol.end()) { return *this->bySymbol[id]; }
    }
    MSG_FATAL("Invalid element: " << id);
}

const Element &PeriodicTable::getElement(int Z) const {
    if (Z < 0 or Z > PeriodicTable::nElements) { MSG_FATAL("Invalid element number: " << Z); }
    return this->elements[Z];
}
// clang-format off
const Element PeriodicTable::elements[PeriodicTable::nElements] = {
    Element(   1, "Q" , "Dummy"         ,   0.0000000, 0.00000, 0.000000, -1.000000 ),
    Element(   1, "H" , "Hydrogen"      ,   1.0079400, 1.35000, 0.320000, -1.000000 ),
    Element(   2, "He", "Helium"        ,   4.0026020, 1.27500, 0.930000, -1.000000 ),
    Element(   3, "Li", "Lithium"       ,   6.9410000, 2.12500, 1.230000, -1.000000 ),
    Element(   4, "Be", "Beryllium"     ,   9.0121820, 1.85833, 0.900000, -1.000000 ),
    Element(   5, "B" , "Boron"         ,  10.8110000, 1.79167, 0.820000, -1.000000 ),
    Element(   6, "C" , "Carbon"        ,  12.0110000, 1.70000, 0.770000, -1.000000 ),
    Element(   7, "N" , "Nitrogen"      ,  14.0067400, 1.60833, 0.750000, -1.000000 ),
    Element(   8, "O" , "Oxygen"        ,  15.9994000, 1.51667, 0.730000, -1.000000 ),
    Element(   9, "F" , "Fluorine"      ,  18.9984032, 1.42500, 0.720000, -1.000000 ),
    Element(  10, "Ne", "Neon"          ,  20.1797000, 1.33333, 0.710000, -1.000000 ),
    Element(  11, "Na", "Sodium"        ,  22.9897680, 2.25000, 1.540000, -1.000000 ),
    Element(  12, "Mg", "Magnesium"     ,  24.3050000, 2.02500, 1.360000, -1.000000 ),
    Element(  13, "Al", "Aluminum"      ,  26.9815390, 1.96667, 1.180000, -1.000000 ),
    Element(  14, "Si", "Silicon"       ,  28.0855000, 1.90833, 1.110000, -1.000000 ),
    Element(  15, "P" , "Phosphorus"    ,  30.9736200, 1.85000, 1.060000, -1.000000 ),
    Element(  16, "S" , "Sulfur"        ,  32.0660000, 1.79167, 1.020000, -1.000000 ),
    Element(  17, "Cl", "Chlorine"      ,  35.4527000, 1.72500, 0.990000, -1.000000 ),
    Element(  18, "Ar", "Argon"         ,  39.9480000, 1.65833, 0.980000, -1.000000 ),
    Element(  19, "K" , "Potassium"     ,  39.0983000, 2.57500, 2.030000, -1.000000 ),
    Element(  20, "Ca", "Calcium"       ,  40.0780000, 2.34167, 1.910000, -1.000000 ),
    Element(  21, "Sc", "Scandium"      ,  44.9559100, 2.17500, 1.620000, -1.000000 ),
    Element(  22, "Ti", "Titanium"      ,  47.8800000, 1.99167, 1.450000, -1.000000 ),
    Element(  23, "V" , "Vanadium"      ,  50.9415000, 1.90833, 1.340000, -1.000000 ),
    Element(  24, "Cr", "Chromium"      ,  51.9961000, 1.87500, 1.180000, -1.000000 ),
    Element(  25, "Mn", "Manganese"     ,  54.9308500, 1.86667, 1.170000, -1.000000 ),
    Element(  26, "Fe", "Iron"          ,  55.8470000, 1.85833, 1.170000, -1.000000 ),
    Element(  27, "Co", "Cobalt"        ,  58.9332000, 1.85833, 1.160000, -1.000000 ),
    Element(  28, "Ni", "Nickel"        ,  58.6900000, 1.85000, 1.150000, -1.000000 ),
    Element(  29, "Cu", "Copper"        ,  63.5460000, 1.88333, 1.170000, -1.000000 ),
    Element(  30, "Zn", "Zinc"          ,  65.3900000, 1.90833, 1.250000, -1.000000 ),
    Element(  31, "Ga", "Gallium"       ,  69.7230000, 2.05000, 1.260000, -1.000000 ),
    Element(  32, "Ge", "Germanium"     ,  72.6100000, 2.03333, 1.220000, -1.000000 ),
    Element(  33, "As", "Arsenic"       ,  74.9215900, 1.96667, 1.200000, -1.000000 ),
    Element(  34, "Se", "Selenium"      ,  78.9600000, 1.90833, 1.160000, -1.000000 ),
    Element(  35, "Br", "Bromine"       ,  79.9040000, 1.85000, 1.140000, -1.000000 ),
    Element(  36, "Kr", "Krypton"       ,  83.8000000, 1.79167, 1.120000, -1.000000 ),
    Element(  37, "Rb", "Rubidium"      ,  85.4678000, 2.70833, 2.160000, -1.000000 ),
    Element(  38, "Sr", "Strontium"     ,  87.6200000, 2.50000, 1.910000, -1.000000 ),
    Element(  39, "Y" , "Yttrium"       ,  88.9058500, 2.25833, 1.620000, -1.000000 ),
    Element(  40, "Zr", "Zirconium"     ,  91.2240000, 2.11667, 1.450000, -1.000000 ),
    Element(  41, "Nb", "Niobium"       ,  92.9063800, 2.02500, 1.340000, -1.000000 ),
    Element(  42, "Mo", "Molybdenum"    ,  95.9400000, 1.99167, 1.300000, -1.000000 ),
    Element(  43, "Tc", "Technetium"    , -98.0000000, 1.96667, 1.270000, -1.000000 ),
    Element(  44, "Ru", "Ruthenium"     , 101.0700000, 1.95000, 1.250000, -1.000000 ),
    Element(  45, "Rh", "Rhodium"       , 102.9055000, 1.95000, 1.250000, -1.000000 ),
    Element(  46, "Pd", "Palladium"     , 106.4200000, 1.97500, 1.280000, -1.000000 ),
    Element(  47, "Ag", "Silver"        , 107.8682000, 2.02500, 1.340000, -1.000000 ),
    Element(  48, "Cd", "Cadmium"       , 112.4110000, 2.08333, 1.480000, -1.000000 ),
    Element(  49, "In", "Indium"        , 114.8200000, 2.20000, 1.440000, -1.000000 ),
    Element(  50, "Sn", "Tin"           , 118.7100000, 2.15833, 1.410000, -1.000000 ),
    Element(  51, "Sb", "Antimony"      , 121.7500000, 2.10000, 1.400000, -1.000000 ),
    Element(  52, "Te", "Tellurium"     , 127.6000000, 2.03333, 1.360000, -1.000000 ),
    Element(  53, "I" , "Iodine"        , 126.9044700, 1.96667, 1.330000, -1.000000 ),
    Element(  54, "Xe", "Xenon"         , 131.2900000, 1.90000, 1.310000, -1.000000 ),
    Element(  55, "Cs", "Cesium"        , 132.9054300, 2.86667, 2.350000, -1.000000 ),
    Element(  56, "Ba", "Barium"        , 137.3270000, 2.55833, 1.980000, -1.000000 ),
    Element(  57, "La", "Lanthanum"     , 138.9055000, 2.31667, 1.690000, -1.000000 ),
    Element(  58, "Ce", "Cerium"        , 140.1150000, 2.28333, 1.650000, -1.000000 ),
    Element(  59, "Pr", "Praseodymium"  , 140.9076500, 2.27500, 1.650000, -1.000000 ),
    Element(  60, "Nd", "Neodymium"     , 144.2400000, 2.27500, 1.640000, -1.000000 ),
    Element(  61, "Pm", "Promethium"    ,-145.0000000, 2.26667, 1.630000, -1.000000 ),
    Element(  62, "Sm", "Samarium"      , 150.3600000, 2.25833, 1.620000, -1.000000 ),
    Element(  63, "Eu", "Europium"      , 151.9650000, 2.45000, 1.850000, -1.000000 ),
    Element(  64, "Gd", "Gadolinium"    , 157.2500000, 2.25833, 1.610000, -1.000000 ),
    Element(  65, "Tb", "Terbium"       , 158.9253400, 2.25000, 1.590000, -1.000000 ),
    Element(  66, "Dy", "Dysprosium"    , 162.5000000, 2.24167, 1.590000, -1.000000 ),
    Element(  67, "Ho", "Holmium"       , 164.9303200, 2.22500, 1.580000, -1.000000 ),
    Element(  68, "Er", "Erbium"        , 167.2600000, 2.22500, 1.570000, -1.000000 ),
    Element(  69, "Tm", "Thulium"       , 168.9342100, 2.22500, 1.560000, -1.000000 ),
    Element(  70, "Yb", "Ytterbium"     , 173.0400000, 2.32500, 1.740000, -1.000000 ),
    Element(  71, "Lu", "Lutetium"      , 174.9670000, 2.20833, 1.560000, -1.000000 ),
    Element(  72, "Hf", "Hafnium"       , 178.4900000, 2.10833, 1.440000, -1.000000 ),
    Element(  73, "Ta", "Tantalum"      , 180.9479000, 2.02500, 1.340000, -1.000000 ),
    Element(  74, "W" , "Tungsten"      , 183.8500000, 1.99167, 1.300000, -1.000000 ),
    Element(  75, "Re", "Rhenium"       , 186.2070000, 1.97500, 1.280000, -1.000000 ),
    Element(  76, "Os", "Osmium"        , 190.2000000, 1.95833, 1.260000, -1.000000 ),
    Element(  77, "Ir", "Iridium"       , 192.2200000, 1.96667, 1.270000, -1.000000 ),
    Element(  78, "Pt", "Platinum"      , 195.0800000, 1.99167, 1.300000, -1.000000 ),
    Element(  79, "Au", "Gold"          , 196.9665400, 2.02500, 1.340000, -1.000000 ),
    Element(  80, "Hg", "Mercury"       , 200.5900000, 2.10833, 1.490000, -1.000000 ),
    Element(  81, "Tl", "Thallium"      , 204.3833000, 2.15833, 1.480000, -1.000000 ),
    Element(  82, "Pb", "Lead"          , 207.2000000, 2.28333, 1.470000, -1.000000 ),
    Element(  83, "Bi", "Bismuth"       , 208.9803700, 2.21667, 1.460000, -1.000000 ),
    Element(  84, "Po", "Polonium"      ,-209.0000000, 2.15833, 1.460000, -1.000000 ),
    Element(  85, "At", "Astatine"      , 210.0000000, 2.09167, 1.450000, -1.000000 ),
    Element(  86, "Rn", "Radon"         ,-222.0000000, 2.02500, 1.430000, -1.000000 ),
    Element(  87, "Fr", "Francium"      ,-223.0000000, 3.03333, 2.500000, -1.000000 ),
    Element(  88, "Ra", "Radium"        , 226.0250000, 2.72500, 2.400000, -1.000000 ),
    Element(  89, "Ac", "Actinium"      , 227.0280000, 2.56667, 2.200000, -1.000000 ),
    Element(  90, "Th", "Thorium"       , 232.0381000, 2.28333, 1.650000, -1.000000 ),
    Element(  91, "Pa", "Protactinium"  , 231.0358800, 2.20000, 0.000000, -1.000000 ),
    Element(  92, "U" , "Uranium"       , 238.0289000, 2.10000, 1.420000, -1.000000 ),
    Element(  93, "Np", "Neptunium"     , 237.0480000, 2.10000, 0.000000, -1.000000 ),
    Element(  94, "Pu", "Plutonium"     ,-244.0000000, 2.10000, 0.000000, -1.000000 ),
    Element(  95, "Am", "Americium"     ,-243.0000000, 3.02000, 0.000000, -1.000000 ), // Americum to lawrencium have no radius in source PCMSolver
    Element(  96, "Cm", "Curium"        ,-247.0000000, 2.99000, 0.000000, -1.000000 ),
    Element(  97, "Bk", "Berkelium"     ,-247.0000000, 2.97000, 0.000000, -1.000000 ),
    Element(  98, "Cf", "Californium"   ,-251.0000000, 2.95000, 0.000000, -1.000000 ),
    Element(  99, "Es", "Einsteinium"   ,-252.0000000, 2.92000, 0.000000, -1.000000 ),
    Element( 100, "Fm", "Fermium"       ,-257.0000000, 2.90000, 0.000000, -1.000000 ),
    Element( 101, "Md", "Mendelevium"   ,-258.0000000, 2.87000, 0.000000, -1.000000 ),
    Element( 102, "No", "Nobelium"      ,-259.0000000, 2.85000, 0.000000, -1.000000 ),
    Element( 103, "Lr", "Lawrencium"    ,-260.0000000, 2.82000, 0.000000, -1.000000 ), // Americum to lawrencium have no radius in source PCMSolver
    Element( 104, "Rf", "Rutherfordium" ,-257.0000000, 2.27500, 0.000000, -1.000000 ),
    Element( 105, "Ha", "Hahnium"       ,-262.0000000, 2.19167, 0.000000, -1.000000 ),
    Element( 106, "Sq", "Seaborgium"    ,-263.0000000, 0.00000, 0.000000, -1.000000 ), // Seaborgium has no radius in source  PCMSolver
    Element( 107, "Ns", "Nielsbohrium"  ,-262.0000000, 1.35000, 0.000000, -1.000000 ),
    Element( 108, "Hs", "Hassium"       ,-264.0000000, 0.00000, 0.000000, -1.000000 ),
    Element( 109, "Mt", "Meitnerium"    ,-266.0000000, 0.00000, 0.000000, -1.000000 ),
    Element(   0, "X" , "Heavy"         ,   1.0e6    , 0.00000, 0.000000, -1.000000 ),
    Element(   0, "Gh", "Heavy"         ,   1.0e6    , 0.00000, 0.000000, -1.000000 )
};
// clang-format on

PeriodicTable::map_t PeriodicTable::_init_byname() {
    string name;
    map_t _map;
    for (const auto &element : elements) {
        name = element.getName();
        _map[name] = &element;
    }
    return _map;
}

PeriodicTable::map_t PeriodicTable::_init_bysymbol() {
    string name;
    map_t _map;
    for (const auto &element : elements) {
        name = element.getSymbol();
        _map[name] = &element;
    }
    return _map;
}

PeriodicTable::map_t PeriodicTable::byName = _init_byname();
PeriodicTable::map_t PeriodicTable::bySymbol = _init_bysymbol();

} // namespace mrchem
